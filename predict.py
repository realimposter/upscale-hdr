import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from typing import List
from pathlib import Path
import numpy as np
import cv2
import torch
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


# sRGB to Linear to ACES AP0 matrix (perfect precision)
M_sRGB_to_ACES = np.array([
    [0.4396329819, 0.3829886981, 0.1773783199],
    [0.0897764433, 0.8134394287, 0.0967841280],
    [0.0173623251, 0.1088480851, 0.8737895898]
], dtype=np.float32)

def srgb_to_aces_ap0(img_srgb: np.ndarray) -> np.ndarray:
    # 1. Decode sRGB to Linear
    img_lin = np.where(img_srgb <= 0.04045, img_srgb / 12.92, ((img_srgb + 0.055) / 1.055) ** 2.4)
    # 2. Multiply by ACES AP0 transform matrix
    img_aces = np.dot(img_lin, M_sRGB_to_ACES.T)
    # Clamp just above 0 to prevent negative noise
    return np.clip(img_aces, 0.0, None)

def extract_video_info(video_path: str):
    import json
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(res.stdout)["streams"][0]
    num, den = map(int, info["r_frame_rate"].split('/'))
    fps = num / den if den != 0 else 24.0
    return int(info["width"]), int(info["height"]), fps

def has_audio(video_path: str) -> bool:
    cmd = ["ffprobe", "-i", video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return len(res.stdout) > 0


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the models into memory"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize x4plus model (standard RealESRGAN)
        model_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler_x4 = RealESRGANer(
            scale=4,
            model_path="weights/RealESRGAN_x4plus.pth",
            dni_weight=None,
            model=model_x4,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0
        )

    def predict(
        self,
        media: CogPath = Input(description="Input video or image file"),
        preset: str = Input(
            description="Export Preset Standard",
            choices=["Netflix Original (EXR Sequence)", "VFX Compositing (ProRes 4444)"],
            default="Netflix Original (EXR Sequence)"
        ),
        target_resolution: str = Input(
            description="Output resolution constraint. Real-ESRGAN will scale 4x, then Lanczos downsample to fit.",
            choices=["Native 4x", "DCI 4K (4096x2304)", "UHD 4K (3840x2160)"],
            default="DCI 4K (4096x2304)"
        ),
    ) -> CogPath:
        
        work_dir = Path(tempfile.mkdtemp())
        input_path = str(media)
        
        # Is it video or image?
        is_video = input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
        
        print("[predict] Extracting media...")
        frames_dir = work_dir / "frames_in"
        frames_dir.mkdir()
        
        audio_path = work_dir / "audio.wav"
        has_aud = False
        
        if is_video:
            w, h, fps = extract_video_info(input_path)
            subprocess.run(["ffmpeg", "-i", input_path, "-qscale:v", "1", "-qmin", "1", f"{frames_dir}/%06d.png"], check=True)
            if has_audio(input_path):
                subprocess.run(["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s24le", "-ar", "48000", str(audio_path)], check=True)
                has_aud = True
        else:
            shutil.copy(input_path, frames_dir / "000001.png")
            fps = 24.0
            
        frame_files = sorted(frames_dir.glob("*.png"))
        print(f"[predict] Processing {len(frame_files)} frames...")
        
        upsampler = self.upsampler_x4
        
        out_w, out_h = None, None
        if target_resolution == "DCI 4K (4096x2304)":
            out_w, out_h = 4096, 2304
        elif target_resolution == "UHD 4K (3840x2160)":
            out_w, out_h = 3840, 2160

        # FFmpeg subprocess to pipe raw 32-bit ACES frames into encoding
        out_file = work_dir / "output.mov" if "ProRes" in preset else work_dir / "output.tar"
        
        pipe_cmd = None
        ff_proc = None
        
        # Start processing
        for i, f_path in enumerate(frame_files):
            # Load BGR
            img = cv2.imread(str(f_path), cv2.IMREAD_COLOR)
            
            # Upscale
            try:
                output, _ = upsampler.enhance(img, outscale=4)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                output = img
            
            # Lanczos resize if constrained
            if out_w is not None and out_h is not None:
                # Maintain aspect ratio? The prompt asked for exactly this constraint usually, but let's force fit.
                output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                out_w, out_h = output.shape[1], output.shape[0]

            # Convert BGR [0,255] -> RGB [0,1] float32
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Linearize and map to ACES AP0
            aces_ap0_frame = srgb_to_aces_ap0(output_rgb)
            
            # Initialize FFmpeg pipe on first frame
            if ff_proc is None:
                if "ProRes" in preset:
                    audio_args = ["-i", str(audio_path), "-c:a", "pcm_s24le"] if has_aud else []
                    pipe_cmd = [
                        "ffmpeg", "-y", "-loglevel", "warning",
                        "-f", "rawvideo", "-pix_fmt", "gbrpf32le",
                        "-s", f"{out_w}x{out_h}", "-r", str(fps),
                        "-i", "pipe:0"
                    ] + audio_args + [
                        "-c:v", "prores_ks", "-profile:v", "4444xq",
                        "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
                        "-color_primaries", "smpte431", # Closest tag for wide gamuts in some containers
                        str(out_file)
                    ]
                else:
                    # Netflix EXR Sequence Pipeline using FFmpeg PIZ
                    # FFMpeg natively outputs EXR sequences. We tar them up!
                    seq_dir = work_dir / "exr_seq"
                    seq_dir.mkdir()
                    pipe_cmd = [
                        "ffmpeg", "-y", "-loglevel", "warning",
                        "-f", "rawvideo", "-pix_fmt", "gbrpf32le",
                        "-s", f"{out_w}x{out_h}", "-r", str(fps),
                        "-i", "pipe:0",
                        "-c:v", "exr", "-compression", "piz",
                        f"{seq_dir}/%06d.exr"
                    ]
                print(f"[predict] Starting FFmpeg export: {' '.join(pipe_cmd)}")
                ff_proc = subprocess.Popen(pipe_cmd, stdin=subprocess.PIPE)

            # Write raw float32 frame to ffmpeg!
            ff_proc.stdin.write(aces_ap0_frame.tobytes())

        # Close pipe and finish encoding
        ff_proc.stdin.close()
        ff_proc.wait()
        
        # If EXR, Tar it up
        if "EXR" in preset:
            print("[predict] Archiving EXR sequence...")
            # We copy audio if it existed
            if has_aud:
                shutil.copy(str(audio_path), str(work_dir / "exr_seq" / "audio.wav"))
            with tarfile.open(str(out_file), "w") as tar:
                tar.add(str(work_dir / "exr_seq"), arcname="netflix_aces_ap0_sequence")

        return CogPath(out_file)
