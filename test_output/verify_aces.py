import cv2
import numpy as np

# Inverse matrix to visualize ACES AP0 down to sRGB for basic verification
M_ACES_to_sRGB = np.linalg.inv(np.array([
    [0.4396329819, 0.3829886981, 0.1773783199],
    [0.0897764433, 0.8134394287, 0.0967841280],
    [0.0173623251, 0.1088480851, 0.8737895898]
], dtype=np.float32))

def verify_exr_aces(exr_path, out_preview_path):
    print(f"Loading {exr_path}...")
    
    # Read raw EXR (usually float32/float16 in OpenCV depending on build)
    # OpenCV IMREAD_UNCHANGED natively reads OpenEXR if built with OpenEXR support.
    try:
        img_aces = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except Exception as e:
        print("Failed to read via OpenCV, assuming it might not have OpenEXR:", e)
        return

    if img_aces is None:
        print("EXR loading failed. Check if file exists and OpenCV supports it.")
        return

    # Check properties
    print(f"Dimensions: {img_aces.shape[1]}x{img_aces.shape[0]}")
    print(f"Data Type: {img_aces.dtype}")
    print(f"Value Range [Linear ACES]: {img_aces.min():.4f} to {img_aces.max():.4f}")

    # It's BGR from opencv, convert to RGB
    if img_aces.shape[2] >= 3:
        img_aces_rgb = cv2.cvtColor(img_aces[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        img_aces_rgb = img_aces
        
    print("\n--- Verifying Math ---")
    # Convert Linear ACES AP0 back to Linear sRGB
    img_lin = np.dot(img_aces_rgb, M_ACES_to_sRGB.T)
    print(f"Unmapped [Linear sRGB] bounds: {img_lin.min():.4f} to {img_lin.max():.4f}")

    # Gamma correct perfectly back to sRGB for viewing validation
    img_srgb = np.where(img_lin <= 0.0031308, img_lin * 12.92, 1.055 * (np.power(np.maximum(img_lin, 0), 1/2.4)) - 0.055)
    
    # Clamp for saving preview
    img_srgb_clamped = np.clip(img_srgb * 255.0, 0, 255).astype(np.uint8)
    
    # BGR for saving
    img_bgr_clamped = cv2.cvtColor(img_srgb_clamped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_preview_path, img_bgr_clamped)
    
    print(f"\n✅ Validation complete! Generated {out_preview_path} to visually prove geometric fidelity.")

if __name__ == "__main__":
    verify_exr_aces("sample.exr", "preview_srgb.png")
