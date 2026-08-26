import os
import sys
import time
import requests
import tarfile

API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is required")
MODEL_ENDPOINT = "realimposter/upscale-hdr"
TEST_MEDIA_URL = "https://replicate.delivery/pbxt/L00w2oB4z4mPElO8QjKxP62vB2Ew2k5I2qA3F0Y5wZkXmS9C/test_image.jpg" # Fallback test image
# Actually, I'll use a known image from replicate, or wait... I will just use a generic HD test image URL
TEST_MEDIA_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Shaqi_jrvej.jpg/1280px-Shaqi_jrvej.jpg"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

print(f"Fetching latest version for {MODEL_ENDPOINT}...")
meta_url = f"https://api.replicate.com/v1/models/{MODEL_ENDPOINT}"
meta_resp = requests.get(meta_url, headers=headers)
if meta_resp.status_code != 200:
    print(f"Failed to fetch model info: {meta_resp.status_code} {meta_resp.text}")
    sys.exit(1)

latest_version = meta_resp.json()["latest_version"]["id"]
print(f"Latest version: {latest_version}")

print(f"Starting Replicate API prediction for {MODEL_ENDPOINT}...")
data = {
    "version": latest_version,
    "input": {
        "media": TEST_MEDIA_URL,
        "preset": "Netflix Original (EXR Sequence)",
        "target_resolution": "Native 4x"
    }
}

url = "https://api.replicate.com/v1/predictions"
response = requests.post(url, headers=headers, json=data)
if response.status_code != 201:
    print(f"Failed to start prediction: {response.status_code} {response.text}")
    sys.exit(1)

prediction = response.json()
get_url = prediction["urls"]["get"]
print(f"Prediction started. ID: {prediction['id']}")

while True:
    time.sleep(3)
    resp = requests.get(get_url, headers=headers).json()
    status = resp["status"]
    print(f"Status: {status}...")
    if status == "succeeded":
        out_url = resp["output"]
        print(f"Success! Output: {out_url}")
        break
    elif status == "failed":
        print(f"Prediction failed: {resp.get('error')}")
        sys.exit(1)

print("Downloading tarball...")
r = requests.get(out_url)
with open("output.tar", "wb") as f:
    f.write(r.content)

print("Extracting tarball...")
with tarfile.open("output.tar", "r") as tar:
    tar.extractall("extracted_frames")

print("Validating first EXR...")
exr_files = [f for f in os.listdir("extracted_frames/exr_seq") if f.endswith(".exr")]
if not exr_files:
    print("No EXR files found!")
    sys.exit(1)

first_exr = os.path.join("extracted_frames/exr_seq", sorted(exr_files)[0])

import verify_aces
verify_aces.verify_exr_aces(first_exr, "aces_preview.png")
