#!/bin/bash
set -e

# Real-ESRGAN HDR Option B Deployment Script

echo "═══════════════════════════════════════════════════════════"
echo "  📋 Next Steps to Deploy via GitHub Integration"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Since Docker is not running on this Mac, we will use GitHub to auto-build."
echo ""
echo "  1. Create a new GitHub repo in your browser named 'upscale-hdr'"
echo "  2. Push this folder to that repo:"
echo "       git remote add origin https://github.com/YOUR_USERNAME/upscale-hdr.git"
echo "       git branch -M main"
echo "       git push -u origin main"
echo ""
echo "  3. Go to https://replicate.com/create"
echo "  4. Create model 'upscale-hdr'"
echo "  5. Under 'Settings' → 'GitHub', connect the repo"
echo "  6. Replicate will automatically build and deploy the container in the cloud!"
echo ""
