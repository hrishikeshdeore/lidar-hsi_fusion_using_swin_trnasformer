I'm working on a semantic segmentation project using hyperspectral imaging (HSI) and LiDAR data for land cover classification. The data is provided in a .mat file and contains:

hsi: a 3D array (H, W, C) representing hyperspectral images with 144 spectral channels,

lidar: a 2D elevation map with one channel,

train and test masks: 2D arrays labeling the land cover classes.

🔧 What I'm trying to do:
Normalize and preprocess the data.

Extract patches from both HSI and LiDAR using a sliding window.

Pass patches through two separate Swin Transformer backbones (one for HSI and one for LiDAR).

Fuse their deepest features via a Conv2D layer.

Decode the fused features into segmentation maps.

Train using a CrossEntropyLoss, and evaluate on a test set.

🧠 Model:
I use timm's swin_tiny_patch4_window7_224 model for both HSI (in_chans=144) and LiDAR (in_chans=1).

The final fusion step uses torch.cat([hf, lf], dim=1) followed by a Conv2d.


pgsql
Copy
Edit
RuntimeError: Given groups=1, weight of size [96, 192, 1, 1], expected input[8, 112, 56, 96] to have 192 channels, but got 112 channels instead
This seems to indicate the number of channels from the backbone outputs is not what I expected. I'm not sure if:

I’m using the wrong index for the Swin feature outputs.

My assumptions about the output channel dimensions are wrong.

The timm.create_model(..., features_only=True) API changed behavior.

Can you help me:

Identify why this mismatch happens?

Correct the fusion logic accordingly?

Optionally validate that my patch extraction and upsampling steps make sense for segmentation?

Let me know if you need specific parts of the code or more context!

