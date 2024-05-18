KEYFRAMES_DIR="feature_warp_results_group_random_pnp_SD_1.5/woman-running/A silver statue of a woman is running/attn_0.5_f_0.5_c_0.5_s_1.0/batch_size_8/50/img_ode"
ORIGINAL_DIR='data/woman-running' 
BLEND_DIR="$KEYFRAMES_DIR/blended"
rm -rf "$BLEND_DIR"
mkdir "$BLEND_DIR"
for i in {00000..00039}.png; do cp "$ORIGINAL_DIR/$i" "$BLEND_DIR"; done
cp "$BLEND_DIR/00039.png" "$BLEND_DIR/00040.png"
mkdir "$BLEND_DIR/img_ode"
cp "$KEYFRAMES_DIR"/*.png "$BLEND_DIR/img_ode"
mv "$BLEND_DIR/img_ode/00039.png" "$BLEND_DIR/img_ode/00040.png" 
CUDA_VISIBLE_DEVICES=0 python video_blend.py "$BLEND_DIR" \
  --beg 0 \
  --end 40 \
  --itv 8 \
  --key img_ode \
  --output  "$KEYFRAMES_DIR/output.mp4" \
  --fps 24.0 \

  