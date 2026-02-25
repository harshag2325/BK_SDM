from diffusers import StableDiffusionPipeline
import torch
import os
import matplotlib.pyplot as plt

model_dir = "/kaggle/working/customer_SD_model/results/toy_bk_small"

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16  # faster on GPU
)

pipe = pipe.to("cuda")

prompt = "chair"

# 🔥 generate in safe batches (prevents CUDA OOM)
all_images = []
batch_size = 5
total_images = 5

for _ in range(total_images // batch_size):
    result = pipe(
        prompt,
        num_images_per_prompt=batch_size,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    all_images.extend(result.images)

print(f"Generated {len(all_images)} images")

# ✅ save images
save_dir = "/kaggle/working/customer_SD_model/generated"
os.makedirs(save_dir, exist_ok=True)

for i, img in enumerate(all_images):
    img.save(f"{save_dir}/chair_{i:02d}.png")

print("Saved all images!")

# ✅ show grid
cols = 6
rows = (len(all_images) + cols - 1) // cols

plt.figure(figsize=(15, 10))
for i, img in enumerate(all_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()
plt.show()
