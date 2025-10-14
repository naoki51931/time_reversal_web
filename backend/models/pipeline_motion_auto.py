import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import cv2

def generate_motion_frame(
    img_a: Image.Image,
    img_b: Image.Image,
    out_dir="outputs",
    strength=0.5,
    guidance_scale=6.0,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    # --- 1️⃣ 差分解析 ---
    arr_a = np.array(img_a).astype(np.float32)
    arr_b = np.array(img_b).astype(np.float32)
    diff = np.abs(arr_b - arr_a).mean(axis=(0,1))

    diff_intensity = np.mean(diff)
    print(f"[Motion] Average difference intensity: {diff_intensity:.2f}")

    # --- 2️⃣ 自動プロンプト決定 ---
    if diff_intensity < 5:
        auto_prompt = "subtle facial motion, blinking, breathing, natural small movement, consistent face"
    elif diff_intensity < 20:
        auto_prompt = "smooth facial or upper body motion, natural pose transition, same person, soft movement"
    else:
        auto_prompt = "dynamic body motion, mid-action frame, realistic transition between two poses, consistent person, balanced lighting"

    print(f"[Motion] Auto prompt: {auto_prompt}")

    # --- 3️⃣ 中間画像作成 ---
    blended = (arr_a * 0.5 + arr_b * 0.5).astype(np.uint8)
    mid_img = Image.fromarray(blended)

    # --- 4️⃣ Stable Diffusion生成 ---
    result = pipe(
        prompt=auto_prompt,
        image=mid_img,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )

    out_path = os.path.join(out_dir, "motion_frame_00.png")
    result.images[0].save(out_path)
    return [out_path]

