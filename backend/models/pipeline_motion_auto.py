import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


def generate_motion_frame(
    img_a: Image.Image,
    img_b: Image.Image,
    out_dir="outputs",
    frames=5,
    strength=0.35,
    guidance_scale=18.5,
    prompt=None,
):
    """
    2枚の画像（A→B）の間を補間し、
    Stable Diffusion によって「創造的かつ自然な動作中間フレーム」を生成。

    特徴:
      - 創造性を戻し、静的な線形補間にランダム変化を追加
      - 差分強度に応じてプロンプトを自動生成
      - fp32固定（dtypeエラー回避）
      - frames数に応じて時系列的な滑らかさを維持
    """

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # === 1️⃣ Stable Diffusion Pipeline ===
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # ✅ fp32で安全実行
        ).to(device)

        # safety_checker無効化
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = dummy_checker

        # === 2️⃣ 差分解析 ===
        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)
        diff_intensity = float(np.mean(np.abs(arr_b - arr_a)))
        print(f"[Motion] Difference Intensity: {diff_intensity:.2f}")

        # === 3️⃣ プロンプト生成 ===
        if prompt:
            base_prompt = prompt
        elif diff_intensity < 5:
            base_prompt = "subtle facial motion, blinking, breathing, natural small movement, same subject"
        elif diff_intensity < 20:
            base_prompt = "smooth head or body motion, natural pose transition, same character, cinematic lighting"
        else:
            base_prompt = "dynamic body motion, action transition, realistic cinematic frame, same person continuity"

        print(f"[Motion] Motion Prompt: {base_prompt}")

        # === 4️⃣ フレーム生成 ===
        frame_paths = []
        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)

        for i in range(1, frames + 1):
            t = i / (frames + 1)

            # 🔁 線形補間 + 創造的ノイズ（シーンの想像力を刺激）
            blend = arr_a * (1 - t) + arr_b * t
            # ノイズの強度を時間位置に応じて可変（中間で最大）
            noise_intensity = (np.sin(np.pi * t) ** 2) * 20
            noise = np.random.normal(0, noise_intensity, blend.shape)
            blended = np.clip(blend + noise, 0, 255).astype(np.uint8)
            mid_img = Image.fromarray(blended)

            # 🎨 創造的な拡張プロンプト
            creative_prompt = (
                f"{base_prompt}, frame {i}/{frames}, lineart, smooth interpolation, cinematic atmosphere, "
                "fluid motion between poses, dynamic energy, consistent character identity"
            )

            print(f"[Motion] Frame {i}/{frames} → noise={noise_intensity:.1f}, strength={strength:.2f}")

            # === 5️⃣ Diffusion再生成 ===
            result = pipe(
                prompt=creative_prompt,
                image=mid_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=40,
            )

            if not hasattr(result, "images") or not result.images:
                print(f"[WARN] Frame {i}: invalid result, skipping.")
                continue

            img_out = result.images[0]
            if isinstance(img_out, bool):
                img_out = mid_img

            out_path = os.path.join(out_dir, f"motion_frame_{i:02d}.png")
            img_out.save(out_path)
            frame_paths.append(out_path)

        print(f"[Motion] Generated {len(frame_paths)} creative frames.")
        return frame_paths

    except Exception as e:
        print(f"[Motion] Diffusion pipeline error: {e}")
        return []

