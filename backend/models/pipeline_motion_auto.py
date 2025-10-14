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
    strength=0.45,
    guidance_scale=6.5,
):
    """
    画像A→画像Bの動作を推定し、frames数に応じて中間動作フレームを生成する。
    Stable Diffusionを使い、動きの強さに応じて自動プロンプトを生成する。
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # --- 1️⃣ パイプライン初期化 ---
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
        ).to(device)

        # ✅ SafetyChecker無効化
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = dummy_checker

        # --- 2️⃣ 差分解析 ---
        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)
        diff_intensity = float(np.mean(np.abs(arr_b - arr_a)))

        print(f"[Motion] Average difference intensity: {diff_intensity:.2f}")

        # --- 3️⃣ 自動プロンプト生成 ---
        if diff_intensity < 5:
            auto_prompt = "subtle facial motion, blinking, breathing, natural small movement, consistent face"
        elif diff_intensity < 20:
            auto_prompt = "smooth facial or upper body motion, natural pose transition, same person, soft movement"
        else:
            auto_prompt = "dynamic body motion, mid-action frame, realistic transition between two poses, consistent person, balanced lighting"

        print(f"[Motion] Auto prompt: {auto_prompt}")

        # --- 4️⃣ 中間フレーム生成 ---
        arr_a = np.array(img_a).astype(np.float32)
        arr_b = np.array(img_b).astype(np.float32)

        frame_paths = []
        for i in range(1, frames + 1):
            t = i / (frames + 1)
            blended = (arr_a * (1 - t) + arr_b * t).astype(np.uint8)
            mid_img = Image.fromarray(blended)

            print(f"[Motion] Generating frame {i}/{frames} (blend={t:.2f})")

            # --- Diffusionで自然な補正 ---
            result = pipe(
                prompt=auto_prompt,
                image=mid_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=40,
            )

            if not hasattr(result, "images") or not result.images:
                print(f"[WARN] Frame {i}: invalid result, skipping.")
                continue

            out_path = os.path.join(out_dir, f"motion_frame_{i:02d}.png")
            img_out = result.images[0]

            # Fallback: bool → PIL fallback
            if isinstance(img_out, bool):
                print(f"[WARN] Frame {i}: bool image fallback.")
                img_out = mid_img

            img_out.save(out_path)
            frame_paths.append(out_path)

        print(f"[Motion] Generated {len(frame_paths)} frames.")
        return frame_paths

    except Exception as e:
        print(f"[Motion] Diffusion pipeline error: {e}")
        return []

