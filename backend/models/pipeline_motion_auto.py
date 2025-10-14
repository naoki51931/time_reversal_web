import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline


def generate_motion_frame(
    img_a: Image.Image,
    img_b: Image.Image,
    out_dir="outputs",
    strength=0.5,
    guidance_scale=6.0,
):
    """
    画像A→画像Bの動作を推定し、中間動作（顔や体の自然な変化）を生成する。
    Stable Diffusionを使い、動きの強さに応じて自動プロンプトを生成する。
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # --- 1️⃣ パイプライン初期化 (fp32固定) ---
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
        ).to(device)

        # ✅ SafetyChecker完全無効化
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

        # --- 4️⃣ 中間画像ブレンド ---
        blended = (arr_a * 0.5 + arr_b * 0.5).astype(np.uint8)
        mid_img = Image.fromarray(blended)

        # --- 5️⃣ Stable Diffusion生成 ---
        result = pipe(
            prompt=auto_prompt,
            image=mid_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=40,
        )

        # ✅ 戻り値の型チェック
        if not hasattr(result, "images") or not isinstance(result.images, list):
            print(f"[Motion] Unexpected result type: {type(result)}")
            return []

        if not result.images:
            print("[Motion] No images returned from pipeline.")
            return []

        out_path = os.path.join(out_dir, "motion_frame_00.png")
        img_out = result.images[0]

        # 万一 bool が来ても PIL.Image に変換
        if isinstance(img_out, bool):
            print("[Motion] Got bool instead of Image, fallback to blended frame.")
            img_out = mid_img

        img_out.save(out_path)
        print(f"[Motion] Saved: {out_path}")
        return [out_path]

    except Exception as e:
        print(f"[Motion] Diffusion pipeline error: {e}")
        return []

