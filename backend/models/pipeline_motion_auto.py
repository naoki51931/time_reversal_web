import os
import gc
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# ===== .env 読み込み =====
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ======== ヘルパ関数群 ========
def _get_dtype_from_env(env_key: str, default: str = "fp32"):
    val = (os.getenv(env_key) or default).lower()
    if val in ("fp16", "float16", "half"):
        return torch.float16
    if val in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _bool_env(key: str, default: bool = False):
    return (os.getenv(key) or str(default)).lower() in ("1", "true", "yes", "on")


def _int_env(key: str, default: int):
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def _float_env(key: str, default: float):
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


# ======== パイプライン構築 ========
def _build_pipe(model_id: str, torch_dtype: torch.dtype, device: str):
    """Hugging Face モデルを読み込み、.env の設定を反映して最適化"""
    from huggingface_hub import login

    token = os.getenv("HUGGINGFACE_TOKEN", None)
    if token:
        try:
            login(token=token)
            print("[Auth] Hugging Face login success ✅")
        except Exception as e:
            print(f"[Auth] Login failed: {e}")

    revision = os.getenv("MOTION_MODEL_REVISION", None)
    use_safetensors = _bool_env("MOTION_USE_SAFETENSORS", False)

    print(f"[Build] Loading model: {model_id} (revision={revision})")
    print(f"[Build] safetensors={use_safetensors}, dtype={torch_dtype}, device={device}")

    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            safety_checker=None,
            use_safetensors=use_safetensors,
            use_auth_token=token,
        ).to(device)
    except Exception as e:
        # fallback: from_single_file
        if os.path.exists(model_id) and model_id.endswith((".ckpt", ".safetensors")):
            print(f"[Build:Fallback] Loading single file model: {model_id}")
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
            ).to(device)
        else:
            raise e

    # === メモリ最適化 ===
    if _bool_env("MOTION_ENABLE_XFORMERS", True):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Opt] xFormers attention enabled ✅")
        except Exception as e:
            print(f"[Opt] xFormers unavailable: {e}")

    if _bool_env("MOTION_ENABLE_ATTENTION_SLICING", True):
        pipe.enable_attention_slicing()
        print("[Opt] Attention slicing ✅")

    if _bool_env("MOTION_ENABLE_VAE_TILING", True):
        pipe.enable_vae_tiling()
        print("[Opt] VAE tiling ✅")

    return pipe


# ======== メイン関数 ========
def generate_motion_frame(
    img_a: Image.Image,
    img_b: Image.Image,
    out_dir="outputs",
    frames=5,
    strength=None,
    guidance_scale=None,
    prompt=None,
):
    """
    2枚の画像の間を "創造的に" 補間し、線画・スケッチ風の中間フレームを生成。
    .env でパラメータ（モデル・dtype・サイズなど）を制御可能。
    """
    os.makedirs(out_dir, exist_ok=True)

    # === .env 設定を読み取り ===
    model_id = os.getenv("MOTION_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    dtype = _get_dtype_from_env("MOTION_DTYPE", "fp32")
    size = _int_env("MOTION_IMAGE_SIZE", 512)
    steps = _int_env("MOTION_INFERENCE_STEPS", 30)
    default_strength = _float_env("MOTION_DEFAULT_STRENGTH", 0.55)
    default_guidance = _float_env("MOTION_DEFAULT_GUIDANCE", 7.5)

    if strength is None:
        strength = default_strength
    if guidance_scale is None:
        guidance_scale = default_guidance

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Motion] Model={model_id}, dtype={str(dtype).split('.')[-1]}, device={device}")
    print(f"[Motion] size={size}, steps={steps}, strength={strength}, guidance={guidance_scale}")

    try:
        if device == "cuda":
            torch.cuda.empty_cache()

        pipe = _build_pipe(model_id=model_id, torch_dtype=dtype, device=device)

        img_a = img_a.convert("RGB").resize((size, size), Image.LANCZOS)
        img_b = img_b.convert("RGB").resize((size, size), Image.LANCZOS)

        arr_a = np.array(img_a, dtype=np.float32)
        arr_b = np.array(img_b, dtype=np.float32)

        diff_intensity = float(np.mean(np.abs(arr_b - arr_a)))
        print(f"[Motion] Difference Intensity: {diff_intensity:.2f}")

        # ==== 自動プロンプト生成 ====
        if prompt:
            base_prompt = prompt
        elif diff_intensity < 5:
            base_prompt = "clean line art, subtle motion, sketch style, consistent outline"
        elif diff_intensity < 20:
            base_prompt = "smooth motion between two anime frames, line art, sketch aesthetic"
        else:
            base_prompt = "dynamic line art motion, expressive sketch, doodle style, consistent character"

        print(f"[Motion] Base Prompt: {base_prompt}")

        frame_paths = []

        # ==== フレーム生成 ====
        for i in range(1, frames + 1):
            t = i / (frames + 1)
            blend = arr_a * (1 - t) + arr_b * t

            # ノイズを中間点で強くする（創造的）
            noise_strength = (np.sin(np.pi * t) ** 2) * 20.0
            noisy = np.clip(
                blend + np.random.normal(0, noise_strength, blend.shape),
                0,
                255
            ).astype(np.uint8)

            mid_img = Image.fromarray(noisy)

            creative_prompt = (
                f"{base_prompt}, frame {i}/{frames}, detailed clean lines, "
                "smooth motion, consistent lighting, artistic sketch, "
                "creative line weight balance"
            )

            print(f"[Motion] Frame {i}/{frames} → noise={noise_strength:.1f}")

            result = pipe(
                prompt=creative_prompt,
                image=mid_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            )

            if not hasattr(result, "images") or not result.images:
                print(f"[WARN] Frame {i}: invalid result, fallback to blended image.")
                img_out = mid_img
            else:
                img_out = result.images[0]

            out_path = os.path.join(out_dir, f"motion_frame_{i:02d}.png")
            img_out.save(out_path)
            frame_paths.append(out_path)

        print(f"[Motion] Generated {len(frame_paths)} frames successfully ✅")
        return frame_paths

    except Exception as e:
        print(f"[Motion] Diffusion pipeline error: {e}")
        return []
    finally:
        try:
            del pipe
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

