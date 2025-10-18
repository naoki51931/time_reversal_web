import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
import uuid
from dotenv import load_dotenv

# =========================================
# 🌐 環境変数の読み込み
# =========================================
load_dotenv()

MODEL_ID = os.getenv("MOTION_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DTYPE = os.getenv("MOTION_DTYPE", "fp16")
STRENGTH_DEFAULT = float(os.getenv("MOTION_DEFAULT_STRENGTH", 0.4))
GUIDANCE_DEFAULT = float(os.getenv("MOTION_DEFAULT_GUIDANCE", 15.0))
IMG_SIZE = int(os.getenv("MOTION_IMAGE_SIZE", 512))
STEPS = int(os.getenv("MOTION_INFERENCE_STEPS", 40))

ENABLE_XFORMERS = os.getenv("MOTION_ENABLE_XFORMERS", "true").lower() == "true"
ENABLE_TILING = os.getenv("MOTION_ENABLE_VAE_TILING", "true").lower() == "true"


# =========================================
# 🧩 パイプライン構築
# =========================================
def _build_pipe():
    torch_dtype = torch.float16 if DTYPE == "fp16" else torch.float32
    print(f"[Motion] 🚀 Loading model: {MODEL_ID} ({DTYPE})")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    if ENABLE_XFORMERS:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    if ENABLE_TILING:
        pipe.vae.enable_tiling()

    return pipe


# =========================================
# 🖼️ 自動リサイズ（アスペクト維持で1400px上限）
# =========================================
def auto_resize(img1: Image.Image, img2: Image.Image, max_size=1400):
    w1, h1 = img1.size
    w2, h2 = img2.size
    min_w, min_h = min(w1, w2), min(h1, h2)

    scale = min(max_size / max(min_w, min_h), 1.0)
    new_size = (int(min_w * scale), int(min_h * scale))

    return img1.resize(new_size, Image.LANCZOS), img2.resize(new_size, Image.LANCZOS)


# =========================================
# 🧭 サイズを強制一致させる補助関数
# =========================================
def match_size(base: Image.Image, target: Image.Image) -> Image.Image:
    """targetのサイズをbaseに合わせる"""
    if target.size != base.size:
        return target.resize(base.size, Image.LANCZOS)
    return target


# =========================================
# 🌀 モーション補間
# =========================================
def generate_motion_interpolation(
    img1: Image.Image,
    img2: Image.Image,
    M: int = 3,
    strength: float = STRENGTH_DEFAULT,
    guidance_scale: float = GUIDANCE_DEFAULT,
    out_dir: str = "outputs",
    t0: float = 0.0,
):
    """
    動作補間: 前フレームを次の入力に使って順次生成
    t0: 補間開始位置 (0.0〜1.0)
    out_dir: 出力ディレクトリ
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = None
    try:
        pipe = _build_pipe()

        # === 入力リサイズ ===
        img1, img2 = auto_resize(img1, img2, max_size=1400)
        img1 = img1.convert("RGB")
        img2 = img2.convert("RGB")

        os.makedirs(out_dir, exist_ok=True)

        base_prompt = "dynamic motion, smooth transition, natural lighting, anime-style"
        frames = []

        prev_image = img1  # 最初はAを入力
        print(f"[Motion] Start motion interpolation (M={M}, t0={t0:.2f})")

        for i in range(M):
            # t0を補間の開始点として加算
            t = t0 + ((i + 1) / (M + 1)) * (1.0 - t0)
            t = min(max(t, 0.0), 1.0)  # 安全にクリップ

            prev_image = match_size(img2, prev_image)
            blend = Image.blend(prev_image, img2, alpha=t)
            output_name = os.path.join(
                out_dir, f"motion_frame_{i:03d}_{uuid.uuid4().hex[:8]}.png"
            )

            print(
                f"[Frame {i+1}/{M}] t={t:.3f}, steps={STEPS}, strength={strength}, out={output_name}"
            )

            try:
                result = pipe(
                    prompt=base_prompt,
                    image=blend,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=STEPS,
                )

                out_img = result.images[0]
                out_img.save(output_name)
                frames.append(output_name)
                prev_image = out_img

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"[Motion] ❌ OOM on frame {i+1}, retrying with smaller settings..."
                    )
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    strength = max(strength * 0.8, 0.1)
                    guidance_scale = max(guidance_scale * 0.9, 5.0)
                    continue
                else:
                    print(f"[Motion] ❌ Error on frame {i+1}: {e}")
                    continue

        print(f"[Motion] ✅ Finished. Generated {len(frames)}/{M} frames.")
        return {"status": "ok", "frames": frames, "generated": len(frames)}

    finally:
        if pipe:
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

