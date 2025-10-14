import os
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# === 各パイプライン ===
from models.pipeline_time_reversal import TimeReversalPipeline as BasePipeline
from models.pipeline_time_reversal_lineart import TimeReversalPipeline as LineartPipeline
from models.pipeline_time_reversal_denoise import TimeReversalPipeline as DenoisePipeline
from models.pipeline_time_reversal_sampling import generate_midframes_trs  # ← diffusion_trs 対応

# =====================================================
# FastAPI 初期化
# =====================================================
app = FastAPI(title="Time Reversal Hybrid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# =====================================================
# 各パイプライン初期化
# =====================================================
pipe_normal = BasePipeline()
pipe_lineart = LineartPipeline()
pipe_denoise = DenoisePipeline()
pipe_lineart_denoise = DenoisePipeline(denoise_sigma=0.4)


# =====================================================
# 前処理関数群
# =====================================================
def enhance_contrast_and_sharpness(image: Image.Image) -> Image.Image:
    enhancer_c = ImageEnhance.Contrast(image)
    image = enhancer_c.enhance(2.4)
    enhancer_s = ImageEnhance.Sharpness(image)
    image = enhancer_s.enhance(3.0)
    return image


def enhance_contrast_cv(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 2.0, blur, -1.0, 0)
    return sharpened


def emphasize_lines(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 80)
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(edges, kernel, iterations=2)
    inv = cv2.bitwise_not(thick)
    edge_colored = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    gamma = 0.7
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
    edge_colored = cv2.LUT(edge_colored, look_up)
    blend = cv2.addWeighted(img, 0.7, edge_colored, 1.2, -10)
    return blend


# =====================================================
# 生成エンドポイント
# =====================================================
@app.post("/generate")
async def generate(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(8),
    t0: float = Form(5.0),
    lineart: bool = Form(False),
    denoise: bool = Form(False),
    diffusion_trs: bool = Form(False),
):
    print("[Start] /generate called")
    print(f"[Info] Params: frames={frames}, t0={t0}, lineart={lineart}, denoise={denoise}, diffusion_trs={diffusion_trs}")

    img1 = Image.open(BytesIO(await image_1.read())).convert("RGB")
    img2 = Image.open(BytesIO(await image_2.read())).convert("RGB")
    print("[Info] Input images loaded")

    mode = "normal"
    if lineart and denoise:
        mode = "lineart_denoise"
    elif lineart:
        mode = "lineart"
    elif denoise:
        mode = "denoise"
    elif diffusion_trs:
        mode = "diffusion_trs"

    print(f"[Run] Starting pipeline... (mode={mode})")

    try:
        # =====================================================
        # 通常 / 線画 / ノイズ / ハイブリッド
        # =====================================================
        if mode == "normal":
            result = pipe_normal(img1, img2, M=frames, t0=t0)

        elif mode == "lineart":
            result = pipe_lineart(img1, img2, M=frames, t0=t0)

        elif mode == "denoise":
            result = pipe_denoise(img1, img2, M=frames, t0=t0)

        elif mode == "lineart_denoise":
            print("[Hybrid] Applying contrast + sharpness + line emphasis...")
            img1 = enhance_contrast_and_sharpness(img1)
            img2 = enhance_contrast_and_sharpness(img2)
            res_denoise = pipe_denoise(img1, img2, M=frames, t0=t0)
            frame_paths = res_denoise.frames
            new_paths = []

            for fp in frame_paths:
                img = cv2.imread(fp)
                img = enhance_contrast_cv(img)
                img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                strong = emphasize_lines(img_denoised)
                out_path = fp.replace(".png", "_strongline.png")
                cv2.imwrite(out_path, strong)
                new_paths.append(out_path)

            return JSONResponse({
                "status": "ok",
                "frames_generated": len(new_paths),
                "image_urls": [f"/outputs/{os.path.basename(p)}" for p in new_paths],
                "debug": {"mode": "lineart_denoise_strong"}
            })

        # =====================================================
        # Stable Diffusion Time Reversal Sampling
        # =====================================================
        elif mode == "diffusion_trs":
            from models.pipeline_time_reversal_sampling import generate_midframes_trs
            print("[TRS] Running Time Reversal Sampling pipeline...")

            out_paths = generate_midframes_trs(
                img1, img2,
                frames=frames,          # Reactからの指定値を使用
                tau_step=35,
                num_steps=50,
                guidance_scale=5.0,
                prompt="smooth interpolation frame between two photos",
                out_dir="outputs"
            )

            return JSONResponse({
                "status": "ok",
                "frames_generated": len(out_paths),
                "image_urls": [f"/outputs/{os.path.basename(p)}" for p in out_paths],
                "debug": {"mode": "diffusion_trs", "frames": frames}
            })

        # =====================================================
        # 結果返却（旧パイプライン互換）
        # =====================================================
        return JSONResponse({
            "status": result.status,
            "frames_generated": result.frames_generated,
            "image_urls": [f"/outputs/{os.path.basename(p)}" for p in result.frames],
            "debug": result.debug,
        })

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

