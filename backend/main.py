import os
import uuid
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageEnhance
import cv2

# --- 各パイプライン読み込み ---
from models.pipeline_time_reversal import TimeReversalPipeline as BasePipeline
from models.pipeline_time_reversal_lineart import TimeReversalPipeline as LineartPipeline
from models.pipeline_time_reversal_denoise import TimeReversalPipeline as DenoisePipeline
from models.pipeline_time_reversal_sampling import generate_midframes_trs
from models.pipeline_motion_auto import generate_motion_frame  # ✅ 動作補間のみ使用

# ==============================================================
# FastAPI アプリ設定
# ==============================================================

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

# ==============================================================
# パイプライン初期化
# ==============================================================

pipe_normal = BasePipeline()
pipe_lineart = LineartPipeline()
pipe_denoise = DenoisePipeline()
pipe_lineart_denoise = DenoisePipeline(denoise_sigma=0.4)


def str2bool(v):
    """文字列→bool変換"""
    return str(v).lower() in ("1", "true", "yes", "on")


# ==============================================================
# メインAPI
# ==============================================================

@app.post("/generate")
async def generate(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(8),
    t0: float = Form(0.0),
    lineart: str = Form("false"),
    denoise: str = Form("false"),
    diffusion_trs: str = Form("false"),
    motion: str = Form("false"),  # ✅ 瞬き削除済み
):
    lineart = str2bool(lineart)
    denoise = str2bool(denoise)
    diffusion_trs = str2bool(diffusion_trs)
    motion = str2bool(motion)

    print(f"\n[Start] /generate called")
    print(f"[Info] Params: frames={frames}, t0={t0}, lineart={lineart}, denoise={denoise}, diffusion_trs={diffusion_trs}, motion={motion}")

    # 画像ロード
    img1 = Image.open(BytesIO(await image_1.read())).convert("RGB")
    img2 = Image.open(BytesIO(await image_2.read())).convert("RGB")

    session_id = uuid.uuid4().hex[:8]

    # モード決定
    mode = "normal"
    if motion:
        mode = "motion"
    elif lineart and denoise:
        mode = "lineart_denoise"
    elif lineart:
        mode = "lineart"
    elif denoise:
        mode = "denoise"
    elif diffusion_trs:
        mode = "diffusion_trs"

    print(f"[Mode] Selected -> {mode}")

    try:
        # ======================================================
        # 各モード
        # ======================================================

        if mode == "normal":
            result = pipe_normal(img1, img2, M=frames, t0=t0)

        elif mode == "lineart":
            result = pipe_lineart(img1, img2, M=frames, t0=t0)

        elif mode == "denoise":
            result = pipe_denoise(img1, img2, M=frames, t0=t0)

        elif mode == "lineart_denoise":
            print("[Hybrid] Applying lineart + denoise enhancement...")
            img1 = ImageEnhance.Contrast(img1).enhance(2.2)
            img2 = ImageEnhance.Contrast(img2).enhance(2.2)
            res_denoise = pipe_denoise(img1, img2, M=frames, t0=t0)
            frame_paths = res_denoise.frames
            new_paths = []

            for i, fp in enumerate(frame_paths):
                img = cv2.imread(fp)
                blur = cv2.GaussianBlur(img, (0, 0), 2.0)
                sharp = cv2.addWeighted(img, 2.0, blur, -1.0, 0)
                out_path = os.path.join("outputs", f"{session_id}_hybrid_{i:02d}.png")
                cv2.imwrite(out_path, sharp)
                new_paths.append(out_path)

            return JSONResponse({
                "status": "ok",
                "frames_generated": len(new_paths),
                "image_urls": [f"/outputs/{os.path.basename(p)}" for p in new_paths],
                "debug": {"mode": "lineart_denoise"},
            })

        elif mode == "diffusion_trs":
            print("[TRS] Running Time Reversal Sampling pipeline...")
            out_paths = generate_midframes_trs(
                img1, img2,
                frames=frames,
                tau_step=35,
                num_steps=50,
                guidance_scale=5.0,
                prompt="smooth interpolation between two photos, same subject, consistent lighting",
                out_dir="outputs",
            )

            if not isinstance(out_paths, list):
                print("[WARN] TRS returned non-list result.")
                out_paths = []

            renamed_paths = []
            for i, p in enumerate(out_paths):
                new_p = os.path.join("outputs", f"{session_id}_trs_{i:02d}.png")
                os.rename(p, new_p)
                renamed_paths.append(new_p)

            return JSONResponse({
                "status": "ok",
                "frames_generated": len(renamed_paths),
                "image_urls": [f"/outputs/{os.path.basename(p)}" for p in renamed_paths],
                "debug": {"mode": "diffusion_trs"},
            })

        elif mode == "motion":
            print("[Motion] Generating dynamic motion interpolation frame...")
            out_paths = generate_motion_frame(img1, img2, out_dir="outputs")

            # ✅ boolなどが返っても落ちない安全策
            if not isinstance(out_paths, list):
                print("[ERROR] Motion pipeline did not return list; coercing to empty list.")
                out_paths = []

            renamed_paths = []
            for i, p in enumerate(out_paths):
                new_p = os.path.join("outputs", f"{session_id}_motion_{i:02d}.png")
                os.rename(p, new_p)
                renamed_paths.append(new_p)

            return JSONResponse({
                "status": "ok",
                "frames_generated": len(renamed_paths),
                "image_urls": [f"/outputs/{os.path.basename(p)}" for p in renamed_paths],
                "debug": {"mode": "motion_auto"},
            })

        # ======================================================
        # 通常出力
        # ======================================================
        return JSONResponse({
            "status": result.status,
            "frames_generated": result.frames_generated,
            "image_urls": [f"/outputs/{os.path.basename(p)}" for p in result.frames],
            "debug": {"mode": mode},
        })

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

