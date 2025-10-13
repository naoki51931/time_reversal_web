import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import uvicorn

# ============================================================
# 起動ログ
# ============================================================
BACKEND_MODE = os.getenv("BACKEND_MODE", "diffusers_full")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[StartUp] BACKEND_MODE={BACKEND_MODE}, DEVICE={DEVICE}")

# ============================================================
# FastAPI設定
# ============================================================
app = FastAPI(title="Time Reversal Web API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 静的ファイル配信
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# ============================================================
# Pipelineロード
# ============================================================
from models.pipeline_time_reversal import TimeReversalPipeline

pipeline = None
try:
    pipeline = TimeReversalPipeline(device=DEVICE)
    print(f"[Init] TimeReversalPipeline initialized ({DEVICE})")
except Exception as e:
    print(f"[WARN] Pipeline initialization failed: {e}")
    pipeline = None

# ============================================================
# 生成エンドポイント
# ============================================================
@app.post("/generate")
async def generate(
    image_1: UploadFile,
    image_2: UploadFile,
    frames: int = Form(2),
    t0: float = Form(5.0),
):
    print("[Start] /generate called")
    print(f"[Info] Parameters -> frames={frames}, t0={t0}")

    try:
        img1 = Image.open(image_1.file).convert("RGB")
        img2 = Image.open(image_2.file).convert("RGB")
        print("[Info] Input images loaded")
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"画像読み込み失敗: {e}"}, status_code=400)

    if pipeline is None:
        print("[Warn] Pipeline not initialized, running in mock mode.")
        output_paths = []
        for i in range(frames):
            out_path = OUTPUT_DIR / f"frame_{i:03d}.png"
            img1.save(out_path)
            output_paths.append(out_path)
    else:
        try:
            print("[Pipeline] Start processing (diffusers)")
            result = pipeline(img1, img2, M=frames, t0=t0)

            # ✅ diffusersのImagePipelineOutput対応
            if hasattr(result, "images"):
                images = result.images
            elif isinstance(result, dict) and "images" in result:
                images = result["images"]
            else:
                raise TypeError("Pipelineの戻り値が不明な形式です")

            output_paths = []
            for i, img in enumerate(images):
                out_path = OUTPUT_DIR / f"frame_{i:03d}.png"
                img.save(out_path)
                print(f"[Pipeline] Saved: {out_path}")
                output_paths.append(out_path)

        except Exception as e:
            print(f"[ERROR] Pipeline execution failed: {e}")
            return JSONResponse(
                {"status": "error", "message": f"Pipelineエラー: {e}"},
                status_code=500,
            )

    # ============================================================
    # URL生成
    # ============================================================
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    urls = [f"{base_url}/outputs/{Path(p).name}" for p in output_paths if Path(p).exists()]

    print(f"[Info] Generated {len(urls)} frames -> {urls}")

    return JSONResponse(
        {
            "status": "ok",
            "mode": BACKEND_MODE,
            "device": DEVICE,
            "frames_generated": len(urls),
            "image_urls": urls,
        }
    )

# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

