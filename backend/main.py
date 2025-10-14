# backend/main.py
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from PIL import Image
import torch
import os
import traceback

from models.pipeline_time_reversal import TimeReversalPipeline, _to_serializable

app = FastAPI()

# === CORS設定 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === グローバル設定 ===
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === パイプライン初期化 ===
print(f"[StartUp] BACKEND_MODE=diffusers_full, DEVICE={DEVICE}")
pipeline = TimeReversalPipeline(device=DEVICE)
print("[Init] Pipeline loaded.")


@app.post("/generate")
async def generate(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(2),
    t0: float = Form(5.0),
):
    """2枚の画像を受け取り、中間フレームを生成して返す"""
    print("[Start] /generate called")
    print(f"[Info] Parameters -> frames={frames}, t0={t0}")

    try:
        img1 = Image.open(image_1.file).convert("RGB")
        img2 = Image.open(image_2.file).convert("RGB")
        print("[Info] Input images loaded")

        # === パイプライン呼び出し ===
        print("[Run] Starting pipeline inference...")
        result = pipeline(img1, img2, M=frames, t0=t0)

        # === 戻り値の型に応じて処理 ===
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "to_dict"):
            data = result.to_dict()
        elif hasattr(result, "__dict__"):
            data = vars(result)
        else:
            raise TypeError("Pipelineの戻り値が不明な形式です")

        # === フレームURL構築 ===
        base_url = "http://43.207.92.186:8000"  # ← 本番はIPに合わせて変更
        output_paths = data.get("frames", [])
        image_urls = []
        for p in output_paths:
            path = Path(p)
            if path.exists():
                image_urls.append(f"{base_url}/outputs/{path.name}")
            else:
                print(f"[WARN] Missing file: {p}")

        response = {
            "status": data.get("status", "ok"),
            "frames_generated": data.get("frames_generated", len(image_urls)),
            "image_urls": image_urls,
            "debug": {k: _to_serializable(v) for k, v in data.get("debug", {}).items()},
        }

        print(f"[Info] Generated {len(image_urls)} frame(s)")
        return JSONResponse(content=response)

    except Exception as e:
        print("[ERROR] Pipeline execution failed:", str(e))
        traceback.print_exc()
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )


# === 出力画像を返すルート ===
@app.get("/outputs/{filename}")
async def get_output_image(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path)


@app.get("/")
def root():
    return {"status": "ok", "mode": "diffusers_full", "device": DEVICE}

