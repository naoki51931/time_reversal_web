import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

from models.pipeline_time_reversal import TimeReversalPipeline as NormalPipeline
from models.pipeline_time_reversal_lineart import TimeReversalPipeline as LineartPipeline

app = FastAPI()

# CORS許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 出力ディレクトリを静的公開
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# パイプライン初期化
normal_pipeline = NormalPipeline(device="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu")
lineart_pipeline = LineartPipeline(device="cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu")

@app.post("/generate")
async def generate(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(3),
    t0: float = Form(5.0),
    lineart: bool = Form(False),
):
    print("[Start] /generate called")
    print(f"[Info] Parameters -> frames={frames}, t0={t0}, lineart={lineart}")

    try:
        img1 = Image.open(BytesIO(await image_1.read()))
        img2 = Image.open(BytesIO(await image_2.read()))
        print("[Info] Input images loaded")

        # パイプライン選択
        pipeline = lineart_pipeline if lineart else normal_pipeline
        print(f"[Run] Starting pipeline inference... (mode={'lineart' if lineart else 'normal'})")

        result = pipeline(img1, img2, M=frames, t0=t0)
        if not hasattr(result, "frames"):
            raise ValueError("Pipelineの戻り値が不明な形式です")

        base_url = "http://43.207.92.186:8000"  # or 外部IPに変更
        urls = [f"{base_url}/outputs/{os.path.basename(p)}" for p in result.frames]

        return JSONResponse(
            {
                "status": "ok",
                "mode": "lineart" if lineart else "normal",
                "frames_generated": result.frames_generated,
                "image_urls": urls,
                "debug": result.debug,
            }
        )

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

