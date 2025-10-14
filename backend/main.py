import os
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from models.pipeline_time_reversal import TimeReversalPipeline as NormalPipeline
from models.pipeline_time_reversal_lineart import TimeReversalPipeline as LineartPipeline
from models.pipeline_time_reversal_denoise import TimeReversalPipeline as DenoisePipeline

app = FastAPI(title="Time Reversal WebAPI")
from fastapi.middleware.cors import CORSMiddleware

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では ["http://43.207.92.186:3000"] に限定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 出力フォルダ
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# 各パイプライン初期化
print("[Startup] Initializing pipelines...")
normal_pipeline = NormalPipeline(device="cuda" if os.environ.get("DEVICE") != "cpu" else "cpu")
lineart_pipeline = LineartPipeline(device="cuda" if os.environ.get("DEVICE") != "cpu" else "cpu")
denoise_pipeline = DenoisePipeline(device="cuda" if os.environ.get("DEVICE") != "cpu" else "cpu")
print("[Startup] All pipelines ready.")


@app.post("/generate")
async def generate(
    request: Request,
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(3),
    t0: float = Form(5.0),
    lineart: bool = Form(False),
    denoise: bool = Form(False),
):
    print("[Start] /generate called")
    print(f"[Info] Parameters -> frames={frames}, t0={t0}, lineart={lineart}, denoise={denoise}")

    try:
        img1 = Image.open(BytesIO(await image_1.read()))
        img2 = Image.open(BytesIO(await image_2.read()))
        print("[Info] Input images loaded")

        # モード選択
        if lineart:
            pipeline = lineart_pipeline
            mode = "lineart"
        elif denoise:
            pipeline = denoise_pipeline
            mode = "denoise"
        else:
            pipeline = normal_pipeline
            mode = "normal"

        print(f"[Run] Starting pipeline inference... (mode={mode})")

        result = pipeline(img1, img2, M=frames, t0=t0)
        if not hasattr(result, "frames"):
            raise ValueError("Pipelineの戻り値が不明な形式です")

        base_url = str(request.base_url).rstrip("/")
        urls = [f"{base_url}/outputs/{os.path.basename(p)}" for p in result.frames]

        return JSONResponse(
            {
                "status": "ok",
                "mode": mode,
                "frames_generated": result.frames_generated,
                "image_urls": urls,
                "debug": result.debug,
            }
        )

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/outputs/{filename}")
async def get_output_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})


@app.get("/")
async def root():
    return {"status": "running", "message": "Time Reversal WebAPI is active."}

