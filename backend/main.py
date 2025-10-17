import os
import uuid
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

# =========================================
# 🌍 環境変数読み込み
# =========================================
if load_dotenv():
    print("[✅ ENV] .envファイルを読み込みました。")
else:
    print("[⚠️ ENV] .envファイルが見つかりません。")

# ベースURL（自動的に環境変数から取得）
BASE_URL = os.getenv("BASE_URL", "http://13.159.71.138:8000")

# =========================================
# 🚀 FastAPI アプリ設定
# =========================================
app = FastAPI(title="Time Reversal Web", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中は全許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# 📁 出力ディレクトリ作成
# =========================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# 🌀 補間エンドポイント
# =========================================
@app.post("/generate")
async def generate(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    frames: int = Form(8),
    t0: float = Form(0.0),
    lineart: bool = Form(False),
    denoise: bool = Form(False),
    diffusion_trs: bool = Form(False),
    motion: bool = Form(False),
    strength: float = Form(0.4),
    guidance: float = Form(15.0),
):
    print(f"[Start] /generate called")
    print(
        f"[Info] Params: frames={frames}, t0={t0}, "
        f"lineart={lineart}, denoise={denoise}, diffusion_trs={diffusion_trs}, motion={motion}"
    )

    # === ファイル読み込み ===
    img1 = Image.open(BytesIO(await image_1.read())).convert("RGB")
    img2 = Image.open(BytesIO(await image_2.read())).convert("RGB")

    # === フレーム数補正 ===
    if frames <= 0:
        print(f"[Warn] frames={frames} → 自動補正: 1 に変更")
        frames = 1

    # === 一意のIDを生成 ===
    run_id = uuid.uuid4().hex[:8]
    subdir = os.path.join(OUTPUT_DIR, f"run_{run_id}")
    os.makedirs(subdir, exist_ok=True)
    print(f"[Run] Session ID = {run_id}")

    try:
        # =========================================
        # 🎞️ Motion モード
        # =========================================
        if motion:
            print("[Mode] Selected -> motion")
            from models.pipeline_motion_auto import generate_motion_interpolation

            result = generate_motion_interpolation(
                img1,
                img2,
                M=frames,
                strength=strength,
                guidance_scale=guidance,
                out_dir=subdir,  # UUIDディレクトリへ保存
            )

            image_urls = [
                f"{BASE_URL}/{path}" if not path.startswith("http") else path
                for path in result.get("frames", [])
            ]

            return JSONResponse({
                "status": "ok",
                "id": run_id,
                "mode": "motion",
                "frames_generated": result.get("generated", 0),
                "image_urls": image_urls,
            })

        # =========================================
        # 🌫️ Time Reversal Sampling モード
        # =========================================
        elif diffusion_trs:
            print("[Mode] Selected -> diffusion_trs")
            from models.pipeline_time_reversal_sampling import generate_midframes_trs

            result = generate_midframes_trs(img1, img2, frames=frames, t0=t0, out_dir=subdir)
            urls = [f"{BASE_URL}/{p}" for p in result["frames"]]

            return JSONResponse({
                "status": "ok",
                "id": run_id,
                "mode": "diffusion_trs",
                "frames_generated": result["generated"],
                "image_urls": urls,
            })

        # =========================================
        # ✏️ 線画モード
        # =========================================
        elif lineart:
            print("[Mode] Selected -> lineart")
            from models.pipeline_time_reversal_lineart import generate_lineart_frames

            result = generate_lineart_frames(img1, img2, frames=frames, out_dir=subdir)
            urls = [f"{BASE_URL}/{p}" for p in result["frames"]]

            return JSONResponse({
                "status": "ok",
                "id": run_id,
                "mode": "lineart",
                "frames_generated": result["generated"],
                "image_urls": urls,
            })

        # =========================================
        # 🌈 ノイズ除去モード
        # =========================================
        elif denoise:
            print("[Mode] Selected -> denoise")
            from models.pipeline_time_reversal_denoise import generate_denoised_frames

            result = generate_denoised_frames(img1, img2, frames=frames, out_dir=subdir)
            urls = [f"{BASE_URL}/{p}" for p in result["frames"]]

            return JSONResponse({
                "status": "ok",
                "id": run_id,
                "mode": "denoise",
                "frames_generated": result["generated"],
                "image_urls": urls,
            })

        # =========================================
        # 🔵 通常モード
        # =========================================
        else:
            print("[Mode] Selected -> normal")
            from models.pipeline_time_reversal_sampling import generate_midframes_trs

            result = generate_midframes_trs(img1, img2, frames=frames, t0=t0, out_dir=subdir)
            urls = [f"{BASE_URL}/{p}" for p in result["frames"]]

            return JSONResponse({
                "status": "ok",
                "id": run_id,
                "mode": "normal",
                "frames_generated": result["generated"],
                "image_urls": urls,
            })

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


# =========================================
# 🖼️ 静的ファイル（出力画像）公開
# =========================================
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# =========================================
# 🏁 起動時メッセージ
# =========================================
@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI backend is running")
    print(f"📂 BASE_URL = {BASE_URL}")
    print(f"📁 OUTPUT_DIR = {OUTPUT_DIR}/")
    print("✅ Available modes: normal, lineart, denoise, diffusion_trs, motion")

