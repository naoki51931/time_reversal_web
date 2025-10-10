from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import tempfile, os, zipfile, traceback
from PIL import Image
import torch
import imageio
from models.pipeline_time_reversal import TimeReversalPipeline

app = FastAPI()

# ------------------------------------------------------
# 🚀 CORS 設定
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 全許可（デバッグ用）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 🚀 モデル初期化 (HuggingFace からロード)
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[Init] Loading base pipeline...")
try:
    base_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    # メモリ最適化（StableVideoDiffusion対応）
    torch.cuda.empty_cache()
    base_pipe.enable_attention_slicing()
    base_pipe.enable_model_cpu_offload()
    base_pipe.enable_xformers_memory_efficient_attention()
    print("[Init] Base pipeline loaded.")

except Exception as e:
    print("[ERROR] Failed to load base pipeline:", e)
    traceback.print_exc()
    raise e

try:
    unet = base_pipe.unet
    vae = base_pipe.vae
    scheduler = base_pipe.scheduler
    image_encoder = getattr(base_pipe, "image_encoder", None)

    print("[Init] Wrapping into TimeReversalPipeline...")
    pipeline = TimeReversalPipeline(unet, vae, scheduler, image_encoder, device=device)
    print("[Init] Pipeline ready.")
except Exception as e:
    print("[ERROR] Pipeline initialization failed:", e)
    traceback.print_exc()
    raise e


# ------------------------------------------------------
# 🚀 API エンドポイント
# ------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Time-Reversal API is running!"}


@app.post("/generate")
async def generate(
    file1: UploadFile,
    file2: UploadFile,
    frames: int = Form(8),
    t0: int = Form(5),
    s_churn: float = Form(0.5),
    w_o_noise_re_injection: bool = Form(False)
):
    print("[Start] /generate called")
    tmpdir = tempfile.mkdtemp()

    try:
        # ファイル保存
        path1 = os.path.join(tmpdir, "a.png")
        path2 = os.path.join(tmpdir, "b.png")
        with open(path1, "wb") as f:
            f.write(await file1.read())
        with open(path2, "wb") as f:
            f.write(await file2.read())
        print(f"[Info] Input images saved to {path1}, {path2}")

        # PIL 読み込み
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        print("[Pipeline] Images loaded.")

        # --- 🔹 自動リサイズ処理をここに追加 ---
        max_size = 512  # GPU負荷を抑える目安解像度（512～768が妥当）
        if img1.width > max_size or img1.height > max_size:
            img1.thumbnail((max_size, max_size))
            print(f"[Info] Resized img1 -> {img1.size}")
        if img2.width > max_size or img2.height > max_size:
            img2.thumbnail((max_size, max_size))
            print(f"[Info] Resized img2 -> {img2.size}")
        # ----------------------------------------

    except Exception as e:
        print("[ERROR] Failed to load input images:", e)
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": "画像の読み込みに失敗しました", "detail": str(e)})

    try:
        generator = torch.manual_seed(42)
        cutoff_t = 0 if w_o_noise_re_injection else t0
        print(f"[Info] Parameters -> frames={frames}, t0={cutoff_t}, s_churn={s_churn}")

        print("[Run] Starting pipeline inference...")
        result = pipeline(
            img1, img2,
            s_churn=s_churn,
            M=frames,
            t0=cutoff_t,
            decode_chunk_size=8,
            generator=generator
        )
        torch.cuda.empty_cache()
        frame_list = result.frames[0]
        print("[Run] Pipeline finished successfully.")

    except torch.cuda.OutOfMemoryError as e:
        print("[CUDA ERROR] Out of memory during inference!")
        torch.cuda.empty_cache()
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "GPUメモリ不足です。画像サイズを小さくしてください。", "detail": str(e)})

    except Exception as e:
        print("[ERROR] Pipeline execution failed:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "推論中にエラーが発生しました", "detail": str(e)})

    try:
        # 結果保存
        frame_paths = []
        total_frames = len(frame_list)
        for i, frame in enumerate(frame_list):
            fpath = os.path.join(tmpdir, f"frame_{i:03d}.png")
            frame.save(fpath)
            frame_paths.append(fpath)
            print(f"[Progress] Saved frame {i+1}/{total_frames}")

        video_path = os.path.join(tmpdir, "result.mp4")
        imageio.mimsave(video_path, frame_list, fps=7)
        print(f"[Progress] Video saved: {video_path}")

        zip_path = os.path.join(tmpdir, "result.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(video_path, arcname="result.mp4")
            for f in frame_paths:
                zipf.write(f, arcname=os.path.basename(f))
        print(f"[Done] Result packaged into: {zip_path}")

        return FileResponse(zip_path, media_type="application/zip", filename="result.zip")

    except Exception as e:
        print("[ERROR] Failed to save or zip results:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "結果の保存に失敗しました", "detail": str(e)})
