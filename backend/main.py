from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import tempfile, os, zipfile
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
    allow_origins=["http://localhost:3000"],  # React の URL (開発環境)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 🚀 モデル初期化 (HuggingFace からロード)
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[Init] Loading base pipeline...")
base_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
print("[Init] Base pipeline loaded.")

unet = base_pipe.unet
vae = base_pipe.vae
scheduler = base_pipe.scheduler
image_encoder = getattr(base_pipe, "image_encoder", None)  # ある場合のみ

print("[Init] Wrapping into TimeReversalPipeline...")
pipeline = TimeReversalPipeline(unet, vae, scheduler, image_encoder, device=device)
print("[Init] Pipeline ready.")

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
    frames: int = Form(8),          # M : 補間フレーム数
    t0: int = Form(5),              # ノイズ注入の cutoff timestep
    s_churn: float = Form(0.5),     # スケジューラの churn
    w_o_noise_re_injection: bool = Form(False) # noise 再注入なし
):
    print("[Start] /generate called")
    tmpdir = tempfile.mkdtemp()

    # 画像保存
    path1 = os.path.join(tmpdir, "a.png")
    path2 = os.path.join(tmpdir, "b.png")
    with open(path1, "wb") as f:
        f.write(await file1.read())
    with open(path2, "wb") as f:
        f.write(await file2.read())
    print(f"[Info] Input images saved to {path1}, {path2}")

    # PIL 読み込み（リサイズはしない）
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    print("[Info] Input images loaded into PIL")

    generator = torch.manual_seed(42)

    # noise 再注入フラグ
    cutoff_t = 0 if w_o_noise_re_injection else t0
    print(f"[Info] Parameters -> frames={frames}, t0={cutoff_t}, s_churn={s_churn}")

    # パイプライン実行
    print("[Run] Starting pipeline inference...")
    result = pipeline(
        img1, img2,
        s_churn=s_churn,
        M=frames,
        t0=cutoff_t,
        decode_chunk_size=8,
        generator=generator
    )
    frame_list = result.frames[0]
    print("[Run] Pipeline finished")

    # フレーム保存
    frame_paths = []
    total_frames = len(frame_list)
    for i, frame in enumerate(frame_list):
        fpath = os.path.join(tmpdir, f"frame_{i:03d}.png")
        frame.save(fpath)
        frame_paths.append(fpath)
        print(f"[Progress] {i+1}/{total_frames} frames saved")

    # 動画保存
    video_path = os.path.join(tmpdir, "result.mp4")
    imageio.mimsave(video_path, frame_list, fps=7)
    print(f"[Progress] Video saved: {video_path}")

    # zip にまとめる
    zip_path = os.path.join(tmpdir, "result.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(video_path, arcname="result.mp4")
        for f in frame_paths:
            zipf.write(f, arcname=os.path.basename(f))
    print(f"[Done] Result packaged into: {zip_path}")

    return FileResponse(zip_path, media_type="application/zip", filename="result.zip")
