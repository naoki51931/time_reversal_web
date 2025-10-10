# models/pipeline_time_reversal.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

from typing import Tuple


class TimeReversalPipeline:
    """
    Stable Video Diffusion の UNet (8ch潜在) を自作の補間パイプラインとして1ステップだけ回す簡易版。
    - 画像2枚 → VAEで潜在化 (4ch) → 8chに拡張 → フレーム方向に線形補間 → UNetで1ステップ予測 → スケジューラで1ステップ進める
    - その後、各フレームを VAE.decode(4ch) で復元して PIL に戻す
    """

    def __init__(self, unet, vae, scheduler, image_encoder, device=None, dtype=torch.float16):
        self.unet = unet.to(device or "cuda", dtype=dtype)
        self.vae = vae.to(device or "cuda", dtype=dtype)
        self.scheduler = scheduler
        self.image_encoder: CLIPVisionModel = image_encoder.to(device or "cuda", dtype=dtype)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # CLIP 前処理（ViT-L/14）
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # UNet 側のコンテキスト次元（SVD は 1024）
        self.context_dim = getattr(self.unet.config, "encoder_hid_dim", 1024)
        clip_out_dim = getattr(self.image_encoder.config, "projection_dim", 768)
        self.encoder_hidden_proj = nn.Linear(clip_out_dim, self.context_dim).to(self.device, dtype=torch.float16)

        # スケジューラ未初期化時のデフォルトステップ数
        self.default_inference_steps = 10

    # ---- 画像 → Tensor [-1,1] ----
    def _pil_to_norm_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = T.ToTensor()(image).unsqueeze(0).to(self.device, dtype=self.dtype)  # [1,3,H,W]
        tensor = tensor * 2.0 - 1.0
        return tensor

    # ---- VAE.encode -> 4ch latent ----
    @torch.no_grad()
    def _encode_latent4(self, x_norm: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(x_norm).latent_dist.sample()  # [1,4,h,w]
        return latents.to(self.dtype)

    # ---- 4ch → 8ch へ拡張 (リピート) ----
    def _to_8ch(self, lat4: torch.Tensor) -> torch.Tensor:
        if lat4.shape[1] == 8:
            return lat4
        if lat4.shape[1] != 4:
            raise RuntimeError(f"[ERR] VAE latent channel should be 4 or 8, got {lat4.shape}")
        # [1,4,h,w] -> [1,8,h,w]
        lat8 = lat4.repeat(1, 2, 1, 1)
        print("[WARN] Expand channels 4→8")
        return lat8

    # ---- 画像サイズを (max_w, max_h) を超える場合にだけ縮小 ----
    def _resize_if_needed(self, img: Image.Image, max_wh: Tuple[int, int]) -> Image.Image:
        max_w, max_h = max_wh
        w, h = img.size
        if w <= max_w and h <= max_h:
            return img
        # アスペクト維持で縮小
        scale = min(max_w / w, max_h / h)
        new_w = max(64, int(w * scale)) // 64 * 64  # 64の倍数に丸める（SVD的に安全）
        new_h = max(64, int(h * scale)) // 64 * 64
        resized = img.resize((new_w, new_h), Image.BICUBIC)
        return resized

    # ---- 埋め込み作成: [B, F*H*W, 1024] ----
    @torch.no_grad()
    def _make_context(self, img1: Image.Image, img2: Image.Image, F: int, h: int, w: int) -> torch.Tensor:
        # 画像ごとに CLIP 埋め込み取得 → 線形補間で F 個作成 → 射影 → 各フレームを空間分だけ複製
        def encode_one(pil: Image.Image) -> torch.Tensor:
            pixel = self.image_processor(images=pil, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
            emb = self.image_encoder(pixel).image_embeds  # [1, 768 or proj]
            return emb

        emb1 = encode_one(img1)  # [1, D]
        emb2 = encode_one(img2)  # [1, D]

        embs = []
        for i in range(F):
            t = 0.0 if F == 1 else i / (F - 1)
            e = (1 - t) * emb1 + t * emb2  # [1, D]
            e = self.encoder_hidden_proj(e)  # [1, 1024]
            embs.append(e)

        frame_ctx = torch.cat(embs, dim=0)  # [F, 1024]
        frame_ctx = frame_ctx.repeat_interleave(h * w, dim=0)  # [F*h*w, 1024]
        frame_ctx = frame_ctx.unsqueeze(0)  # [1, F*h*w, 1024]
        return frame_ctx.to(self.dtype)

    # ---- メイン：画像2枚 → Fフレーム ----
    @torch.no_grad()
    def __call__(
        self,
        image_1: Image.Image,
        image_2: Image.Image,
        M: int = 2,
        t0: int = 5,
        max_input_size: Tuple[int, int] = (512, 320),
        fallback_size: Tuple[int, int] = (320, 200),
        decode_chunk_size: int = 8,
    ):
        device, dtype = self.device, self.dtype
        print("[Pipeline] Start processing...")

        # 入力をまず上限制御で縮小
        img1 = self._resize_if_needed(image_1, max_input_size)
        img2 = self._resize_if_needed(image_2, max_input_size)
        print(f"[Info] Resized img1 -> {img1.size}")
        print(f"[Info] Resized img2 -> {img2.size}")

        # 正規化テンソルへ
        x1 = self._pil_to_norm_tensor(img1)  # [1,3,H,W]
        x2 = self._pil_to_norm_tensor(img2)

        # VAE エンコード → 4ch latent
        try:
            lat4_a = self._encode_latent4(x1)  # [1,4,h,w]
            lat4_b = self._encode_latent4(x2)
        except torch.cuda.OutOfMemoryError:
            # さらに縮小してリトライ
            print("[OOM] retrying with smaller size...")
            img1_small = self._resize_if_needed(img1, fallback_size)
            img2_small = self._resize_if_needed(img2, fallback_size)
            print(f"[Info] Resized -> {img1_small.size}")
            print(f"[Info] Resized -> {img2_small.size}")
            x1 = self._pil_to_norm_tensor(img1_small)
            x2 = self._pil_to_norm_tensor(img2_small)
            lat4_a = self._encode_latent4(x1)
            lat4_b = self._encode_latent4(x2)

        # 4→8ch
        lat8_a = self._to_8ch(lat4_a)  # [1,8,h,w]
        lat8_b = self._to_8ch(lat4_b)  # [1,8,h,w]
        print(f"[Pipeline] Latents adjusted: {lat8_a.shape}, {lat8_b.shape}")

        # フレーム方向に線形補間 → [1,8,F,h,w]
        F = int(max(1, M))
        h, w = lat8_a.shape[-2], lat8_a.shape[-1]
        frames = []
        for i in range(F):
            t = 0.0 if F == 1 else i / (F - 1)
            frames.append((1 - t) * lat8_a + t * lat8_b)
        latents = torch.stack(frames, dim=2)  # [1,8,F,h,w]
        print(f"[Pipeline] Latent sequence: {latents.shape}  # [B, C, F, H, W]")

        # コンテキスト埋め込み [1, F*h*w, 1024]
        encoder_hidden_states = self._make_context(img1, img2, F=F, h=h, w=w)
        print(f"[Pipeline] Encoded hidden states: {encoder_hidden_states.shape}  # [B, F*H*W, 1024]")

        # UNet 入力そのまま（SVD は [B,8,F,h,w]）
        sample = latents  # [1,8,F,h,w]
        print(f"[Pipeline] Latents ready for UNet (final): {sample.shape}  # [B,F,C,H,W]")

        # スケジューラ timesteps を用意（未設定ならデフォルトで作成）
        if not hasattr(self.scheduler, "timesteps") or len(getattr(self.scheduler, "timesteps", [])) == 0:
            self.scheduler.set_timesteps(self.default_inference_steps, device=device)

        timestep = self.scheduler.timesteps[min(t0, len(self.scheduler.timesteps) - 1)]

        # UNet 1ステップ → scheduler 1ステップ
        try:
            # UNet は fp16 前提
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=self.dtype):
                noise_pred = self.unet(
                    sample,
                    timestep.repeat(sample.shape[0]),  # [B]
                    encoder_hidden_states=encoder_hidden_states,  # [B, F*h*w, 1024]
                ).sample  # [1,8,F,h,w]
            print("[Pipeline] Noise predicted.")

            latents_denoised = self.scheduler.step(noise_pred, timestep, sample).prev_sample  # [1,8,F,h,w]
            print("[Pipeline] Denoising done.")
        except torch.cuda.OutOfMemoryError:
            print("[CUDA ERROR] Out of memory during inference!")
            raise

        # 各フレームをデコード（VAE は 4ch想定なので 8ch→先頭4chを使用）
        frames_out = []
        for f in range(F):
            z8 = latents_denoised[:, :, f, :, :]  # [1,8,h,w]
            z4 = z8[:, :4, :, :]                  # [1,4,h,w]
            rec = self.vae.decode(z4).sample      # [1,3,H,W], [-1,1]
            rec = (rec / 2 + 0.5).clamp(0, 1).squeeze(0).to(torch.float32)  # [3,H,W]
            frames_out.append(T.ToPILImage()(rec.cpu()))

        # API 互換の戻り値
        return type("Result", (), {"frames": [frames_out]})()

