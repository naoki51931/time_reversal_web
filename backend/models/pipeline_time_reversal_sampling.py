# -*- coding: utf-8 -*-
"""
Stable Diffusion + DDIM による Time Reversal Sampling（時反転サンプリング）
完全fp32版（GPU/CPU両対応）
- dtype混在エラー完全排除
- すべてのモデル/テンソルをfloat32に統一
- 複数中間フレーム生成対応
"""

import os
import math
from typing import Optional, List
import torch
from PIL import Image
from torchvision import transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel


# ======================================================
# Utility
# ======================================================
def _seed_all(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class _LongestMaxSize(torch.nn.Module):
    def __init__(self, max_size: int = 512):
        super().__init__()
        self.max_size = max_size

    def forward(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        if max(w, h) == self.max_size:
            return im
        scale = self.max_size / float(max(w, h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return im.resize((nw, nh), Image.LANCZOS)


class _CenterPadToMultiple(torch.nn.Module):
    def __init__(self, multiple: int = 64, fill=(0, 0, 0)):
        super().__init__()
        self.multiple = multiple
        self.fill = fill

    def forward(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        tw = math.ceil(w / self.multiple) * self.multiple
        th = math.ceil(h / self.multiple) * self.multiple
        if (tw, th) == (w, h):
            return im
        bg = Image.new("RGB", (tw, th), self.fill)
        offx = (tw - w) // 2
        offy = (th - h) // 2
        bg.paste(im, (offx, offy))
        return bg


# ======================================================
# Main TRS Pipeline (float32版)
# ======================================================
class TimeReversalPipeline:
    def __init__(self, sd_repo="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        print(f"[TRS] Device={self.device}, dtype={self.dtype}")

        # --- モデル読込（全てfp32） ---
        self.vae = AutoencoderKL.from_pretrained(sd_repo, subfolder="vae").to(self.device, dtype=self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(sd_repo, subfolder="unet").to(self.device, dtype=self.dtype)
        self.scheduler = DDIMScheduler.from_pretrained(sd_repo, subfolder="scheduler")

        self.tokenizer = CLIPTokenizer.from_pretrained(sd_repo, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_repo, subfolder="text_encoder").to(self.device, dtype=self.dtype)

        # --- 前処理 ---
        self.preproc = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            _LongestMaxSize(512),
            _CenterPadToMultiple(64),
            T.ToTensor()
        ])

    # --------------------------------------------------
    # Encode / Decode
    # --------------------------------------------------
    @torch.no_grad()
    def _encode_vae(self, img01_bchw: torch.Tensor) -> torch.Tensor:
        img_norm = img01_bchw * 2 - 1
        z = self.vae.encode(img_norm.to(self.device, dtype=self.dtype)).latent_dist.sample()
        z = z * 0.18215
        return z.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def _decode_vae(self, latents: torch.Tensor) -> torch.Tensor:
        z = (latents / 0.18215).to(self.device, dtype=self.dtype)
        img = self.vae.decode(z).sample
        img = (img.clamp(-1, 1) + 1) / 2
        return img

    # --------------------------------------------------
    # Text Encoder
    # --------------------------------------------------
    @torch.no_grad()
    def _text_embeds(self, prompt: str = "", neg: str = ""):
        tok = self.tokenizer

        def enc(txt):
            t = tok([txt], padding="max_length", max_length=tok.model_max_length,
                    truncation=True, return_tensors="pt")
            return self.text_encoder(t.input_ids.to(self.device)).last_hidden_state.to(self.device, dtype=self.dtype)

        uncond = enc(neg or "")
        cond = enc(prompt or "")
        return uncond, cond

    # --------------------------------------------------
    # Core TRS (float32統一)
    # --------------------------------------------------
    @torch.no_grad()
    def generate_midframe(
        self,
        imgA: Image.Image,
        imgB: Image.Image,
        t: float = 0.5,
        tau_step: int = 35,
        num_steps: int = 50,
        guidance_scale: float = 5.0,
        prompt: str = "smooth interpolation frame between two images",
        out_path: str = "outputs/trs_midframe.png",
    ) -> str:

        _seed_all(42)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # --- 前処理 ---
        A = self.preproc(imgA).unsqueeze(0).to(self.device, dtype=self.dtype)
        B = self.preproc(imgB).unsqueeze(0).to(self.device, dtype=self.dtype)

        # --- latent空間へ ---
        zA = self._encode_vae(A)
        zB = self._encode_vae(B)

        # --- テキスト埋め込み ---
        uncond, cond = self._text_embeds(prompt)
        uncond = uncond.to(self.device, dtype=self.dtype)
        cond = cond.to(self.device, dtype=self.dtype)

        # --- スケジューラ初期化 ---
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        tau_step = min(max(1, tau_step), len(timesteps) - 1)
        t_tau = timesteps[tau_step]

        # --- ノイズ付与 ---
        noiseA = torch.randn_like(zA, dtype=self.dtype)
        noiseB = torch.randn_like(zB, dtype=self.dtype)
        zA_tau = self.scheduler.add_noise(zA, noiseA, t_tau).to(self.device, dtype=self.dtype)
        zB_tau = self.scheduler.add_noise(zB, noiseB, t_tau).to(self.device, dtype=self.dtype)

        # --- 線形補間 ---
        t = float(t)
        z_tau = ((1 - t) * zA_tau + t * zB_tau).to(self.device, dtype=self.dtype)

        # --- 時反転逆拡散ループ ---
        zt = z_tau.clone()
        for t_i in timesteps[tau_step:]:
            z_in = torch.cat([zt, zt], dim=0).to(self.device, dtype=self.dtype)
            cond_emb = torch.cat([uncond, cond], dim=0).to(self.device, dtype=self.dtype)

            eps = self.unet(z_in, t_i, encoder_hidden_states=cond_emb).sample.to(self.device, dtype=self.dtype)
            e_uncond, e_cond = eps.chunk(2, dim=0)
            e = e_uncond + guidance_scale * (e_cond - e_uncond)
            e = e.to(self.device, dtype=self.dtype)

            step = self.scheduler.step(e, t_i, zt)
            zt = step.prev_sample.to(self.device, dtype=self.dtype)

        # --- デコード ---
        img01 = self._decode_vae(zt)
        img_np = (img01[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img_np).save(out_path)
        return out_path


# ======================================================
# Public API for FastAPI
# ======================================================
_TRS_SINGLETON: Optional[TimeReversalPipeline] = None


def _get_trs_singleton():
    global _TRS_SINGLETON
    if _TRS_SINGLETON is None:
        _TRS_SINGLETON = TimeReversalPipeline()
    return _TRS_SINGLETON


@torch.no_grad()
def generate_midframes_trs(
    imgA: Image.Image,
    imgB: Image.Image,
    frames: int = 5,
    tau_step: int = 35,
    num_steps: int = 50,
    guidance_scale: float = 5.0,
    prompt: str = "smooth interpolation frame between two images",
    out_dir: str = "outputs",
) -> List[str]:
    pipe = _get_trs_singleton()
    os.makedirs(out_dir, exist_ok=True)

    results: List[str] = []
    for i in range(frames):
        t = (i + 1) / (frames + 1)
        path = os.path.join(out_dir, f"trs_midframe_{i:02d}.png")
        pipe.generate_midframe(
            imgA=imgA,
            imgB=imgB,
            t=t,
            tau_step=tau_step,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            prompt=prompt,
            out_path=path,
        )
        results.append(path)
    return results

