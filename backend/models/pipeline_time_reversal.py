# backend/models/pipeline_time_reversal.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from PIL import Image
from diffusers.models import AutoencoderKL


# =====================================================================
# 🔧 Utility
# =====================================================================

def _to_serializable(x):
    """JSON化できないtorch.deviceなどを文字列化"""
    if isinstance(x, torch.device):
        return str(x)
    return x


def _make_multiple_of_8(w: int, h: int, max_side: int = 768) -> Tuple[int, int]:
    """
    アスペクト比を維持しつつ、各辺を8の倍数に丸め、最大辺を max_side に抑える。
    """
    scale = min(1.0, max_side / max(w, h))
    w = int(round(w * scale))
    h = int(round(h * scale))
    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return w, h


def _pil_to_tensor(pil: Image.Image, size: Tuple[int, int], device, dtype) -> torch.Tensor:
    """
    PIL → 標準化テンソル [-1,1], shape: [1,3,H,W]
    """
    pil = pil.convert("RGB").resize(size, Image.LANCZOS)
    t = torch.tensor(list(pil.getdata()), dtype=torch.float32).view(pil.size[1], pil.size[0], 3)
    t = t.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1,3,H,W]
    t = (t - 0.5) * 2.0  # [-1,1]
    return t.to(device=device, dtype=dtype)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """標準化テンソル [-1,1] → PIL"""
    if t.ndim == 4:
        t = t[0]
    t = (t / 2 + 0.5).clamp(0, 1)
    t = (t * 255).round().to(torch.uint8)
    t = t.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(t)


# =====================================================================
# 🎬 Dataclass
# =====================================================================

@dataclass
class PipelineResult:
    status: str
    frames_generated: int
    frames: List[str]
    debug: dict


# =====================================================================
# 🧩 Pipeline core
# =====================================================================

class TimeReversalPipeline:
    """
    Diffusers 実働版パイプライン（安定版）:
      - Stable Diffusion 1.5 の VAE のみ使用。
      - 2枚の画像の潜在表現を線形/非線形補間。
      - t0 パラメータで補間カーブを制御（加速・減速）。
    """

    def __init__(
        self,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        model_id: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype | None = None,
        output_dir: str = "outputs",
        max_side: int = 768,
    ):
        self.device = torch.device(device)
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.dtype = torch_dtype
        self.output_dir = output_dir
        self.max_side = max_side
        os.makedirs(self.output_dir, exist_ok=True)

        # VAE単体ロード
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        # SD VAE のスケール係数
        self.scaling_factor: float = getattr(self.vae.config, "scaling_factor", 0.18215)
        print(f"[Init] TimeReversalPipeline ready on {self.device} (dtype={self.dtype})")

    # -----------------------------------------------------------------
    @torch.inference_mode()
    def __call__(
        self, image_1: Image.Image, image_2: Image.Image, M: int = 2, t0: float = 5.0
    ) -> PipelineResult:
        """
        画像AとBの潜在空間補間で M フレームを生成。
        t0 で非線形補間カーブを調整（t0>0で後半加速、t0<0で前半加速）。
        """
        print("[Pipeline] Start processing (diffusers_full/vae_interpolate)")
        M = max(2, int(M))

        # --- サイズ調整 ---
        w0, h0 = image_1.size
        w1, h1 = image_2.size
        w, h = _make_multiple_of_8(min(w0, w1), min(h0, h1), self.max_side)

        # --- 前処理 ---
        img1 = _pil_to_tensor(image_1, (w, h), self.device, self.dtype)
        img2 = _pil_to_tensor(image_2, (w, h), self.device, self.dtype)
        print(f"[Pipeline] Preprocessed size: {(w, h)}")

        # --- エンコード（潜在へ） ---
        posterior_1 = self.vae.encode(img1).latent_dist
        posterior_2 = self.vae.encode(img2).latent_dist
        z1 = posterior_1.sample() * self.scaling_factor
        z2 = posterior_2.sample() * self.scaling_factor
        print(f"[Pipeline] Latents shape: {tuple(z1.shape)}")

        # --- 非線形補間カーブ（t0利用） ---
        t = torch.linspace(0, 1, steps=M, device=self.device, dtype=self.dtype)
        if abs(t0) > 1e-6:
            # float → tensor に変換（デバイスとdtypeを合わせる）
            t0_tensor = torch.tensor(t0, device=self.device, dtype=self.dtype)
            alphas = (torch.exp(t * t0_tensor) - 1) / (torch.exp(t0_tensor) - 1)
        else:
            alphas = t

        frames: List[str] = []
        for i, a in enumerate(alphas):
            z = (1 - a) * z1 + a * z2
            x = self.vae.decode(z / self.scaling_factor).sample
            out_pil = _tensor_to_pil(x)
            out_path = os.path.join(self.output_dir, f"frame_{i:03d}.png")
            out_pil.save(out_path)
            frames.append(out_path)
            print(f"[Pipeline] Saved: {out_path}")

        debug = {
            "device": _to_serializable(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "size": [h, w],
            "latent_shape": [int(x) for x in z1.shape],
            "mode": "diffusers_full_vae_only",
            "t0": float(t0),
            "alphas": [float(a) for a in alphas.cpu()],
        }

        return PipelineResult(
            status="ok",
            frames_generated=len(frames),
            frames=frames,
            debug=debug,
        )

