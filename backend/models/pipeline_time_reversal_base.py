from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from PIL import Image
import numpy as np
from diffusers.models import AutoencoderKL


# ---------- Utility ----------

def _to_serializable(x):
    if isinstance(x, torch.device):
        return str(x)
    return x


def _make_multiple_of_8(w: int, h: int, max_side: int = 768) -> Tuple[int, int]:
    scale = min(1.0, max_side / max(w, h))
    w = int(round(w * scale))
    h = int(round(h * scale))
    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return w, h


def _pil_to_tensor(pil: Image.Image, size: Tuple[int, int], device, dtype) -> torch.Tensor:
    pil = pil.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = (t - 0.5) * 2.0
    return t.to(device=device, dtype=dtype)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    t = (t / 2 + 0.5).clamp(0, 1)
    arr = (t * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ---------- Data Class ----------

@dataclass
class PipelineResult:
    status: str
    frames_generated: int
    frames: List[str]
    debug: dict


# ---------- Base Class ----------

class TimeReversalBase:
    """
    共通基底クラス:
      - VAEの初期化
      - Tensor変換ユーティリティ
      - 潜在補間ループ
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "outputs",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_side: int = 768,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.output_dir = output_dir
        self.max_side = max_side
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[Init] Loading VAE ({model_id})...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)

        print(f"[Init] Ready on {self.device} (dtype={self.dtype})")

    # ---- Hooks ----
    def preprocess_output(self, image: Image.Image) -> Image.Image:
        """派生クラスが出力画像を加工するためにオーバーライド"""
        return image

    def preprocess_latent(self, z: torch.Tensor) -> torch.Tensor:
        """派生クラスが潜在zを加工する場合にオーバーライド"""
        return z

    # ---- Main ----
    @torch.inference_mode()
    def __call__(self, image_1: Image.Image, image_2: Image.Image, M: int = 3, t0: float = 5.0) -> PipelineResult:
        print(f"[Pipeline] Start {self.__class__.__name__}")
        M = max(2, int(M))

        # サイズ統一
        w, h = _make_multiple_of_8(*image_1.size, self.max_side)
        img1 = _pil_to_tensor(image_1, (w, h), self.device, self.dtype)
        img2 = _pil_to_tensor(image_2, (w, h), self.device, self.dtype)

        # encode -> latent
        posterior_1 = self.vae.encode(img1).latent_dist
        posterior_2 = self.vae.encode(img2).latent_dist
        z1 = posterior_1.mean * self.scaling_factor
        z2 = posterior_2.mean * self.scaling_factor
        print(f"[Pipeline] Latents: {tuple(z1.shape)}")

        # 非線形補間
        t = torch.linspace(0, 1, steps=M, device=self.device, dtype=self.dtype)
        t0_tensor = torch.tensor(t0, device=self.device, dtype=self.dtype)
        alphas = (torch.exp(t * t0_tensor) - 1) / (torch.exp(t0_tensor) - 1) if t0 != 0 else t

        frames: List[str] = []
        for i, a in enumerate(alphas):
            z = (1 - a) * z1 + a * z2
            z = self.preprocess_latent(z)
            x = self.vae.decode(z / self.scaling_factor).sample
            out_pil = _tensor_to_pil(x)
            out_pil = self.preprocess_output(out_pil)
            path = os.path.join(self.output_dir, f"frame_{i:03d}.png")
            out_pil.save(path)
            frames.append(path)
            print(f"[Pipeline] Saved: {path}")

        debug = {
            "device": _to_serializable(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "size": [h, w],
            "latent_shape": [int(x) for x in z1.shape],
            "mode": self.__class__.__name__,
            "t0": float(t0),
        }

        return PipelineResult("ok", len(frames), frames, debug)

