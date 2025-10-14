from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL


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
    t = torch.tensor(list(pil.getdata()), dtype=torch.float32).view(pil.size[1], pil.size[0], 3)
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = (t / 255.0 - 0.5) * 2.0
    return t.to(device=device, dtype=dtype)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    t = (t / 2 + 0.5).clamp(0, 1)
    t = (t * 255).round().to(torch.uint8)
    t = t.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(t)


def _extract_lineart(pil: Image.Image, blur_ksize=3, threshold_block=9, threshold_C=2) -> Image.Image:
    """
    画像から線画を抽出する OpenCV 処理。
    - グレースケール化 → ノイズ除去 → 自適応二値化
    - 白背景＋黒線化
    """
    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ノイズ軽減
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 適応的二値化で線抽出
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        threshold_block,
        threshold_C,
    )

    # 黒線→黒、背景→白
    line = 255 - edges

    # 3ch化して戻す
    out = cv2.cvtColor(line, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out)


@dataclass
class PipelineResult:
    status: str
    frames_generated: int
    frames: List[str]
    debug: dict


class TimeReversalPipeline:
    """
    Diffusers の VAE を使った潜在補間＋線画抽出。
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

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()
        self.scaling_factor: float = getattr(self.vae.config, "scaling_factor", 0.18215)

        print(f"[Init] TimeReversalPipeline ready on {self.device} (dtype={self.dtype})")

    @torch.inference_mode()
    def __call__(self, image_1: Image.Image, image_2: Image.Image, M: int = 3, t0: float = 5.0) -> PipelineResult:
        print("[Pipeline] Start processing (vae_interpolate + lineart)")
        M = max(2, int(M))

        # サイズ統一
        w0, h0 = image_1.size
        w1, h1 = image_2.size
        w, h = _make_multiple_of_8(min(w0, w1), min(h0, h1), self.max_side)

        img1 = _pil_to_tensor(image_1, (w, h), self.device, self.dtype)
        img2 = _pil_to_tensor(image_2, (w, h), self.device, self.dtype)
        print(f"[Pipeline] Preprocessed size: {(w, h)}")

        # encode -> latent
        posterior_1 = self.vae.encode(img1).latent_dist
        posterior_2 = self.vae.encode(img2).latent_dist
        z1 = posterior_1.mean * self.scaling_factor
        z2 = posterior_2.mean * self.scaling_factor
        print(f"[Pipeline] Latents shape: {tuple(z1.shape)}")

        # 非線形補間
        t = torch.linspace(0, 1, steps=M, device=self.device, dtype=self.dtype)
        t0_tensor = torch.tensor(t0, device=self.device, dtype=self.dtype)
        alphas = (torch.exp(t * t0_tensor) - 1) / (torch.exp(t0_tensor) - 1) if t0 != 0 else t

        frames: List[str] = []
        for i, a in enumerate(alphas):
            z = (1 - a) * z1 + a * z2
            z = z.to(dtype=self.dtype)
            x = self.vae.decode(z / self.scaling_factor).sample
            out_pil = _tensor_to_pil(x)

            # 🔧 線画抽出処理を追加
            out_pil = _extract_lineart(out_pil, blur_ksize=3, threshold_block=9, threshold_C=2)

            out_path = os.path.join(self.output_dir, f"frame_{i:03d}.png")
            out_pil.save(out_path)
            frames.append(out_path)
            print(f"[Pipeline] Saved: {out_path}")

        debug = {
            "device": _to_serializable(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "size": [h, w],
            "latent_shape": [int(x) for x in z1.shape],
            "mode": "vae_interpolate_lineart",
            "t0": float(t0),
        }

        return PipelineResult(status="ok", frames_generated=len(frames), frames=frames, debug=debug)

