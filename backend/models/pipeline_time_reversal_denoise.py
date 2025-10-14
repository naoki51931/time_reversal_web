from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL


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
    t = t.permute(2, 0, 1).unsqueeze(0) / 255.0
    t = (t - 0.5) * 2.0
    return t.to(device=device, dtype=dtype)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4:
        t = t[0]
    t = (t / 2 + 0.5).clamp(0, 1)
    t = (t * 255).round().to(torch.uint8)
    t = t.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(t)


@dataclass
class PipelineResult:
    status: str
    frames_generated: int
    frames: List[str]
    debug: dict


class TimeReversalPipeline:
    """
    ノイズ除去軽量版:
      - VAE補間 + Gaussian平滑
      - Stable Diffusion UNetを使わず軽量にノイズ除去
    """

    def __init__(
        self,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "outputs",
        torch_dtype: torch.dtype | None = None,
        max_side: int = 768,
        denoise_sigma: float = 0.5,
    ):
        self.device = torch.device(device)
        self.dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.output_dir = output_dir
        self.max_side = max_side
        self.denoise_sigma = denoise_sigma

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"[Init] Loading VAE only ({model_id})...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        self.vae.eval()
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        print(f"[Init] Denoise pipeline ready on {self.device} (dtype={self.dtype})")

    @torch.inference_mode()
    def __call__(self, image_1: Image.Image, image_2: Image.Image, M: int = 2, t0: float = 5.0) -> PipelineResult:
        print("[Pipeline] Start processing (denoise)")
        M = max(2, int(M))

        w, h = _make_multiple_of_8(*image_1.size, max_side=self.max_side)
        img1 = _pil_to_tensor(image_1, (w, h), self.device, self.dtype)
        img2 = _pil_to_tensor(image_2, (w, h), self.device, self.dtype)

        z1 = self.vae.encode(img1).latent_dist.sample() * self.scaling_factor
        z2 = self.vae.encode(img2).latent_dist.sample() * self.scaling_factor
        print(f"[Pipeline] Latents shape: {tuple(z1.shape)}")

        alphas = torch.linspace(0, 1, steps=M, device=self.device, dtype=self.dtype)
        frames: List[str] = []

        for i, a in enumerate(alphas):
            z = (1 - a) * z1 + a * z2

            # --- 軽いノイズ除去: Gaussian blurを潜在に適用 ---
            z = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1) * (1 - self.denoise_sigma) + z * self.denoise_sigma

            # --- Decode back to image ---
            x = self.vae.decode(z / self.scaling_factor).sample
            out_pil = _tensor_to_pil(x)
            out_path = os.path.join(self.output_dir, f"frame_{i:03d}.png")
            out_pil.save(out_path)
            frames.append(out_path)
            print(f"[Pipeline] Saved: {out_path}")

        debug = {
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "size": [h, w],
            "latent_shape": list(z1.shape),
            "mode": "diffusers_denoise_light",
            "denoise_sigma": self.denoise_sigma,
        }

        return PipelineResult(status="ok", frames_generated=len(frames), frames=frames, debug=debug)

