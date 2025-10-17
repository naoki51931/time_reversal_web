# -*- coding: utf-8 -*-
"""
Time Reversal Pipeline with light denoising.
Baseクラスを利用し、潜在空間に軽いノイズ除去を適用。
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult


# ======================================================
# ノイズ除去版パイプライン本体
# ======================================================
class TimeReversalDenoisePipeline(TimeReversalBase):
    """潜在空間に軽いノイズ除去を加えたバージョン"""

    def __init__(self, *args, denoise_sigma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_sigma = denoise_sigma

    def preprocess_latent(self, z):
        # 軽い平滑処理：ノイズを抑えつつ形状を保持
        smooth = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        z = smooth * (1 - self.denoise_sigma) + z * self.denoise_sigma
        return z


# ======================================================
# FastAPI 公開用関数
# ======================================================
_DENOISE_SINGLETON: Optional[TimeReversalDenoisePipeline] = None


def _get_denoise_singleton() -> TimeReversalDenoisePipeline:
    global _DENOISE_SINGLETON
    if _DENOISE_SINGLETON is None:
        print("[Denoise] Initializing pipeline...")
        _DENOISE_SINGLETON = TimeReversalDenoisePipeline()
    return _DENOISE_SINGLETON


@torch.no_grad()
def generate_denoised_frames(
    imgA: Image.Image,
    imgB: Image.Image,
    frames: int = 5,
    t0: float = 5.0,
    out_dir: str = "outputs",
    denoise_sigma: float = 0.5,
) -> dict:
    """
    Baseクラス + 軽いノイズ除去で中間フレームを生成
    """
    os.makedirs(out_dir, exist_ok=True)
    pipe = _get_denoise_singleton()
    pipe.denoise_sigma = denoise_sigma

    print(f"[Denoise] Running denoise pipeline (frames={frames}, t0={t0}, sigma={denoise_sigma})")

    result: PipelineResult = pipe(imgA, imgB, M=frames, t0=t0)

    return {
        "status": result.status,
        "generated": result.frames_generated,
        "frames": result.frames,
        "debug": result.debug,
    }

