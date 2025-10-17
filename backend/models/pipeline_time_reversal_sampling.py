# -*- coding: utf-8 -*-
"""
Stable Diffusion + DDIM による Time Reversal Sampling（時反転サンプリング）
Baseクラス参照 + t0対応 + FastAPI互換構造
"""

import os
import torch
from typing import List, Optional
from PIL import Image

# Baseを参照
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult


# ======================================================
# Time Reversal Sampling Pipeline
# ======================================================
class TimeReversalPipeline(TimeReversalBase):
    """
    Stable Diffusion VAEの潜在空間で指数補間を行い、Time Reversal Samplingを模倣。
    Baseクラスの補間式を拡張してTRS用補間を実行。
    """
    def preprocess_latent(self, z: torch.Tensor) -> torch.Tensor:
        # TRS補間中に軽い正規化（オプション）
        return z / (z.abs().max() + 1e-6)


# ======================================================
# Public API for FastAPI
# ======================================================
_TRS_SINGLETON: Optional[TimeReversalPipeline] = None


def _get_trs_singleton() -> TimeReversalPipeline:
    """
    Singletonでパイプラインを保持（毎回ロードを防ぐ）
    """
    global _TRS_SINGLETON
    if _TRS_SINGLETON is None:
        print("[TRS] Initializing TimeReversalPipeline singleton...")
        _TRS_SINGLETON = TimeReversalPipeline()
    return _TRS_SINGLETON


@torch.no_grad()
def generate_midframes_trs(
    imgA: Image.Image,
    imgB: Image.Image,
    frames: int = 5,
    t0: float = 5.0,
    out_dir: str = "outputs",
) -> dict:
    """
    Baseクラスを利用して中間フレーム生成
    - t0: 補間指数パラメータ（大きいほど非線形）
    - frames: 中間フレーム数
    """
    pipe = _get_trs_singleton()
    os.makedirs(out_dir, exist_ok=True)

    print(f"[TRS] Running Time Reversal Sampling (frames={frames}, t0={t0})")

    result: PipelineResult = pipe(imgA, imgB, M=frames, t0=t0)

    return {
        "status": result.status,
        "generated": result.frames_generated,
        "frames": result.frames,
        "debug": result.debug,
    }

