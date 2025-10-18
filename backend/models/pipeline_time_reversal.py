# -*- coding: utf-8 -*-
"""
標準VAE補間（通常 Time Reversal モード）
Baseクラスの基本的な潜在補間を使用
"""

import os
import torch
from typing import Optional
from PIL import Image
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult

# ======================================================
# Pipeline 本体
# ======================================================
class TimeReversalPipeline(TimeReversalBase):
    """基本的な潜在補間（標準VAEモード）"""
    pass


# ======================================================
# FastAPI 連携ラッパー関数
# ======================================================
_TRS_NORMAL_SINGLETON: Optional[TimeReversalPipeline] = None


def _get_normal_singleton() -> TimeReversalPipeline:
    global _TRS_NORMAL_SINGLETON
    if _TRS_NORMAL_SINGLETON is None:
        print("[Normal] Initializing TimeReversalPipeline singleton...")
        _TRS_NORMAL_SINGLETON = TimeReversalPipeline()
    return _TRS_NORMAL_SINGLETON


@torch.no_grad()
def generate_midframes_normal(
    imgA: Image.Image,
    imgB: Image.Image,
    frames: int = 5,
    t0: float = 5.0,
    out_dir: str = "outputs",
) -> dict:
    """
    Baseクラスの基本補間を用いた通常モード
    """
    os.makedirs(out_dir, exist_ok=True)
    pipe = _get_normal_singleton()

    # ✅ 出力先を毎回更新
    pipe.output_dir = out_dir  

    print(f"[Normal] Running TimeReversalPipeline (frames={frames}, t0={t0})")
    print(f"[Normal] Output directory = {pipe.output_dir}")

    result: PipelineResult = pipe(imgA, imgB, M=frames, t0=t0)

    # === 出力確認 ===
    if result.frames:
        print(f"[Normal] ✅ Generated {len(result.frames)} frames:")
        for f in result.frames:
            print(f"   → {f}")
    else:
        print("[Normal] ⚠️ No frames generated.")

    return {
        "status": result.status,
        "generated": result.frames_generated,
        "frames": result.frames,
        "debug": result.debug,
    }

