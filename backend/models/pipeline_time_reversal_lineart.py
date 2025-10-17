# -*- coding: utf-8 -*-
"""
線画抽出付き Time Reversal Pipeline
- Baseクラスを利用して潜在補間
- 出力時に線画フィルタを適用
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import Optional, List
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult


# ======================================================
# Utility: 線画抽出関数
# ======================================================
def _extract_lineart(
    pil: Image.Image,
    blur_ksize=3,
    threshold_block=9,
    threshold_C=2
) -> Image.Image:
    """
    Adaptive Threshold による線画抽出
    """
    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, threshold_block, threshold_C
    )
    line = 255 - edges
    out = cv2.cvtColor(line, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out)


# ======================================================
# 線画版パイプライン本体
# ======================================================
class TimeReversalLineartPipeline(TimeReversalBase):
    """出力フレームに線画抽出を適用するバージョン"""

    def preprocess_output(self, image: Image.Image) -> Image.Image:
        return _extract_lineart(image)


# ======================================================
# FastAPI 公開用関数
# ======================================================
_LINEART_SINGLETON: Optional[TimeReversalLineartPipeline] = None


def _get_lineart_singleton() -> TimeReversalLineartPipeline:
    global _LINEART_SINGLETON
    if _LINEART_SINGLETON is None:
        print("[LineArt] Initializing pipeline...")
        _LINEART_SINGLETON = TimeReversalLineartPipeline()
    return _LINEART_SINGLETON


@torch.no_grad()
def generate_lineart_frames(
    imgA: Image.Image,
    imgB: Image.Image,
    frames: int = 5,
    t0: float = 5.0,
    out_dir: str = "outputs",
) -> dict:
    """
    Baseクラス + 線画出力で中間フレームを生成する
    """
    os.makedirs(out_dir, exist_ok=True)
    pipe = _get_lineart_singleton()

    print(f"[LineArt] Running lineart pipeline (frames={frames}, t0={t0})")

    result: PipelineResult = pipe(imgA, imgB, M=frames, t0=t0)

    return {
        "status": result.status,
        "generated": result.frames_generated,
        "frames": result.frames,
        "debug": result.debug,
    }

