# -*- coding: utf-8 -*-
"""
Stable Diffusion + DDIM による Time Reversal Sampling（時反転サンプリング）
白背景維持 + 線をくっきり強調（アンシャープマスク＋輝度補正）
"""

import os
import torch
from typing import Optional
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult


# ======================================================
# Time Reversal Sampling Pipeline
# ======================================================
class TimeReversalPipeline(TimeReversalBase):
    """Time Reversal Sampling with clear white background and bold lines"""

    def preprocess_latent(self, z: torch.Tensor) -> torch.Tensor:
        # 軽い正規化
        return z / (z.abs().max() + 1e-6)

    def postprocess_output(self, image: Image.Image) -> Image.Image:
        """白背景維持 + 線をはっきりくっきり描き出す"""
        # ---- モノクロ変換 ----
        img = image.convert("L")

        # ---- 輝度正規化 ----
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-5)
        np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

        # ---- ノイズ平滑化（線を残す）----
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # ---- コントラスト強調 ----
        img = ImageEnhance.Contrast(img).enhance(2.2)

        # ---- 明るさ少し上げて白地を維持 ----
        img = ImageEnhance.Brightness(img).enhance(1.15)

        # ---- アンシャープマスクで線を太くする ----
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=4))

        # ---- しきい値補正（薄い線を残しつつ白飛び防止）----
        np_img = np.array(img)
        low, high = 170, 240
        mask_dark = np_img < low
        mask_mid = (np_img >= low) & (np_img <= high)
        np_img[mask_mid] = np_img[mask_mid] * 0.7
        np_img[mask_dark] = np.clip(np_img[mask_dark] * 0.5, 0, 120)
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)

        # ---- オートコントラストで最終調整 ----
        img = Image.fromarray(np_img)
        img = ImageOps.autocontrast(img, cutoff=0.5)

        # ---- RGB変換して返す ----
        return img.convert("RGB")


# ======================================================
# Public API for FastAPI
# ======================================================
_TRS_SINGLETON: Optional[TimeReversalPipeline] = None


def _get_trs_singleton() -> TimeReversalPipeline:
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
    pipe = _get_trs_singleton()
    os.makedirs(out_dir, exist_ok=True)

    print(f"[TRS] Running Time Reversal Sampling (frames={frames}, t0={t0})")

    result: PipelineResult = pipe(imgA, imgB, M=frames, t0=t0)

    # === 白背景＋線強調を適用 ===
    enhanced_frames = []
    for f in result.frames:
        try:
            im = Image.open(f)
            im = pipe.postprocess_output(im)
            im.save(f)
            enhanced_frames.append(f)
        except Exception as e:
            print(f"[TRS] Warning: postprocess failed on {f}: {e}")
            enhanced_frames.append(f)

    result.frames = enhanced_frames

    return {
        "status": result.status,
        "generated": result.frames_generated,
        "frames": result.frames,
        "debug": result.debug,
    }

