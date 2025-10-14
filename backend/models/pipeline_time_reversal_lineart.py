import cv2
import numpy as np
from PIL import Image
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult

def _extract_lineart(pil: Image.Image, blur_ksize=3, threshold_block=9, threshold_C=2) -> Image.Image:
    img = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, threshold_block, threshold_C)
    line = 255 - edges
    out = cv2.cvtColor(line, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out)

class TimeReversalPipeline(TimeReversalBase):
    """出力フレームに線画抽出を適用するバージョン"""

    def preprocess_output(self, image: Image.Image) -> Image.Image:
        return _extract_lineart(image)

