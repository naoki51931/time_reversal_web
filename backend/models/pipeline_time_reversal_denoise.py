import torch.nn.functional as F
from .pipeline_time_reversal_base import TimeReversalBase, PipelineResult

class TimeReversalPipeline(TimeReversalBase):
    """潜在空間に軽いノイズ除去を加えたバージョン"""

    def __init__(self, *args, denoise_sigma: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_sigma = denoise_sigma

    def preprocess_latent(self, z):
        # 軽い平滑処理
        z = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1) * (1 - self.denoise_sigma) + z * self.denoise_sigma
        return z

