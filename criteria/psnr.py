import torch


class PSNR(torch.nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, mval=1., dim=[1, 2, 3]):
        super(PSNR, self).__init__()
        self.mval = mval
        self.dim = dim

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2, dim=self.dim)
        return 20 * torch.log10(self.mval / torch.sqrt(mse))
