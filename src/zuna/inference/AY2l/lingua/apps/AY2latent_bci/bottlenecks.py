from torch import Tensor
import torch

def imq_kernel2(X: Tensor, Y: Tensor, C: float) -> Tensor:
    dist_sq = torch.cdist(X, Y, p=2).pow(2)
    return C / (C + dist_sq)


@torch.compile()
def mmd_imq(X: Tensor, Y: Tensor, C: float) -> Tensor:
    K_XX = imq_kernel2(X, X, C)
    K_YY = imq_kernel2(Y, Y, C)
    K_XY = imq_kernel2(X, Y, C)
    n, m = X.size(0), Y.size(0)
    term1 = (K_XX.sum().double() - K_XX.diag().sum().double()).double() / (n*(n-1)) if n>1 else 0.0
    term2 = (K_YY.sum().double() - K_YY.diag().sum().double()).double() / (m*(m-1)) if m>1 else 0.0
    term3 = 2 * K_XY.mean()
    return (term1 + term2 - term3).float()