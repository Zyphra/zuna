from torch import Tensor
import torch

# def imq_kernel(X: Tensor, Y: Tensor, C: float) -> Tensor:
#     """
#     Was this
#     """
#     X_sqnorms = torch.einsum('...i,...i', X, X)
#     Y_sqnorms = torch.einsum('...i,...i', Y, Y)
#     dist = X_sqnorms.unsqueeze(-1) + Y_sqnorms.unsqueeze(-2) - 2 * (X @ Y.T)
#     dist = torch.clamp(dist, min=0.0)
#     return C / (C + dist)

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

# @torch.compile()
# def mmd_imq(X: Tensor, Y: Tensor, C: float) -> Tensor:
#     """
#     Was this
#     """
#     K_XX = imq_kernel(X, X, C)
#     K_YY = imq_kernel(Y, Y, C)
#     K_XY = imq_kernel(X, Y, C)
#     n, m = X.size(0), Y.size(0)
#     term1 = (K_XX.sum() - K_XX.diag().sum()) / (n*(n-1)) if n>1 else 0.0
#     term2 = (K_YY.sum() - K_YY.diag().sum()) / (m*(m-1)) if m>1 else 0.0
#     term3 = 2 * K_XY.mean()
#     return term1 + term2 - term3


# from pykeops.torch import Vi, Vj
# @torch.compiler.disable
# def mmd_imq_keops(X, Y, C):
#     """
#     Beren: pykeops one is best but requires you being able to install pykeops
#     (CW) - I can't install pykeops - has conflicts.
#     """
#     n, m = X.shape[0], Y.shape[0]
#     Xi, Xj = Vi(X), Vj(X)
#     Yi, Yj = Vi(Y), Vj(Y)
#     K_xx = C / (C + ((Xi - Xj)**2).sum(-1))
#     K_yy = C / (C + ((Yi - Yj)**2).sum(-1))
#     K_xy = C / (C + ((Xi - Yj)**2).sum(-1))
#     s_xx = K_xx.sum_reduction(axis=1).sum() - n       
#     s_yy = K_yy.sum_reduction(axis=1).sum() - m
#     s_xy = K_xy.sum_reduction(axis=1).sum()
#     term1 = s_xx / (n * (n - 1)) if n > 1 else 0.0
#     term2 = s_yy / (m * (m - 1)) if m > 1 else 0.0
#     term3 = 2.0 * s_xy / (n * m)
#     return (term1 + term2 - term3)