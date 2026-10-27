import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter


# -----------------------------
# A) Build density map from global cell centroids
# -----------------------------
@torch.no_grad()
def build_density_map_torch(cells_xy: torch.Tensor,
                            slide_hw: torch.Tensor,
                            grid_hw=(256, 256),
                            sigma=2.0):
    """
    cells_xy: [M,2] pixel coords (float)
    slide_hw: [2] (W,H) (long)
    returns:  dens [1,1,Hg,Wg] on GPU
    """
    device = cells_xy.device
    W = float(slide_hw[0].item())
    H = float(slide_hw[1].item())
    Hg, Wg = grid_hw

    # pixel -> grid indices
    gx = (cells_xy[:, 0] / max(W - 1.0, 1.0) * (Wg - 1)).clamp(0, Wg - 1).long()
    gy = (cells_xy[:, 1] / max(H - 1.0, 1.0) * (Hg - 1)).clamp(0, Hg - 1).long()
    idx = gy * Wg + gx

    dens = torch.bincount(idx, minlength=Hg * Wg).float().view(Hg, Wg)
    dens = dens / dens.max().clamp_min(1e-6)
    dens = dens[None, None]  # [1,1,Hg,Wg]

    # Gaussian blur (separable conv)
    if sigma > 0:
        k = int(max(3, math.ceil(6 * sigma)))
        if k % 2 == 0:
            k += 1
        half = k // 2
        x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        g = torch.exp(-(x * x) / (2 * sigma * sigma))
        g = (g / g.sum()).view(1, 1, -1)  # [1,1,k]

        dens = F.conv2d(dens, g.unsqueeze(2), padding=(0, half))  # horizontal
        dens = F.conv2d(dens, g.unsqueeze(3), padding=(half, 0))  # vertical

    dens = dens / dens.max().clamp_min(1e-6)
    return dens


def add_density_grads(dens: torch.Tensor) -> torch.Tensor:
    """
    dens: [1,1,H,W]
    returns: [1,3,H,W] = [dens, dx, dy]
    """
    dx = dens[..., :, 1:] - dens[..., :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = dens[..., 1:, :] - dens[..., :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.cat([dens, dx, dy], dim=1)


# -----------------------------
# B) BarrierNet: amortized predictor b(x)
# -----------------------------
class BarrierNet(nn.Module):
    """
    Input:  [1,3,Hg,Wg] = [density, dx, dy]
    Output: [1,1,Hg,Wg] barrier map b(x) >= 0
    """
    def __init__(self, in_ch=3, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, 1, 1),
        )
        # init to near-zero barrier (softplus(0) ~ 0.693; you can subtract later if desired)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.softplus(self.net(x))


# -----------------------------
# C) From barrier map to edge costs C_ij (line integral)
# -----------------------------
def _pix_to_norm(xy_pix: torch.Tensor, slide_hw: torch.Tensor) -> torch.Tensor:
    """
    xy_pix: [P,2] in pixel coords
    slide_hw: [2] (W,H)
    output: [P,2] normalized [-1,1] for grid_sample (align_corners=True)
    """
    W = float(slide_hw[0].item())
    H = float(slide_hw[1].item())
    x = xy_pix[:, 0] / max(W - 1.0, 1.0)
    y = xy_pix[:, 1] / max(H - 1.0, 1.0)
    return torch.stack([2 * x - 1, 2 * y - 1], dim=1)


def edge_cost_from_bmap(b_map: torch.Tensor,
                        pos_xy: torch.Tensor,
                        slide_hw: torch.Tensor,
                        src: torch.Tensor,
                        dst: torch.Tensor,
                        K: int = 16) -> torch.Tensor:
    """
    b_map: [1,1,Hg,Wg]
    pos_xy: [N,2] pixel coords
    src,dst: [E'] edge endpoints
    return: C [E'] (larger => harder to cross)
    """
    device = b_map.device
    xi = pos_xy[src].to(device)   # [E',2]
    xj = pos_xy[dst].to(device)

    t = (torch.arange(K, device=device, dtype=torch.float32) + 0.5) / K  # [K]
    pts = xi[:, None, :] + t[None, :, None] * (xj - xi)[:, None, :]      # [E',K,2]
    pts_flat = pts.reshape(-1, 2)                                         # [E'*K,2]

    grid = _pix_to_norm(pts_flat, slide_hw).view(1, -1, 1, 2)             # [1,E'*K,1,2]
    vals = F.grid_sample(b_map, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)       # [1,1,E'*K,1]
    vals = vals.view(-1).view(-1, K)                                      # [E',K]
    b_mean = vals.mean(dim=1)                                             # [E']

    dist = torch.norm(xj - xi, dim=1).clamp_min(1e-6)                     # [E']
    return dist * b_mean


# -----------------------------
# D) Topo-KL between model attention p and field prior q
# -----------------------------
def topo_kl_from_field(eidx: torch.Tensor,
                       alpha: torch.Tensor,
                       pos_xy: torch.Tensor,
                       slide_hw: torch.Tensor,
                       b_map: torch.Tensor,
                       tau_b: float = 1.0,
                       tau_kl: float = 0.5,
                       use_heads=None) -> torch.Tensor:
    """
    eidx:  [2,E_with_self] (from model)
    alpha: [E_with_self,H]
    """
    src_full, dst_full = eidx[0], eidx[1]
    mask = (src_full != dst_full)  # remove self loops
    src = src_full[mask]
    dst = dst_full[mask]
    alpha_ns = alpha[mask]         # [E',H]

    # p: mean heads then group-softmax by src
    if use_heads is not None:
        alpha_ns = alpha_ns[:, use_heads]
    p_edge = alpha_ns.mean(dim=1).clamp_min(1e-12)   # [E']
    p = pyg_softmax(p_edge, index=src)               # [E']

    # q: from barrier field along (src,dst)
    C = edge_cost_from_bmap(b_map, pos_xy, slide_hw, src, dst, K=16)  # [E']
    logg = -C / max(tau_b, 1e-6)                                      # <=0
    q = pyg_softmax(logg / max(tau_kl, 1e-6), index=src)

    return F.kl_div((p + 1e-12).log(), q, reduction="batchmean")