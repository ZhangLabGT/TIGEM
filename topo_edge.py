import numpy as np
import torch

# --- tqdm 兜底导入：没装 tqdm 也能跑 ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): 
        return x
LOG_EPS = float(np.log(1e-6))
# =======================
# 1) Patch 内部统计（只算 N 次）
# =======================
def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    G = X @ X.T
    sq = np.clip(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0.0, None)
    return np.sqrt(sq, dtype=float)

def _median_nnd(A: np.ndarray) -> float:
    n = len(A)
    if n <= 1:
        return np.inf
    D = _pairwise_dists(A)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1)))

def _iqr_over_med_nnd(A: np.ndarray, eps: float = 1e-8) -> float:
    n = len(A)
    if n <= 2:
        return 0.0
    D = _pairwise_dists(A)
    np.fill_diagonal(D, np.inf)
    nnd = D.min(axis=1)
    q1, q3 = np.percentile(nnd, [25, 75])
    med = np.median(nnd)
    return float((q3 - q1) / max(med, eps))

def _emst_bottleneck_prim(A: np.ndarray) -> float:
    """
    用完整图 Prim 求 EMST 的最大边（瓶颈），O(n^2)；n<=50 足够快。
    仅用于 patch 内 flag f_i 判定；不在边级循环里反复调用。
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    if n < 2:
        return 0.0
    D = _pairwise_dists(A)
    inT = np.zeros(n, dtype=bool)
    inT[0] = True
    best = D[0].copy()
    max_edge = 0.0
    for _ in range(n - 1):
        j = np.argmin(np.where(inT, np.inf, best))
        w = best[j]
        max_edge = max(max_edge, float(w))
        inT[j] = True
        best = np.minimum(best, D[j])
    return float(max_edge)

def precompute_patch_stats(patch_cells, tau_split_factor: float = 3.0):
    """
    patch_cells: List[np.ndarray], 每个元素 (n_i,2)
    返回 dict：{'s': np.ndarray, 'h': np.ndarray, 'f': np.ndarray}
      s: median NND
      h: IQR/median of NND
      f: (EMST 瓶颈 > tau* s) 的裂解旗标 {0,1}
    """
    N = len(patch_cells)
    s = np.empty(N, dtype=float)
    h = np.empty(N, dtype=float)
    f = np.empty(N, dtype=int)
    for i, A in enumerate(patch_cells):
        si = _median_nnd(A)
        hi = _iqr_over_med_nnd(A)
        bott = _emst_bottleneck_prim(A)
        fi = int(np.isfinite(si) and si > 0 and bott > tau_split_factor * si)
        s[i], h[i], f[i] = si, hi, fi
    return dict(s=s, h=h, f=f)

# =======================
# 2) 两集合最小距离（替代 EMST 路径瓶颈）
# =======================
def _min_set_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    两集合最小两两距离：O(n_a * n_b)；n_a,n_b ≤ ~50 时毫秒级。
    空集合返回 +inf。
    """
    na, nb = len(A), len(B)
    if na == 0 or nb == 0:
        return np.inf
    A = np.asarray(A, dtype=float)[:, None, :]  # (na,1,2)
    B = np.asarray(B, dtype=float)[None, :, :]  # (1,nb,2)
    d = np.linalg.norm(A - B, axis=2)          # (na,nb)
    return float(d.min())

# =======================
# 3) 单边 log_g（快速版）
# =======================
def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    G = X @ X.T
    sq = np.clip(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0.0, None)
    return np.sqrt(sq, dtype=float)

def _median_nnd(A: np.ndarray) -> float:
    n = len(A)
    if n <= 1:
        return np.inf
    D = _pairwise_dists(A)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1)))

def _iqr_over_med_nnd(A: np.ndarray, eps: float = 1e-8) -> float:
    n = len(A)
    if n <= 2:
        return 0.0
    D = _pairwise_dists(A)
    np.fill_diagonal(D, np.inf)
    nnd = D.min(axis=1)
    q1, q3 = np.percentile(nnd, [25, 75])
    med = np.median(nnd)
    return float((q3 - q1) / max(med, eps))

def _emst_bottleneck_prim(A: np.ndarray) -> float:
    # 仅用于 patch 内 f_i 判定；n<=50 足够快
    A = np.asarray(A, dtype=float)
    n = len(A)
    if n < 2:
        return 0.0
    D = _pairwise_dists(A)
    inT = np.zeros(n, dtype=bool)
    inT[0] = True
    best = D[0].copy()
    max_edge = 0.0
    for _ in range(n - 1):
        j = np.argmin(np.where(inT, np.inf, best))
        w = best[j]
        max_edge = max(max_edge, float(w))
        inT[j] = True
        best = np.minimum(best, D[j])
    return float(max_edge)

def precompute_patch_stats(patch_cells, tau_split_factor: float = 3.0):
    N = len(patch_cells)
    s = np.empty(N, dtype=float)
    h = np.empty(N, dtype=float)
    f = np.empty(N, dtype=int)
    for i, A in enumerate(patch_cells):
        si = _median_nnd(A)
        hi = _iqr_over_med_nnd(A)
        bott = _emst_bottleneck_prim(A)
        fi = int(np.isfinite(si) and si > 0 and bott > tau_split_factor * si)
        s[i], h[i], f[i] = si, hi, fi
    return dict(s=s, h=h, f=f)

def _min_set_distance(A: np.ndarray, B: np.ndarray) -> float:
    na, nb = len(A), len(B)
    if na == 0 or nb == 0:
        return np.inf
    A = np.asarray(A, dtype=float)[:, None, :]
    B = np.asarray(B, dtype=float)[None, :, :]
    d = np.linalg.norm(A - B, axis=2)
    return float(d.min())

def log_g_ij_fast(
    Ai: np.ndarray, Aj: np.ndarray,
    s_i: float, h_i: float, f_i: int,
    s_j: float, h_j: float, f_j: int,
    *,
    r_max: float = 32.0,
    alpha: float = 0.2,
    theta0: float = 0.0, theta1: float = 1.0, theta2: float = 1.0, theta3: float = 1.0,
    alpha_d: float = 1.0, beta_d: float = 0.5, gamma_d: float = 0.5,
    eps: float = 1e-6
): 
    
    if len(Ai) == 0 or len(Aj) == 0:
        return LOG_EPS
    if (not np.isfinite(s_i)) or (not np.isfinite(s_j)) or (s_i <= 0) or (s_j <= 0):
        return LOG_EPS
    # 1) r_star & Π
    r_star = _min_set_distance(Ai, Aj)
    if (not np.isfinite(r_star)) or (r_star >= r_max):
        Pi = 0.0
    else:
        Pi = float(np.log1p(np.exp(alpha * (r_max - r_star))))  # softplus

    # 2) B（仅 b1）
    b1 = r_star / max(np.sqrt(max(s_i, eps) * max(s_j, eps)), eps)
    B_hat = 1.0 / (1.0 + np.exp(-(b1 - 1.0)))  # >1 视作阻隔

    # 3) D
    dens_diff = abs(s_i - s_j) / max(max(s_i, s_j), eps)
    disp_diff = abs(h_i - h_j)
    split_flag = f_i + f_j
    D = alpha_d * dens_diff + beta_d * disp_diff + gamma_d * split_flag

    # 4) Π 压缩
    Pi_hat = np.tanh(0.15 * Pi)

    # 5) 合成 log g
    log_g = theta0 + theta1 * Pi_hat - theta2 * B_hat - theta3 * D
    log_g = float(np.clip(log_g, np.log(1e-6), 0.0))
    return log_g

# =======================
# 4) 批量边的 log_g（带 tqdm + 对称边缓存）
# =======================
def compute_edge_log_g_fast(
    edge_index: torch.Tensor,                # [2, E]（无向或有向都可以）
    patch_cells,                             # List[np.ndarray]
    stats: dict,                             # {'s','h','f'} = precompute_patch_stats(...)
    sample_name: str = "",
    *,
    r_max: float = 32.0,
    alpha: float = 0.2,
    theta0: float = 0.0, theta1: float = 1.0, theta2: float = 1.0, theta3: float = 1.0,
    alpha_d: float = 1.0, beta_d: float = 0.5, gamma_d: float = 0.5
) -> torch.Tensor:
    """
    返回 edge_log_g: FloatTensor[E]（与 edge_index 同序）
    """
    e_src = edge_index[0].cpu().numpy()
    e_dst = edge_index[1].cpu().numpy()
    E = edge_index.size(1)

    s, h, f = stats["s"], stats["h"], stats["f"]
    out = np.empty(E, dtype=np.float32)
    cache = {}

    desc = f"[{sample_name}] log_g fast"
    for kcol in tqdm(range(E), total=E, desc=desc, dynamic_ncols=True):
        i, j = int(e_src[kcol]), int(e_dst[kcol])
        key = (i, j) if i < j else (j, i)   # 无向边共享
        if key in cache:
            lg = cache[key]
        else:
            lg, _ = log_g_ij_fast(
                patch_cells[i], patch_cells[j],
                s[i], h[i], f[i],
                s[j], h[j], f[j],
                r_max=r_max, alpha=alpha,
                theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3,
                alpha_d=alpha_d, beta_d=beta_d, gamma_d=gamma_d
            )
            cache[key] = lg
        out[kcol] = lg
    return torch.from_numpy(out)

# =======================
# 5) 一步到位：给 edge_index 直接产出 edge_log_g
# =======================
def build_edge_log_g_fast(
    edge_index: torch.Tensor,
    patch_cells,
    *,
    sample_name: str = "",
    tau_split_factor: float = 3.0,
    r_max: float = 48.0,
    alpha: float = 0.5,
    theta0: float = 0.0, theta1: float = 2.0, theta2: float = 1.0, theta3: float = 2.0,
    alpha_d: float = 1.0, beta_d: float = 0.5, gamma_d: float = 1
) -> torch.Tensor:
    """
    封装：先预计算每个 patch 的 (s,h,f)，再批量算 edge_log_g。
    """
    stats = precompute_patch_stats(patch_cells, tau_split_factor=tau_split_factor)
    edge_log_g = compute_edge_log_g_fast(
        edge_index, patch_cells, stats, sample_name=sample_name,
        r_max=r_max, alpha=alpha,
        theta0=theta0, theta1=theta1, theta2=theta2, theta3=theta3,
        alpha_d=alpha_d, beta_d=beta_d, gamma_d=gamma_d
    )
    return edge_log_g