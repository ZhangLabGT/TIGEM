# %%
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, remove_self_loops
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr

def build_mnn_edge_index(
    coords_xy: np.ndarray,
    k: int = 6,
    prune: str = "global",   # "none" | "global" | "local"
    c: float = 1.5,          # 阈值系数：1.5~2.0 一般合适
    k_local: int = 6         # 局部阈值用到的近邻数
) -> torch.Tensor:
    """
    用像素坐标构 kNN -> MNN 无向图，并可选按距离裁掉“异常长边”。

    Args
    ----
    coords_xy : (N,2) float array
    k         : kNN 中的 k
    prune     : "none" 不裁边；"global" 全局阈值；"local" 每点局部阈值
    c         : 阈值倍率
    k_local   : 局部阈值所用的近邻数（不含自身）

    Returns
    -------
    edge_index : [2, E] LongTensor (无向、无自环)
    """
    X = np.asarray(coords_xy, dtype=float)
    N = X.shape[0]
    if N == 0:
        return torch.empty((2,0), dtype=torch.long)

    # ---- 1) kNN -> MNN 初始边 ----
    k_eff = min(k + 1, N)  # 包含自己
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X)
    dists, idxs = nbrs.kneighbors(X)         # dists: (N, k+1)
    idxs = idxs[:, 1:]                       # 去掉自己

    neigh_sets = [set(row.tolist()) for row in idxs]
    edges = set()
    for i in range(N):
        for j in idxs[i]:
            if i in neigh_sets[j]:
                u, v = (i, j) if i < j else (j, i)
                edges.add((u, v))

    # 兜底：极端稀疏导致 MNN 为空，则退回对称化 kNN
    if not edges:
        for i in range(N):
            for j in idxs[i]:
                u, v = (i, j) if i < j else (j, i)
                edges.add((u, v))

    edges = np.array(list(edges), dtype=np.int64)  # (E,2)
    if edges.size == 0:
        return torch.empty((2,0), dtype=torch.long)

    # ---- 2) 计算每条边长度 ----
    i_idx, j_idx = edges[:,0], edges[:,1]
    edge_lens = np.linalg.norm(X[i_idx] - X[j_idx], axis=1)  # (E,)

    # ---- 3) 可选：按距离裁边 ----
    if prune == "global":
        # 用全局“典型间距”乘倍率作为阈值
        # 典型间距：每个点的最近邻距离（排除自己）的中位数
        pitch = np.median(dists[:, 1])  # dists[:,0]是0(自己)，[:,1]是最近邻
        tau = c * pitch
        keep = edge_lens <= tau
        edges = edges[keep]

    elif prune == "local":
        # 每个点都有自己的局部阈值 tau_i = c * median(knn_dists_i)
        k_eff_loc = min(k_local + 1, N)
        d_loc, _ = NearestNeighbors(n_neighbors=k_eff_loc, metric="euclidean").fit(X).kneighbors(X)
        # 排除自身后的局部中位数
        loc_med = np.median(d_loc[:, 1:], axis=1)  # (N,)
        keep = (edge_lens <= c * loc_med[i_idx]) & (edge_lens <= c * loc_med[j_idx])
        edges = edges[keep]

    # else: "none" 不裁边

    if edges.size == 0:
        return torch.empty((2,0), dtype=torch.long)

    # ---- 4) 打包为 edge_index，并标准化（无自环、无向）----
    edge_index = torch.from_numpy(edges.T).contiguous()  # [2, E]
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=N)
    return edge_index

def edge_index_to_adj(coords_xy: np.ndarray,
                      edge_index: torch.Tensor,
                      weight: str = "binary",      # "binary" | "distance" | "gaussian" | "invdist" 
                      sigma_scale: float = 1.0) -> sp.csr_matrix:
    """
    返回 scipy.sparse.csr_matrix A（N x N），对称、无自环
    """
    X = np.asarray(coords_xy, float)
    N = X.shape[0]
    if edge_index.numel() == 0:
        return sp.csr_matrix((N, N))

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()

    # 边长度
    d = np.linalg.norm(X[row] - X[col], axis=1)

    if weight == "binary":
        w = np.ones_like(d)
    elif weight == "distance":
        # 近的更强（可反向：1/d），但避免0
        w = 1.0 / np.maximum(d, 1e-8)
    elif weight == "invdist":
        # 跟上面 distance 等价，留作可读性别名
        w = 1.0 / np.maximum(d, 1e-8)
    elif weight == "gaussian":
        # 高斯核：exp(-d^2 / (2*sigma^2)), sigma 取自“典型间距”
        nbrs = NearestNeighbors(n_neighbors=min(2, N), metric='euclidean').fit(X)
        d1, _ = nbrs.kneighbors(X)      # (N,2), [:,1]是最近邻距离
        pitch = np.median(d1[:, 1])
        sigma = sigma_scale * pitch
        w = np.exp(- (d ** 2) / (2.0 * sigma ** 2 + 1e-12))
    else:
        raise ValueError(f"Unknown weight type: {weight}")

    A = sp.coo_matrix((w, (row, col)), shape=(N, N))
    # 确保完全对称（有时 to_undirected 已保证，但这里再保险一次）
    A = 0.5 * (A + A.T)
    # 去自环
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()

# def precompute_graph_basis(A: sp.csr_matrix, 
#                            K: int = 128, 
#                            normalized: bool = True, 
#                            eps: float = 1e-8):
#     """
#     预先分解拉普拉斯，返回基 U (N, K_eff)，降低计算复杂度
#     """
#     N = A.shape[0]
#     deg = np.array(A.sum(1)).flatten()

#     # 拉普拉斯
#     if normalized:
#         with np.errstate(divide='ignore'):
#             dinv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
#         Dinv2 = sp.diags(dinv_sqrt)
#         L = sp.eye(N, format='csr') - (Dinv2 @ A @ Dinv2)
#     else:
#         D = sp.diags(deg)
#         L = D - A

#     # 特征分解
#     k_eff = max(2, min(K, N-1))
#     eigvals, U = eigsh(L, k=k_eff, which='SM')

#     # 丢掉 λ≈0 的分量
#     keep = eigvals > eps
#     if keep.sum() < 2:
#         keep[0] = True
#     return U[:, keep]   # (N, K_eff)


def graph_spectral_correlation(U, X_gt, X_pred):
    """
    用预先分解好的 U 来计算所有基因的谱相似度
    X_gt, X_pred: (N_spots, N_genes)
    返回: (N_genes,) 每个基因的余弦相似度
    """
    # 投影到谱基
    X_hat_gt = U.T @ X_gt    # (K_eff, G)
    X_hat_pred = U.T @ X_pred

    # 余弦相似度 (向量化)
    num = np.sum(X_hat_gt * X_hat_pred, axis=0)
    den = np.linalg.norm(X_hat_gt, axis=0) * np.linalg.norm(X_hat_pred, axis=0)
    return num / (den + 1e-12)

# def compute_gsc(adata, ground_truth, pred_value, K=128):

#     XY = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # (N,2)
#     edge_index = build_mnn_edge_index(XY, prune= "global")
#     A = edge_index_to_adj(XY, edge_index, weight="binary", sigma_scale=1.0)
#     U = precompute_graph_basis(A, K=K)
#     all_corr = []
#     for i in range(ground_truth.shape[1]):
#         x = ground_truth[:, i]
#         y = pred_value[:, i]
#         cor = graph_spectral_correlation(U, x, y)
#         all_corr.append(cor)

#     return all_corr

def precompute_graph_basis_w_per(A: sp.csr_matrix, 
                           K: int = 128, 
                           normalized: bool = True, 
                           eps: float = 1e-8):
    """
    预先分解拉普拉斯，返回基 U (N, K_eff)，降低计算复杂度
    """
    N = A.shape[0]
    deg = np.array(A.sum(1)).flatten()

    # 拉普拉斯
    if normalized:
        with np.errstate(divide='ignore'):
            dinv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        Dinv2 = sp.diags(dinv_sqrt)
        L = sp.eye(N, format='csr') - (Dinv2 @ A @ Dinv2)
    else:
        D = sp.diags(deg)
        L = D - A

    # 特征分解
    k_eff = max(2, min(K, N-1))
    eigvals, U = eigsh(L, k=k_eff, which='SM')  # 一次求前 K_max 个低频模式
    
    # 丢掉 λ≈0 的分量
    keep = eigvals > eps
    if keep.sum() < 2:
        keep[0] = True
    return U, keep


def compute_gsc_per(adata, ground_truth, pred_value, K=128, per=0.9):

    XY = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # (N,2)
    edge_index = build_mnn_edge_index(XY, prune= "global")
    A = edge_index_to_adj(XY, edge_index, weight="binary", sigma_scale=1.0)
    U, keep = precompute_graph_basis_w_per(A, K=K)
    all_corr = []
    small_Kt_count = 0
    for i in range(ground_truth.shape[1]):
        x = ground_truth[:, i]
        y = pred_value[:, i]
        X_hat = U.T @ x                            # (K_max, G)

        X_DC = X_hat[~keep]
        total_energy = np.sum(x ** 2) - np.sum(X_DC ** 2)
        X_hat = X_hat[keep]
        energy_per_mode = X_hat ** 2 # 每个频率模式的能量

        cum_frac = np.cumsum(energy_per_mode) / total_energy

        K_t = np.searchsorted(cum_frac, per)  # 返回0-索引的， 最小的（第一个）>=per的位置
        K_t = max(K_t, 1)
        if K_t < 3:
             small_Kt_count += 1
        # U_DC = U[:, ~keep]               # 直流模式
        U_low = U[:, keep][:, :K_t+1]      # 按能量累积截取的低频模式
        # U_use = np.hstack([U_DC, U_low])
        cor = graph_spectral_correlation(U_low, x, y) # 这里加1是因为切片左闭右开
        all_corr.append(cor)

    # print(small_Kt_count)
    return all_corr


# def compute_esc_energy(adata, ground_truth, pred_value, K=128, per=0.9):

#     XY = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # (N,2)
#     edge_index = build_mnn_edge_index(XY, prune= "global")
#     A = edge_index_to_adj(XY, edge_index, weight="binary", sigma_scale=1.0)
#     U, keep = precompute_graph_basis_w_per(A, K=K)
#     all_corr = []
#     small_Kt_count = 0
#     for i in range(ground_truth.shape[1]):
#         x = ground_truth[:, i]
#         y = pred_value[:, i]
#         X_hat = U.T @ x                            # (K_max, G)

#         X_DC = X_hat[~keep]
#         total_energy = np.sum(x ** 2) - np.sum(X_DC ** 2)
#         X_hat = X_hat[keep]
#         energy_per_mode = X_hat ** 2 # 每个频率模式的能量

#         cum_frac = np.cumsum(energy_per_mode) / total_energy

#         K_t = np.searchsorted(cum_frac, per)  # 返回0-索引的， 最小的（第一个）>=per的位置
#         K_t = max(K_t, 1)
#         if K_t < 3:
#              small_Kt_count += 1
#         # U_DC = U[:, ~keep]               # 直流模式
#         U_low = U[:, keep][:, :K_t+1]      # 按能量累积截取的低频模式
#         # U_use = np.hstack([U_DC, U_low])
#         x_energy = x ** 2
#         y_energy = y ** 2
#         cor = graph_spectral_correlation(U_low, x_energy, y_energy) # 这里加1是因为切片左闭右开
#         all_corr.append(cor)

#     print(small_Kt_count)
#     return all_corr
### 使用方法
#test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
#XY = test_ad.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # (N,2)
#edge_index = build_mnn_edge_index(XY, prune= "global") 
#A = edge_index_to_adj(XY, edge_index, weight="binary", sigma_scale=1.0)
#U = precompute_graph_basis(A, K=128)
#all_corr = []
#for i in range(test_count_mtx_selected_genes.shape[1]):
#    x = test_count_mtx_selected_genes.iloc[:, i].values
#    y = preds[:, i]
#    cor = graph_spectral_correlation(U, x, y)
#    all_corr.append(cor)