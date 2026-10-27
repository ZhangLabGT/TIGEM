import os
import numpy as np
import pandas as pd
import torch
import anndata
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected

def load_global_cells_xy_from_geo(cell_parquet_path: str) -> np.ndarray:
    """
    读取 cellvit geo-parquet，返回全图细胞质心坐标 (M,2) in fullres pixel coords.
    """
    gdf = gpd.read_parquet(cell_parquet_path)
    if "centroid" not in gdf.columns:
        gdf = gdf.copy()
        gdf["centroid"] = gdf.geometry.centroid
    x = gdf["centroid"].x.to_numpy(dtype=float)
    y = gdf["centroid"].y.to_numpy(dtype=float)
    return np.stack([x, y], axis=1)  # [M,2]


def build_mnn_edge_index(coords_xy: np.ndarray,
                         k: int = 6,
                         prune: str = "global",
                         c: float = 1.5,
                         k_local: int = 6) -> torch.Tensor:
    """
    只构图：kNN -> MNN（可选裁边），输出无向无自环 edge_index [2,E]
    """
    X = np.asarray(coords_xy, dtype=float)
    N = X.shape[0]
    if N == 0:
        return torch.empty((2,0), dtype=torch.long)

    k_eff = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(X)
    dists, idxs = nbrs.kneighbors(X)
    idxs = idxs[:, 1:]  # remove self

    neigh_sets = [set(row.tolist()) for row in idxs]
    undirected_edges = set()
    for i in range(N):
        for j in idxs[i]:
            if i in neigh_sets[j]:
                u, v = (i, j) if i < j else (j, i)
                undirected_edges.add((u, v))

    if not undirected_edges:
        for i in range(N):
            for j in idxs[i]:
                u, v = (i, j) if i < j else (j, i)
                undirected_edges.add((u, v))

    edges = np.array(list(undirected_edges), dtype=np.int64)
    if edges.size == 0:
        return torch.empty((2,0), dtype=torch.long)

    # optional prune
    i_idx, j_idx = edges[:,0], edges[:,1]
    edge_lens = np.linalg.norm(X[i_idx] - X[j_idx], axis=1)

    if prune == "global":
        pitch = np.median(dists[:, 1])
        tau = c * pitch
        edges = edges[edge_lens <= tau]
    elif prune == "local":
        k_eff_loc = min(k_local + 1, N)
        d_loc, _ = NearestNeighbors(n_neighbors=k_eff_loc, metric="euclidean").fit(X).kneighbors(X)
        loc_med = np.median(d_loc[:, 1:], axis=1)
        keep = (edge_lens <= c * loc_med[i_idx]) & (edge_lens <= c * loc_med[j_idx])
        edges = edges[keep]

    if edges.size == 0:
        return torch.empty((2,0), dtype=torch.long)

    edge_index = torch.from_numpy(edges.T).contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=N)
    return edge_index


def slide_to_data_field(sample_name: str,
                        gene_list,
                        local_dir: str,
                        mnn_k: int = 6,
                        prune: str = "global",
                        prune_c: float = 1.5,
                        k_local: int = 6,
                        gene_rename: bool = True) -> Data:
    """
    构建 field 版本的 Data：不包含 edge_log_g。
    """
    ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
    if gene_rename:
        ad.var_names = ad.var["SYMBOL"]
        ad.var_names_make_unique()

    # spot coords (fullres pixel)
    XY = ad.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # [N,2]

    # build graph
    edge_index = build_mnn_edge_index(XY, k=mnn_k, prune=prune, c=prune_c, k_local=k_local)

    # node features (PVM embeddings)
    img_ebd_uni = torch.load(os.path.join(local_dir, "processed_data/spot_uni_ebd", f"{sample_name}_uni.pt"), map_location="cpu")
    img_ebd_conch = torch.load(os.path.join(local_dir, "processed_data/spot_conch_ebd", f"{sample_name}_conch.pt"), map_location="cpu")
    x = torch.cat([img_ebd_uni, img_ebd_conch], dim=1).float()  # [N,2048]

    # labels
    have = [g for g in gene_list if g in ad.var_names]
    Y = pd.DataFrame(ad[:, have].X.toarray(), columns=have, index=np.arange(ad.n_obs))
    Y = Y.reindex(columns=gene_list, fill_value=0.0)
    y = torch.tensor(np.log1p(Y.values), dtype=torch.float32)

    # global cells coords
    cell_parquet = os.path.join(local_dir, "cellvit_seg", f"{sample_name}_cellvit_seg.parquet")
    all_cells_xy = load_global_cells_xy_from_geo(cell_parquet)  # [M,2]

    # slide_hw for coordinate normalization (rough but sufficient)
    max_x = float(np.nanmax([XY[:,0].max(), all_cells_xy[:,0].max()]))
    max_y = float(np.nanmax([XY[:,1].max(), all_cells_xy[:,1].max()]))
    slide_hw = torch.tensor([int(np.ceil(max_x))+1, int(np.ceil(max_y))+1], dtype=torch.long)  # [W,H]

    data = Data(x=x, edge_index=edge_index, y=y)
    data.pos = torch.from_numpy(XY).float()                 # [N,2]
    data.cells_xy = torch.from_numpy(all_cells_xy).float()  # [M,2]
    data.slide_hw = slide_hw                                # [2]
    return data