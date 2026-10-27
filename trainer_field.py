# In[]
import sys, os, random
import numpy as np 
import pandas as pd
import torch
import anndata
import torch.nn as nn
import torch.nn.functional as F

from topo_resampler_field import GeneFiLMRegressor
from topo_field import build_density_map_torch, add_density_grads, topo_kl_from_field, BarrierNet
from build_graph_dens import slide_to_data_field


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 多 GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cudnn 固定化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch_field(loader, model, barrier_net,
                    opt=None, device="cuda",
                    lambda_topo=0.05,
                    tau_b=1.0, tau_kl=0.8,
                    grid_hw=(256,256), sigma=2.0,
                    use_heads=None,
                    E=None):
    """
    Field版本训练：q 来自 barrier field，不依赖 data.edge_log_g。
    需要 batch.pos, batch.cells_xy, batch.slide_hw。
    """
    train = opt is not None
    model.train(mode=train)
    barrier_net.train(mode=train)

    total_loss, total_cnt = 0.0, 0

    for batch in loader:
        batch = batch.to(device)

        pos_xy   = batch.pos            # [N,2]
        cells_xy = batch.cells_xy       # [M,2]
        slide_hw = batch.slide_hw       # [2] (W,H)

        # 1) barrier map
        dens = build_density_map_torch(cells_xy, slide_hw, grid_hw=grid_hw, sigma=sigma)
        inp  = add_density_grads(dens)                 # [1,3,Hg,Wg]
        b_map = barrier_net(inp)                       # [1,1,Hg,Wg]

        # 2) forward model (注意：log_g=None；把 E 传进去)
        pred, (eidx, alpha) = model(
            batch.x, batch.edge_index,
            E=E,
            return_attention_weights=True,
            attn_return="last"
        )

        task_loss = F.mse_loss(pred, batch.y)

        topo_loss = topo_kl_from_field(
            eidx, alpha,
            pos_xy=pos_xy, slide_hw=slide_hw,
            b_map=b_map,
            tau_b=tau_b, tau_kl=tau_kl,
            use_heads=use_heads
        )

        loss = task_loss + lambda_topo * topo_loss

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(barrier_net.parameters()), 5.0)
            opt.step()

        total_loss += float(loss) * batch.y.numel()
        total_cnt  += batch.y.numel()

    return total_loss / max(total_cnt, 1)
# %%

# Data loading

from build_graph_dens import slide_to_data_field
from torch_geometric.loader import DataLoader

local_dir='../hest_data/MEND/'

#MISC
# fn_lst = ["MISC"+str(i) for i in range(131, 143)]
# fn_lst.remove("MISC137")
# fn_lst.remove("MISC131")
# fn_lst.remove("MISC138")

#COAD
#fn_lst = ["MISC"+str(i) for i in range(33, 48)]
#fn_lst.remove("MISC41")
#fn_lst.remove("MISC36")
#fn_lst.remove("MISC42")

#MEND
fn_lst = ["MEND"+str(i) for i in range(139, 154)]
fn_lst.remove("MEND139")
#fn_lst.remove("MEND141")

#skin
#fn_lst = ["NCBI"+str(i) for i in range(505, 510)]
#fn_lst.remove("NCBI505")
#fn_lst.remove("MISC36")
#fn_lst.remove("MISC42")

#liver
#fn_lst = ["NCBI"+str(i) for i in range(834, 840)]
#fn_lst.remove("NCBI834")
#fn_lst.remove("MISC36")
#fn_lst.remove("MISC42")


#ESM embedding loading
dir_protein_embed = "../hest_data/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

protein_embed = torch.load(dir_protein_embed + "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", weights_only = False)
selected_genes = list(np.genfromtxt(local_dir + "processed_data/selected_500gene_list.txt", dtype=str))

keep_idx = []                 # 在 selected_genes 里的保留下标
E_list   = []                 # 临时收集 embedding
for i, g in enumerate(selected_genes):
    emb = protein_embed.get(g, None)
    if emb is None:
        continue
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    elif isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu()
    else:
        raise TypeError(f"Embedding for {g} must be numpy or torch.Tensor, got {type(emb)}")
    E_list.append(emb)
    keep_idx.append(i)

if len(E_list) == 0:
    raise RuntimeError("No gene embeddings found that match selected_genes.")
# 维度一致性检查（可选）
d_g = E_list[0].numel()
assert all(e.numel() == d_g for e in E_list), "Inconsistent embedding dims across genes."

E = torch.stack(E_list, dim=0)                 # [G_avail, d_g]
selected_genes_kept = [selected_genes[i] for i in keep_idx]

dataset = []
for sn in fn_lst:
    d = slide_to_data_field(
        sn,
        gene_list=selected_genes_kept,
        local_dir=local_dir,
        mnn_k=6,
        prune="global",
        prune_c=1.5,
        k_local=6,
        gene_rename=False
    )
    # 这条路线暂时不需要 aug_bank
    # d = attach_aug_bank_to_data(d, sn, local_dir)
    print(sn)
    dataset.append(d)

print("The number of genes used in training is:", len(selected_genes_kept))

# %%
# 建议：field 版本先 batch_size=1（每个 batch 一张 slide 更稳）
train_set, val_set, test_set = dataset, [], []
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False) if val_set else None
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False) if test_set else None

set_seed(2025)
model = GeneFiLMRegressor(
    in_dim=dataset[0].x.size(1),
    hid=512,
    out_dim=len(selected_genes_kept),
    d_gene=E.shape[1],
    num_layers=3,
    heads=4,
    dropout=0.2,
    attn_dim=None,
    ffn_mult=2,
    nonneg="softplus"   # 若监督是 log1p，一般用 "none"
).to(device)


barrier_net = BarrierNet(in_ch=3, hid=32).to(device)

def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))

print(f"Total params(model):      {count_params(model, False):,}")
print(f"Trainable params(model):  {count_params(model, True):,}")

# %%
opt = torch.optim.AdamW(
    list(model.parameters()) + list(barrier_net.parameters()),
    lr=1e-4, weight_decay=1e-5
)

E_dev = E.to(device)

for ep in range(1, 401): 
    tr = run_epoch_field(
        train_loader, model, barrier_net,
        opt=opt, device=device,
        lambda_topo=0.05,     # 建议起步 0.01~0.1
        tau_b=0.5,            # field->logg 尺度先固定
        tau_kl=0.8,           # 建议 0.5~1.5 起步，别太小
        grid_hw=(256,256), sigma=2.0,
        E=E_dev
    )

    if val_loader:
        with torch.no_grad():
            vl = run_epoch_field(
                val_loader, model, barrier_net,
                opt=None, device=device,
                lambda_topo=0.0,    # 验证不加 KL（也可以保留但通常不需要）
                tau_b=1.0, tau_kl=0.8,
                grid_hw=(256,256), sigma=2.0,
                E=E_dev
            )
        if ep % 10 == 0:
            print(f"Epoch {ep:03d} | train loss/elt={tr:.6f} | val loss/elt={vl:.6f}")
    else:
        if ep % 10 == 0:
            print(f"Epoch {ep:03d} | train loss/elt={tr:.6f}")


# %%
print("Barrier grad:", barrier_net.net[-1].weight.grad.abs().mean().item())

@torch.no_grad()
def get_b_map_from_batch(batch, barrier_net, device="cuda", grid_hw=(256,256), sigma=2.0):
    batch = batch.to(device)
    dens = build_density_map_torch(batch.cells_xy, batch.slide_hw, grid_hw=grid_hw, sigma=sigma)
    inp  = add_density_grads(dens)
    b_map = barrier_net(inp)  # [1,1,Hg,Wg]
    return b_map

# 用法：拿 train_loader 里第一张 slide 的 b_map
batch0 = next(iter(train_loader)).to(device)
dens = build_density_map_torch(batch0.cells_xy, batch0.slide_hw, grid_hw=(256,256), sigma=2.0)
b_map = barrier_net(add_density_grads(dens))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(dens[0,0].detach().cpu().numpy()); plt.title("density"); plt.colorbar()
plt.subplot(1,2,2); plt.imshow(b_map[0,0].detach().cpu().numpy()); plt.title("b_map"); plt.colorbar()
plt.show()

# %%
print("logg min/mean/max:", logg.min().item(), logg.mean().item(), logg.max().item())
# %%
###test

#MISC
#sample_name = "MISC131"
#COAD
#sample_name = "MISC36"
#MEND
sample_name = "MEND139"
#skin
#sample_name = "NCBI834"

dataset_test = slide_to_data_field(
        sample_name,
        gene_list=selected_genes_kept,
        local_dir=local_dir,
        mnn_k=6,
        prune="global",
        prune_c=1.5,
        k_local=6,
        gene_rename=False
    )
dataset_test = [dataset_test]
test_loader   = DataLoader(dataset_test, batch_size=1, shuffle=False)

# %%
def predict(loader, model, E, device=device):
    model.eval()
    preds, truths = [], []
    E_dev = E.to(device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, E=E_dev)   # <-- 不传 log_g
            preds.append(out.cpu())
            truths.append(batch.y.cpu())
    return torch.cat(preds, dim=0), torch.cat(truths, dim=0)

# 使用
te_pred, te_truth = predict(test_loader, model, E=E, device=device)
te_pred, te_truth = te_pred.numpy(), te_truth.numpy()
# %%
all_corr = []
for i in range(te_truth.shape[1]):
    x = te_truth[:, i]
    y = te_pred[:, i]
    cor = np.corrcoef(x, y)[0][1]
    all_corr.append(cor)

# %%
# evaluation metrics

# PCC
print("PCC-10: ", np.mean(sorted(all_corr)[::-1][:10]))
print("PCC-50: ", np.mean(sorted(all_corr)[::-1][:50]))
print("PCC-200: ", np.mean(sorted(all_corr)[::-1][:200]))
# MSE, MAE
print("MSE: ", np.mean((te_truth - te_pred)**2))
print("MAE: ", np.mean(np.abs(te_truth - te_pred)))


# %%
test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
# gsc
from gsc_metric import compute_gsc_per

def see(test_ad, pred, gt, per):
    all_corr = compute_gsc_per(test_ad, ground_truth=gt, pred_value=pred, K=gt.shape[0], per=per)
    gsc10 = np.mean(sorted(all_corr)[::-1][:10])
    gsc50 = np.mean(sorted(all_corr)[::-1][:50])
    gsc200 = np.mean(sorted(all_corr)[::-1][:200])
    gsc_all = np.mean(all_corr)
    print(f"[GSC] top-10: {gsc10:.4f} | top-50: {gsc50:.4f} | top-200: {gsc200:.4f} | all: {gsc_all:.4f}")

for per in [0.9]:
    print("-----------------------------------------------------")
    print(f"percentage {per * 100}%")

    see(test_ad, pred=te_pred, gt=te_truth, per=per)

# %%
#all_corr[109]
all_corr_gsc = compute_gsc_per(test_ad, ground_truth=te_truth, pred_value=te_pred, K=te_truth.shape[0], per=0.9)
all_corr_gsc[109]
# %%
import matplotlib.pyplot as plt

def compute_gsc_metrics(adata, pred, gt, per):
    """
    返回 (gsc10, gsc50, gsc200, gsc_all)
    """
    all_corr = compute_gsc_per(
        adata, ground_truth=gt, pred_value=pred,
        K=gt.shape[0], per=per
    )
    all_corr = np.asarray(all_corr)
    sorted_corr = np.sort(all_corr)[::-1]

    gsc10  = np.mean(sorted_corr[:10])
    gsc50  = np.mean(sorted_corr[:50])
    gsc200 = np.mean(sorted_corr[:200])
    gsc_all = np.mean(all_corr)

    return gsc10, gsc50, gsc200, gsc_all


def plot_gsc_curves(adata, pred, gt,
                    per_values=None,
                    title="GSC@K vs percentage"):
    """
    在同一张图里画 GSC@10 / GSC@50 / GSC@200 的曲线.

    per_values: list 或 np.array, 比如 np.linspace(0.1, 1.0, 10)
    """
    if per_values is None:
        per_values = np.linspace(0.1, 1.0, 10)
    per_values = np.asarray(per_values)

    gsc10_list  = []
    gsc50_list  = []
    gsc200_list = []

    for per in per_values:
        g10, g50, g200, gall = compute_gsc_metrics(adata, pred, gt, per)
        gsc10_list.append(g10)
        gsc50_list.append(g50)
        gsc200_list.append(g200)
        print(f"per={per:.2f} | GSC@10={g10:.4f} | GSC@50={g50:.4f} | GSC@200={g200:.4f}")

    gsc10_list  = np.asarray(gsc10_list)
    gsc50_list  = np.asarray(gsc50_list)
    gsc200_list = np.asarray(gsc200_list)

    # 画三条曲线在同一张图
    plt.figure(figsize=(5,4))
    x = per_values * 100  # 转成百分比显示会直观一点

    plt.plot(x, gsc10_list,  marker='o', label='GSC@10')
    plt.plot(x, gsc50_list,  marker='s', label='GSC@50')
    plt.plot(x, gsc200_list, marker='^', label='GSC@200')

    plt.xlabel("Percentage of spots (%)")
    plt.ylabel("GSC@K")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 调用示例
per_values = np.linspace(0.1, 1.0, 10)  # 或者 [0.5, 0.6, ..., 1.0]
plot_gsc_curves(test_ad, te_pred, te_truth, per_values=per_values)


#%%

import scanpy as sc
test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
#test_ad.var_names = test_ad.var["SYMBOL"]
test_ad.var_names_make_unique()
# %%
have = [g for g in selected_genes_kept if g in test_ad.var_names]
test_ad = test_ad[:, have]
# %%
test_ad.layers["predicted"] = te_pred
test_ad.layers['truth'] = te_truth

lib = next(iter(test_ad.uns["spatial"].keys()))              # -> 'ST'
img_keys = list(test_ad.uns["spatial"][lib]["images"].keys())# -> ['downscaled_fullres']
print("library_id:", lib, "| img_keys:", img_keys)
# %%
#预测值
num = 165
sc.pl.spatial(
    test_ad,
    color='AZGP1',
    layer="predicted",          # 你的预测矩阵在这里
    library_id=lib,             # 'ST'
    img_key=img_keys[0],        # 'downscaled_fullres'
    spot_size=220,
    alpha_img=0,
    vmin = 0,
    vmax = 3.5,
    cmap="coolwarm"
)
#真实值
sc.pl.spatial(
    test_ad,
    color='AZGP1', 
    layer="truth",  
    library_id=lib,             # 'ST'
    img_key=img_keys[0],        # 'downscaled_fullres'
    spot_size=220,
    alpha_img=0,
    vmin = 0,
    vmax = 4,
    cmap="coolwarm"
)

sc.pl.spatial(
    test_ad,
    color=selected_genes_kept[num], 
    layer="truth",  
    library_id=lib,             # 'ST'
    img_key=img_keys[0],        # 'downscaled_fullres'
    spot_size=0,
    alpha_img=1,
    vmin = 0,
    vmax = 4,
    cmap="coolwarm"
)

# %%
#zero shot

def make_zero_shot_splits(local_dir,
                          selected_genes_kept,
                          protein_embed,                 # dict: gene -> np.ndarray / torch.Tensor
                          limits=(100, 250, 500),
                          strict_fill=False,             # True: 保证拿到 N 个“有 embedding”的基因
                          dtype=torch.float32):
    """
    返回一个 dict: {N: (genes_list, E_tensor)}，顺序与输入列表一致。
    - genes_list: 长度 ≤ N（slice 模式）或 = N（strict_fill=True 且可用足够多）
    - E_tensor: [len(genes_list), d_g]
    """
    # 1) 读入候选 zeroshot 列表（有序）
    zs_all = list(np.genfromtxt(local_dir + "processed_data/selected_1000gene_list.txt", dtype=str))
    kept = set(selected_genes_kept)

    # 2) 按原顺序做差 & 去重保序
    diff = [g for g in zs_all if g not in kept]
    seen = set()
    diff = [g for g in diff if not (g in seen or seen.add(g))]

    # 3) 把可用 embedding 预取出来（保持 diff 顺序）
    def _to_tensor(emb):
        if emb is None:
            return None
        if isinstance(emb, np.ndarray):
            t = torch.from_numpy(emb)
        elif isinstance(emb, torch.Tensor):
            t = emb.detach().cpu()
        else:
            return None
        return t.reshape(-1).to(dtype)

    available = []   # list[(gene, tensor)]
    d_g = None
    for g in diff:
        t = _to_tensor(protein_embed.get(g, None))
        if t is None:
            continue
        if d_g is None:
            d_g = t.numel()
        if t.numel() != d_g:
            # 维度不一致的丢掉（也可选择 raise）
            continue
        available.append((g, t))

    if d_g is None:
        raise RuntimeError("No valid gene embeddings found for zero-shot candidates.")

    # 4) 组装各个 N 的子集
    out = {}
    # 便于快速按名字取 tensor
    avail_map = {g: t for g, t in available}

    for N in limits:
        if strict_fill:
            sel = available[:min(N, len(available))]
        else:
            # 先切片，再过滤缺失 embedding 的
            want = set(diff[:min(N, len(diff))])
            sel = [(g, avail_map[g]) for g in diff if (g in want and g in avail_map)]

        genes = [g for g, _ in sel]
        if len(genes) == 0:
            E = torch.empty((0, d_g), dtype=dtype)
        else:
            E = torch.stack([t for _, t in sel], dim=0)  # [len(genes), d_g]
        out[N] = (genes, E)

    return out

# ===== 用法 =====
# selected_genes_kept: 你训练用的基因列表
splits = make_zero_shot_splits(local_dir, selected_genes_kept, protein_embed,
                               limits=(100, 250, 500), strict_fill=False)

zs100_genes, E100 = splits[100]
zs250_genes, E250 = splits[250]
zs500_genes, E500 = splits[500]

print(len(zs100_genes), E100.shape)
print(len(zs250_genes), E250.shape)
print(len(zs500_genes), E500.shape)

# %%
############top semantic gene################
import torch
import numpy as np
from typing import List, Dict, Iterable, Tuple

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).reshape(-1).float()
    return None

def _l2norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)

def nearest_gene_splits_by_cosine(
    E_train: torch.Tensor,                      # [G_train, d]
    selected_genes_kept: List[str],            # 长度 = G_train
    protein_embed: Dict[str, np.ndarray],      # gene -> embedding
    pool_genes: Iterable[str] = None,          # 可选：候选集合，不给则用 protein_embed 的所有键
    N_list: Tuple[int, ...] = (100, 250, 500),
    method: str = "avg",                       # "centroid" | "avg" | "max"
) -> Dict[int, Dict[str, object]]:
    """
    返回 {N: {"genes": [..], "E": Tensor[G_N, d], "scores": Tensor[G_N]}}
    - genes 顺序与相似度从高到低一致
    - E 的行顺序与 genes 对齐
    """
    assert E_train.dim() == 2, "E_train must be [G_train, d]"
    d = E_train.size(1)
    train_set = set(selected_genes_kept)

    # 1) 构建候选池（去掉训练集，且维度必须匹配）
    if pool_genes is None:
        pool_genes = protein_embed.keys()

    cand_names, cand_vecs = [], []
    for g in pool_genes:
        if g in train_set:
            continue
        t = _to_tensor(protein_embed.get(g, None))
        if t is None or t.numel() != d:
            continue
        cand_names.append(g)
        cand_vecs.append(t)
    if len(cand_vecs) == 0:
        raise RuntimeError("No valid candidate embeddings with matching dim were found.")

    C = torch.stack(cand_vecs, dim=0)       # [M, d]
    Cn = _l2norm(C)                         # 归一化
    En = _l2norm(E_train.float())           # 归一化

    # 2) 相似度聚合
    method = method.lower()
    if method == "centroid":
        center = _l2norm(En.mean(dim=0, keepdim=True))    # [1, d]
        sims = (Cn * center).sum(dim=-1)                  # [M]
    elif method == "avg":
        sims = (Cn @ En.T).mean(dim=1)                    # [M]
    elif method == "max":
        sims = (Cn @ En.T).max(dim=1).values              # [M]
    else:
        raise ValueError("method must be 'centroid' | 'avg' | 'max'")

    # 3) 排序并切片
    order = torch.argsort(sims, descending=True)
    out = {}
    for N in N_list:
        K = min(N, order.numel())
        idx = order[:K]
        out[N] = {
            "genes": [cand_names[i] for i in idx.tolist()],
            "E": C[idx],                 # [K, d]
            "scores": sims[idx],         # 余弦相似度（越大越近）
        }
    return out

zs_all = list(np.genfromtxt(local_dir + "processed_data/selected_1000gene_list.txt", dtype=str))
kept = set(selected_genes_kept)

    # 2) 按原顺序做差 & 去重保序
diff = [g for g in zs_all if g not in kept]
seen = set()
diff = [g for g in diff if not (g in seen or seen.add(g))]

splits = nearest_gene_splits_by_cosine(
    E_train=E,
    selected_genes_kept=selected_genes_kept,
    protein_embed=protein_embed,
    pool_genes=diff,# 若只想在某个列表里找（比如 zeroshot_genes），传 pool_genes=zeroshot_genes
    N_list=(100, 250, 500),
    method="avg"   # 可换 "centroid" 或 "max"
)

near100_genes, E100, s100 = splits[100]["genes"], splits[100]["E"], splits[100]["scores"]
near250_genes, E250, s250 = splits[250]["genes"], splits[250]["E"], splits[250]["scores"]
near500_genes, E500, s500 = splits[500]["genes"], splits[500]["E"], splits[500]["scores"]

# import numpy as np

# zeroshot_genes = list(np.genfromtxt(local_dir + "processed_data/selected_1000gene_list.txt", dtype=str))
# kept = set(selected_genes_kept)  # 只把已选基因放入集合，O(1) 查询

# # 按 zeroshot_genes 原顺序保留“新增”的部分
# diff = [g for g in zeroshot_genes if g not in kept]

# zs_top100 = diff[:100]
# zs_top250 = diff[:250]
# zs_top500 = diff[:500]

# keep_idx = []                 # 在 selected_genes 里的保留下标
# E_list   = []                 # 临时收集 embedding
# for i, g in enumerate(zeroshot_genes):
#     emb = protein_embed.get(g, None)
#     if emb is None:
#         continue
#     if isinstance(emb, np.ndarray):
#         emb = torch.from_numpy(emb)
#     elif isinstance(emb, torch.Tensor):
#         emb = emb.detach().cpu()
#     else:
#         raise TypeError(f"Embedding for {g} must be numpy or torch.Tensor, got {type(emb)}")
#     E_list.append(emb)
#     keep_idx.append(i)

# if len(E_list) == 0:
#     raise RuntimeError("No gene embeddings found that match selected_genes.")
# # 维度一致性检查（可选）
# d_g = E_list[0].numel()
# assert all(e.numel() == d_g for e in E_list), "Inconsistent embedding dims across genes."

# E = torch.stack(E_list, dim=0)                 # [G_avail, d_g]
# zeroshot_genes_kept = [zeroshot_genes[i] for i in keep_idx]

# %%
dataset_test = slide_to_data_with_gate(sample_name, gene_list=near500_genes, local_dir=local_dir, theta1=1, theta2=50, theta3=100, gamma_d=2, beta_d=2, alpha_d=2, alpha=0.7, gene_rename=True)
dataset_test = [dataset_test]
test_loader   = DataLoader(dataset_test, batch_size=1, shuffle=False)
# %%

def predict(loader, model, E, device=device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            log_g = getattr(batch, "edge_log_g", None)   # 没有该字段时为 None → 等价不加门控
            out = model(batch.x, batch.edge_index, log_g=log_g, E = E.to(device))
            preds.append(out.cpu())
            truths.append(batch.y.cpu())
    return torch.cat(preds, dim=0), torch.cat(truths, dim=0)

te_pred, te_truth = predict(test_loader, model, E500)
te_pred, te_truth = te_pred.numpy(), te_truth.numpy()
# %%
all_corr = []
for i in range(te_truth.shape[1]):
    x = te_truth[:, i]
    y = te_pred[:, i]
    cor = np.corrcoef(x, y)[0][1]
    all_corr.append(cor)

# %%
# evaluation metrics
test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
test_ad.var_names = test_ad.var["SYMBOL"]
test_ad.var_names_make_unique()
have = [g for g in near500_genes if g in test_ad.var_names]
test_ad = test_ad[:, have]

# %%
test_ad.layers["predicted"] = te_pred
test_ad.layers['truth'] = te_truth

lib = next(iter(test_ad.uns["spatial"].keys()))              # -> 'ST'
img_keys = list(test_ad.uns["spatial"][lib]["images"].keys())# -> ['downscaled_fullres']
print("library_id:", lib, "| img_keys:", img_keys)
# %%
# PCC
print("PCC-all: ", np.mean(all_corr))
# MSE, MAE
print("MSE: ", np.mean((te_truth - te_pred)**2))
print("MAE: ", np.mean(np.abs(te_truth - te_pred)))

from gsc_metric import compute_gsc_per

def see(test_ad, pred, gt, per):
    all_corr = compute_gsc_per(test_ad, ground_truth=gt, pred_value=pred, K=gt.shape[0], per=per)
    gsc_all = np.mean(all_corr)
    print(f"[GSC] all: {gsc_all:.4f}")

for per in [0.9, 0.87,0.92]:
    print("-----------------------------------------------------")
    print(f"percentage {per * 100}%")

    see(test_ad, pred=te_pred, gt=te_truth, per=per)


# %%

# from gsc_metric import compute_gsc
# test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
# all_corr = compute_gsc(adata=test_ad, ground_truth=te_truth, pred_value=te_pred, K=256)
# # PCC
# print("gsc-10: ", np.mean(sorted(all_corr)[::-1][:10]))
# print("gsc-50: ", np.mean(sorted(all_corr)[::-1][:50]))
# print("gsc-200: ", np.mean(sorted(all_corr)[::-1][:200]))
# %%
sc.pl.spatial(
    test_ad,
    color=near500_genes[436],
    layer="predicted",          # 你的预测矩阵在这里
    library_id=lib,             # 'ST'
    img_key=img_keys[0],        # 'downscaled_fullres'
    spot_size=240,
    alpha_img=0,
    vmin = 0,
    vmax = 3,
    cmap="coolwarm"
)
#真实值
sc.pl.spatial(
    test_ad,
    color=near500_genes[436], 
    layer="truth",  
    library_id=lib,             # 'ST'
    img_key=img_keys[0],        # 'downscaled_fullres'
    spot_size=240,
    alpha_img=0,
    vmin = 0,
    vmax = 3,
    cmap="coolwarm"
)
# %%
np.argsort(all_corr)[-20:][::-1]

# %%
import numpy as np

def mean_std(values):

    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std  = float(values.std(ddof=1))  # ddof=1 → 无偏估计
    return mean, std

runs = [0.2546,0.5333,0.5724,0.4390,0.3606,0.2853,0.5410,0.3928,0.3888,0.4067,0.4756,0.5171,0.4460,0.2409,0.3043]
mean, std = mean_std(runs)
print(mean, std)
print(len(runs))
# %%
