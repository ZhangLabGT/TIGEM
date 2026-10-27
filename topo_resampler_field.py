# In[]
import os, random
import anndata
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from topo_net import GATv2ConvPlain, _FFN
from film_modulation import FiLMHead, FiLMHead_2x, FiLMHead_2x_CA


class GATFFNBlock(nn.Module):
    """LN -> GATv2 (plain) -> ELU/Drop -> Residual -> LN -> FFN -> Residual"""
    def __init__(self, in_dim, hid, heads=4, dropout=0.2, attn_dim=None, ffn_mult=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_dim)
        self.gat = GATv2ConvPlain(
            in_channels=in_dim, out_channels=hid,
            heads=heads, attn_dim=attn_dim, concat=False,
            dropout=dropout, add_self_loops_flag=True
        )
        self.drop = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, hid) if in_dim != hid else nn.Identity()

        self.ln2 = nn.LayerNorm(hid)
        self.ffn = _FFN(hid, mult=ffn_mult, dropout=dropout)

    def forward(self, x, edge_index, return_attention_weights=False):
        x_in = x
        if return_attention_weights:
            x_gat, (eidx, alpha) = self.gat(self.ln1(x), edge_index, return_attention_weights=True)
        else:
            x_gat = self.gat(self.ln1(x), edge_index, return_attention_weights=False)
            eidx = alpha = None

        x_gat = self.drop(F.elu(x_gat))
        h = self.res_proj(x_in) + x_gat
        h = h + self.ffn(self.ln2(h))

        if return_attention_weights:
            return h, (eidx, alpha)
        return h


class TopoEncoder(nn.Module):
    def __init__(self, in_dim, hid, num_layers=3, heads=4, dropout=0.2, attn_dim=None, ffn_mult=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        din = in_dim
        for _ in range(num_layers):
            self.blocks.append(GATFFNBlock(din, hid, heads=heads, dropout=dropout,
                                           attn_dim=attn_dim, ffn_mult=ffn_mult))
            din = hid

    def forward(self, x, edge_index, return_attention_weights=False):
        attns = []
        for blk in self.blocks:
            if return_attention_weights:
                x, att = blk(x, edge_index, return_attention_weights=True)
                attns.append(att)
            else:
                x = blk(x, edge_index, return_attention_weights=False)
        return (x, attns) if return_attention_weights else x


class GeneFiLMRegressor(nn.Module):
    def __init__(self, in_dim=2048, hid=512, out_dim=200, d_gene=1280,
                 num_layers=3, heads=4, dropout=0.2, attn_dim=None, ffn_mult=2,
                 nonneg: str="none"):
        super().__init__()
        self.encoder = TopoEncoder(in_dim, hid, num_layers=num_layers, heads=heads,
                                   dropout=dropout, attn_dim=attn_dim, ffn_mult=ffn_mult)

        # 你现在用哪个 head 就留哪个
        self.head = FiLMHead(d_model=hid, d_gene=d_gene, nonneg_output=(nonneg=="softplus"))
        self.nonneg = nonneg

    @torch.no_grad()
    def set_gene_embeddings(self, E: torch.Tensor):
        self.head.set_gene_embeddings(E)

    def forward(self, x, edge_index, E=None, return_attention_weights=False, attn_return="last"):
        if return_attention_weights:
            H, attns = self.encoder(x, edge_index, return_attention_weights=True)
            Y, _ = self.head(H, E=E)
            return (Y, attns) if attn_return=="all" else (Y, attns[-1])
        else:
            H = self.encoder(x, edge_index, return_attention_weights=False)
            Y, _ = self.head(H, E=E)
            return Y
# class TopoGraphSampler(nn.Module):
#     """
#     两层 GATv2（带拓扑门控）+ 每层后接 Pre-Norm 残差 FFN，再接线性回归头。
#     - Block1: LN(in) -> GAT1 -> ELU -> Dropout -> Residual(投影) -> LN -> FFN -> Residual
#     - Block2: LN -> GAT2 -> ELU -> Dropout -> Residual(投影) -> LN -> FFN -> Residual
#     """
#     def __init__(self, in_dim=2048, hid=512, out_dim=200,
#                  heads1=4, heads2=1, dropout=0.2,
#                  gating_heads1=(0,1,2,3), gating_heads2=(),
#                  gate_scale1=10, gate_scale2=1.0,
#                  learnable_gate=True, attn_dim=512*2,
#                  ffn_mult1: int = 2, ffn_mult2: int = 2):
#         super().__init__()

#         # ---- GAT 层 ----
#         self.gat1 = GATv2ConvGated(
#             in_dim, hid, heads=heads1, attn_dim=attn_dim, concat=True,
#             dropout=dropout, gating_heads=list(gating_heads1),
#             gate_scale=gate_scale1, learnable_gate=learnable_gate
#         )
#         self.gat2 = GATv2ConvGated(
#             hid*heads1, hid, heads=heads2, attn_dim=attn_dim, concat=False,
#             dropout=dropout, gating_heads=list(gating_heads2),
#             gate_scale=gate_scale2, learnable_gate=learnable_gate
#         )

#         # ---- Pre-Norm + 残差投影（维度不一致时使用） ----
#         self.ln1a = nn.LayerNorm(in_dim)
#         self.ln1b = nn.LayerNorm(hid*heads1)
#         self.res1 = nn.Linear(in_dim, hid*heads1) if in_dim != hid*heads1 else nn.Identity()

#         self.ln2a = nn.LayerNorm(hid*heads1)
#         self.ln2b = nn.LayerNorm(hid)
#         self.res2 = nn.Linear(hid*heads1, hid) if (hid*heads1) != hid else nn.Identity()

#         # ---- FFN 模块 ----
#         self.ffn1 = _FFN(hid*heads1, mult=ffn_mult1, dropout=dropout)
#         self.ffn2 = _FFN(hid,         mult=ffn_mult2, dropout=dropout)

#         # ---- 回归头（与你原来保持一致）----
#         #self.lin = nn.Sequential(
#         #    nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout),
#         #    nn.Linear(hid, out_dim)
#         #)

#         self.dropout = dropout

#     def forward(self, x, edge_index, log_g=None, return_attention_weights=False):
#         # ===== Block 1 =====
#         if return_attention_weights:
#             x1_core, (eidx1, alpha1) = self.gat1(self.ln1a(x), edge_index, log_g=log_g, return_attention_weights=True)
#         else:
#             x1_core = self.gat1(self.ln1a(x), edge_index, log_g=log_g, return_attention_weights=False)
#             eidx1 = alpha1 = None

#         x1_core = F.elu(x1_core)
#         x1_core = F.dropout(x1_core, p=self.dropout, training=self.training)
#         x1 = self.res1(x) + x1_core                     # 残差（维度投影）
#         x1 = x1 + self.ffn1(self.ln1b(x1))              # Pre-Norm FFN 残差

#         # ===== Block 2 =====
#         x2_core = self.gat2(
#             self.ln2a(x1), edge_index,
#             log_g=log_g if self.gat2.gating_mask.any() else None,
#             return_attention_weights=False
#         )
#         x2_core = F.elu(x2_core)
#         x2_core = F.dropout(x2_core, p=self.dropout, training=self.training)
#         h = self.res2(x1) + x2_core                    # 残差（维度投影）
#         h = h + self.ffn2(self.ln2b(h))              # Pre-Norm FFN 残差

#         #out = self.lin(x2)                              # 最终回归

#         if return_attention_weights:
#             # 返回第一层注意力，方便做拓扑 KL 正则
#             return h, (eidx1, alpha1)
#         return h

#     def set_gate_scale(self, val: float, layer: int = 1, heads=None):
#         if layer == 1: self.gat1.set_gate_scale(val, heads)
#         elif layer == 2: self.gat2.set_gate_scale(val, heads)

# %%
###training
###training
# from build_graph import slide_to_data_with_gate
# from torch_geometric.loader import DataLoader

# local_dir='../hest_data/MISC/'
# fn_lst = ["MISC"+str(i) for i in range(131, 142)]
# fn_lst.remove("MISC137")
# selected_genes = list(np.genfromtxt(local_dir + "processed_data/selected_200gene_list.txt", dtype=str))

# dataset = []
# for sn in fn_lst:
#     d = slide_to_data_with_gate(sn, gene_list=selected_genes, local_dir=local_dir,
#                                 theta1=10, theta2=50, theta3=100, gamma_d=2, beta_d=2, alpha_d=2, alpha=0.7)
#     d = attach_aug_bank_to_data(d, sn, local_dir)  # 若各 slide 的 T 不同且想拼 batch，设一个统一的 T_fix
#     dataset.append(d)

# # %%
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# train_set, val_set, test_set = dataset, [], []
# train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
# val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False) if val_set else None
# test_loader  = DataLoader(test_set,  batch_size=2, shuffle=False) if test_set else None
# model = GAT2RegressorGated(in_dim=dataset[0].x.size(1), out_dim=len(selected_genes),
#                           heads1=4, heads2=1, dropout=0.2).to(device)
# opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# def run_epoch(loader, model, aug_mode="replace", lam_max=0.5, p_aug = 0, opt=None, device="cuda"):
#     train = opt is not None
#     model.train(mode=train)

#     total_loss, total_cnt = 0.0, 0
#     for batch in loader:
#         batch = batch.to(device)

#         # 取出 log_g（若数据里没有该字段，则传 None，模型会当成全0门控）
#         log_g = getattr(batch, "edge_log_g", None)

#         #if train:
#             # 用增强视图（或混合）作为当前输入特征
#         #    x_in = sample_aug_x(batch, mode=aug_mode, lam_max=lam_max, p_aug=p_aug)
#         #else:
#         x_in = batch.x0 if hasattr(batch, "x0") else batch.x

#         pred = model(x_in, batch.edge_index, log_g=log_g)   # [N_sum, G]
#         loss = F.mse_loss(pred, batch.y)                       

#         if train:
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             opt.step()

#         total_loss += float(loss) * batch.y.numel()
#         total_cnt  += batch.y.numel()

#     return total_loss / max(total_cnt, 1)

# for ep in range(1, 401):
#     tr = run_epoch(train_loader, model, opt=opt, device=device)   # 训练
#     if val_loader:
#         with torch.no_grad():
#             vl = run_epoch(val_loader, model, opt=None, device=device)  # 验证
#         if ep % 10 == 0:
#             print(f"Epoch {ep:03d} | train MSE/elt={tr:.6f} | val MSE/elt={vl:.6f}")
#     else:
#         if ep % 10 == 0:
#             print(f"Epoch {ep:03d} | train MSE/elt={tr:.6f}")
# # %%
# ###test

# sample_name = "MISC142"

# dataset_test = slide_to_data_with_gate(sample_name, gene_list=selected_genes, local_dir=local_dir, theta1=1, theta2=50, theta3=100, gamma_d=2, beta_d=2, alpha_d=2, alpha=0.7)
# dataset_test = [dataset_test]
# test_loader   = DataLoader(dataset_test, batch_size=1, shuffle=False)

# # %%
# def predict(loader, model, device="cuda:1"):
#     model.eval()
#     preds, truths = [], []
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             log_g = getattr(batch, "edge_log_g", None)   # 没有该字段时为 None → 等价不加门控
#             out = model(batch.x, batch.edge_index, log_g=log_g)
#             preds.append(out.cpu())
#             truths.append(batch.y.cpu())
#     return torch.cat(preds, dim=0), torch.cat(truths, dim=0)

# # 使用
# te_pred, te_truth = predict(test_loader, model)
# te_pred, te_truth = te_pred.numpy(), te_truth.numpy()
# # %%
# all_corr = []
# for i in range(te_truth.shape[1]):
#     x = te_truth[:, i]
#     y = te_pred[:, i]
#     cor = np.corrcoef(x, y)[0][1]
#     all_corr.append(cor)

# # evaluation metrics

# # PCC
# print("PCC-10: ", np.mean(sorted(all_corr)[::-1][:10]))
# print("PCC-50: ", np.mean(sorted(all_corr)[::-1][:50]))
# print("PCC-200: ", np.mean(sorted(all_corr)[::-1][:200]))
# # MSE, MAE
# print("MSE: ", np.mean((te_truth - te_pred)**2))
# print("MAE: ", np.mean(np.abs(te_truth - te_pred)))

# # %%
# from gsc_metric import compute_gsc

# test_ad = anndata.read_h5ad(os.path.join(local_dir, "st", f"{sample_name}.h5ad"))
# all_corr = compute_gsc(adata=test_ad, ground_truth=te_truth, pred_value=te_pred, K=256)
# # PCC
# print("gsc-10: ", np.mean(sorted(all_corr)[::-1][:10]))
# print("gsc-50: ", np.mean(sorted(all_corr)[::-1][:50]))
# print("gsc-200: ", np.mean(sorted(all_corr)[::-1][:200]))
# # %%
# ### 可视化结果
# import scanpy as sc
# test_ad.var_names = test_ad.var["SYMBOL"]
# test_ad.var_names_make_unique()
# # %%
# have = [g for g in selected_genes if g in test_ad.var_names]
# test_ad = test_ad[:, have]
# # %%
# test_ad.layers["predicted"] = te_pred
# test_ad.layers['truth'] = te_truth

# lib = next(iter(test_ad.uns["spatial"].keys()))              # -> 'ST'
# img_keys = list(test_ad.uns["spatial"][lib]["images"].keys())# -> ['downscaled_fullres']
# print("library_id:", lib, "| img_keys:", img_keys)
# # %%
# #预测值
# sc.pl.spatial(
#     test_ad,
#     color=selected_genes[60],
#     layer="predicted",          # 你的预测矩阵在这里
#     library_id=lib,             # 'ST'
#     img_key=img_keys[0],        # 'downscaled_fullres'
#     spot_size=180,
#     alpha_img=1,
#     vmin = 0.5,
#     vmax = 1.5,
#     cmap="viridis"
# )
# #真实值
# sc.pl.spatial(
#     test_ad,
#     color=selected_genes[60], 
#     layer="truth",  
#     library_id=lib,             # 'ST'
#     img_key=img_keys[0],        # 'downscaled_fullres'
#     spot_size=180,
#     alpha_img=1,
#     vmin = 0.5,
#     vmax = 1.5,
#     cmap="viridis"
# )
# # %%

# # 降序排序并取前十个元素
# top_ten_elements = sorted(all_corr, reverse=True)[:20]
# # 获取这些元素在原列表中的位置
# positions = []
# for element in top_ten_elements:
#     # list.index() 找到第一个匹配项的位置
#     positions.append(all_corr.index(element))

# print(positions)
# # %%
# selected_genes.index("TPM2")
# # %%
