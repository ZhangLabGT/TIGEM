import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GATConvGated(MessagePassing):
    """
    α_ij^{(h)} = softmax_j( e_ij^{(h)} + λ_h * log_g_ij )，其中 log_g_ij ∈ [log_eps, 0].
    - gate_scale (λ) 可固定或可学习（per-head）
    - gating_heads: 只对这些 head 应用门控
    """
    def __init__(self, in_channels, out_channels, heads=4, concat=True,
                 negative_slope=0.2, dropout=0.0, add_self_loops=True,
                 gating_heads=None,
                 gate_scale=2.0,          # <--- 新增：默认 λ=1.0；你也可以先设 0.0 再退火
                 learnable_gate=True     # <--- 新增：是否让每个 head 的 λ 可学习
                 ):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # 节点线性投影 (= W)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

        # 注意力向量 (= a) [H, 2*C]
        self.att = nn.Parameter(torch.empty(heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)

        # 哪些头加门控
        mask = torch.zeros(heads, dtype=torch.bool)
        if gating_heads is None:
            mask[:] = True
        else:
            for h in gating_heads:
                mask[h] = True
        self.register_buffer('gating_mask', mask)

        # log_g 的下限（建议与你生成 log_g 的 eps 保持一致）
        self.log_eps = float(torch.log(torch.tensor(1e-6)))

        # ---- 新增：门控缩放 λ（per-head）
        if learnable_gate:
            self.gate_w = nn.Parameter(torch.full((heads,), float(gate_scale)))
        else:
            self.register_buffer('gate_w', torch.full((heads,), float(gate_scale)))

    def set_gate_scale(self, val: float, heads=None):
        """把某些 head 的 λ 设成 val；heads=None 则所有 head。"""
        with torch.no_grad():
            if heads is None:
                self.gate_w.fill_(float(val))
            else:
                for h in heads:
                    self.gate_w[h] = float(val)

    def forward(self, x, edge_index, log_g=None, return_attention_weights=False):
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # 补自环 & 对应补 0 的 log_g
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            if log_g is not None:
                pad = edge_index.size(1) - log_g.numel()
                if pad > 0:
                    zeros = x.new_zeros(pad)
                    log_g = torch.cat([log_g, zeros], dim=0)

        # 节点投影：把 W 做在节点侧
        x = self.lin(x).view(-1, H, C)  # [N,H,C]

        out = self.propagate(edge_index, x=x, log_g=log_g, size=None)

        out = out.view(-1, H * C) if self.concat else out.view(-1, H, C).mean(dim=1)
        if return_attention_weights:
            return out, (edge_index, self._cached_alpha)
        return out

    def message(self, x_i, x_j, index, log_g):
        H, C = self.heads, self.out_channels

        # 原生 GAT 打分：e = LeakyReLU( a^T [z_i || z_j] )
        cat = torch.cat([x_i, x_j], dim=-1)                  # [E,H,2C]
        e = (cat * self.att.view(1, H, 2*C)).sum(dim=-1)     # [E,H]
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # log_g 展开到各 head，并按掩码与缩放系数 λ 作用
        if log_g is None:
            log_g_full = x_i.new_zeros(e.size(0), H)         # [E,H]
        else:
            log_g = torch.clamp(log_g, min=self.log_eps, max=0.0)   # [E]
            log_g_full = log_g.unsqueeze(1).expand(-1, H)           # [E,H]

        if self.gating_mask.any():
            mask = self.gating_mask.view(1, H).to(log_g_full.dtype) # [1,H]
            e = e + log_g_full * self.gate_w.view(1, H) * mask      # e += λ * log_g

        # 归一化得到注意力
        alpha = softmax(e, index=index)                               # [E,H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._cached_alpha = alpha.detach()
        return alpha.unsqueeze(-1) * x_j                              # [E,H,C]


class GATv2ConvGated(MessagePassing):
    """
    e_ij^{(h)} = a_h^T LeakyReLU( W_h [x_i || x_j] )
    α_ij^{(h)} = softmax_j( e_ij^{(h)} + λ_h * log_g_ij )
    - 支持按 head 的门控缩放 λ_h（可学习/可退火）
    - gating_heads：只对这些 head 应用门控
    """
    def __init__(self, in_channels, out_channels, heads=4, attn_dim=None,
                 concat=True, negative_slope=0.2, dropout=0.0, add_self_loops=True,
                 gating_heads=None, gate_scale=1.0, learnable_gate=False):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.heads         = heads
        self.concat        = concat
        self.negative_slope = negative_slope
        self.dropout       = dropout
        self.add_self_loops = add_self_loops

        H, C = heads, out_channels
        A = attn_dim if attn_dim is not None else C  # 注意力中间维度

        # 先把节点投影到每头 C 维（与原 GAT 类似，便于控制维度）
        self.node_lin = nn.Linear(in_channels, H * C, bias=False)
        nn.init.xavier_uniform_(self.node_lin.weight)

        # GATv2 的“拼接后线性”： per-head 的 W_h ∈ R^{2C×A} 和 a_h ∈ R^{A}
        self.Wv = nn.Parameter(torch.empty(H, 2 * C, A))   # [H, 2C, A]
        self.a  = nn.Parameter(torch.empty(H, A))          # [H, A]
        nn.init.xavier_uniform_(self.Wv)
        nn.init.xavier_uniform_(self.a.unsqueeze(1))

        # 哪些 head 需要门控
        mask = torch.zeros(H, dtype=torch.bool)
        if gating_heads is None:
            mask[:] = True
        else:
            for h in gating_heads:
                mask[h] = True
        self.register_buffer('gating_mask', mask)

        # log_g 的下限（与你生成 log_g 的 eps 对齐，例如 1e-6 或 1e-3）
        self.log_eps = float(torch.log(torch.tensor(1e-6)))

        # 门控缩放 λ_h：可学习 or 固定
        if learnable_gate:
            self.gate_w = nn.Parameter(torch.full((H,), float(gate_scale)))
        else:
            self.register_buffer('gate_w', torch.full((H,), float(gate_scale)))

    def set_gate_scale(self, val: float, heads=None):
        """训练中方便退火 λ；heads=None 表示所有 head。"""
        with torch.no_grad():
            if heads is None:
                self.gate_w.fill_(float(val))
            else:
                for h in heads:
                    self.gate_w[h] = float(val)

    def forward(self, x, edge_index, log_g=None, return_attention_weights=False):
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # 自环与 log_g 对齐
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
            if log_g is not None:
                pad = edge_index.size(1) - log_g.numel()
                if pad > 0:
                    zeros = x.new_zeros(pad)
                    log_g = torch.cat([log_g, zeros], dim=0)

        # 节点投影到每头 C 维：x^h
        x = self.node_lin(x).view(-1, H, C)  # [N, H, C]

        out = self.propagate(edge_index, x=x, log_g=log_g, size=None)
        out = out.view(-1, H * C) if self.concat else out.view(-1, H, C).mean(dim=1)

        if return_attention_weights:
            return out, (edge_index, self._cached_alpha)  # α: [E, H]
        return out

    def message(self, x_i, x_j, index, log_g):
        """
        x_i, x_j: [E, H, C]
        """
        H, C = self.heads, self.out_channels

        # 拼接 [x_i || x_j]，再做 per-head 线性 W_h（GATv2）
        cat = torch.cat([x_i, x_j], dim=-1)                     # [E, H, 2C]
        # h = LeakyReLU( cat @ Wv )  → [E, H, A]
        h = torch.einsum('ehd,hda->eha', cat, self.Wv)          # Wv: [H, 2C, A]
        h = F.leaky_relu(h, negative_slope=self.negative_slope)
        # e = a_h^T h  → [E, H]
        e = torch.einsum('eha,ha->eh', h, self.a)

        # 门控：e += λ_h * log_g（按 head；仅对 gating_heads 生效）
        if log_g is None:
            log_g_full = x_i.new_zeros(e.size(0), H)            # [E, H]
        else:
            log_g = torch.clamp(log_g, min=self.log_eps, max=0.0)  # [E]
            log_g_full = log_g.unsqueeze(1).expand(-1, H)          # [E, H]

        if self.gating_mask.any():
            mask = self.gating_mask.view(1, H).to(log_g_full.dtype)
            e = e + log_g_full * self.gate_w.view(1, H) * mask

        # 归一化注意力
        alpha = softmax(e, index=index)                         # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._cached_alpha = alpha.detach()

        return alpha.unsqueeze(-1) * x_j                        # [E, H, C]


class GATv2ConvPlain(MessagePassing):
    """
    Plain GATv2 (no additive gating).

    Per-head logit (GATv2-style):
        e_ij^h = a_h^T LeakyReLU( Wv_h [x_i^h || x_j^h] )
        alpha_ij^h = softmax_j(e_ij^h)

    - Supports return_attention_weights=True and returns alpha WITH gradients,
      so you can backprop through topology-KL regularization.
    """
    def __init__(self,
                 in_channels: int,                  # input feature dim
                 out_channels: int,                 # per-head output dim
                 heads: int = 4,                    # number of attention heads
                 attn_dim: int = None,              # intermediate attention dim A
                 concat: bool = True,               # concat heads or average
                 negative_slope: float = 0.2,       # LeakyReLU slope
                 dropout: float = 0.0,              # dropout on attention weights
                 add_self_loops_flag: bool = True   # whether to add self-loops
                 ):
        super().__init__(aggr='add', node_dim=0)

        # ---- store hyperparameters ----
        self.in_channels = in_channels              # cache input dim
        self.out_channels = out_channels            # cache per-head output dim
        self.heads = heads                          # cache head count
        self.concat = concat                        # whether to concat heads
        self.negative_slope = negative_slope        # LeakyReLU slope
        self.dropout = dropout                      # attention dropout prob
        self.add_self_loops_flag = add_self_loops_flag  # self-loop toggle

        H = heads                                   # alias for readability
        C = out_channels                             # alias for readability
        A = attn_dim if attn_dim is not None else C  # attention hidden dim

        # ---- node projection: x -> [N, H*C] ----
        self.node_lin = nn.Linear(in_channels, H * C, bias=False)  # per-head projection
        nn.init.xavier_uniform_(self.node_lin.weight)              # stable init

        # ---- GATv2 parameters per head ----
        self.Wv = nn.Parameter(torch.empty(H, 2 * C, A))           # per-head linear on concatenated pair
        self.a = nn.Parameter(torch.empty(H, A))                   # per-head attention vector
        nn.init.xavier_uniform_(self.Wv)                           # init Wv
        nn.init.xavier_uniform_(self.a.unsqueeze(1))               # init a (treat as [H, A, 1])

        # ---- cached attention (for visualization) ----
        self._cached_alpha = None                                  # detached alpha cache (optional)
        self._alpha = None                                         # alpha with grad (for return)

    def forward(self, x, edge_index, return_attention_weights: bool = False):
        """
        x: [N, in_channels]
        edge_index: [2, E] (directed)
        return_attention_weights: whether to return (edge_index_used, alpha)
        """
        N = x.size(0)                                              # number of nodes
        H = self.heads                                             # head count
        C = self.out_channels                                      # per-head dim

        # ---- add self loops if needed ----
        if self.add_self_loops_flag:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)  # edge_index becomes [2, E+N]

        # ---- project node features into heads ----
        x_proj = self.node_lin(x).view(-1, H, C)                   # [N, H, C]

        # ---- propagate ----
        out = self.propagate(edge_index, x=x_proj, size=None)      # returns [N, H, C]

        # ---- combine heads ----
        if self.concat:
            out = out.view(-1, H * C)                              # [N, H*C]
        else:
            out = out.mean(dim=1)                                  # [N, C]

        if return_attention_weights:
            # return alpha WITH gradients for topology-KL
            return out, (edge_index, self._alpha)                  # alpha: [E_used, H]
        return out

    def message(self, x_i, x_j, index):
        """
        x_i, x_j: [E, H, C]
        index:    [E] destination indices (for group-softmax in PyG)
        """
        H = self.heads                                             # head count
        C = self.out_channels                                      # per-head dim

        # concat pair features: [E, H, 2C]
        cat = torch.cat([x_i, x_j], dim=-1)                        # pair features for attention

        # per-head linear: h = LeakyReLU(cat @ Wv) -> [E, H, A]
        h = torch.einsum('ehd,hda->eha', cat, self.Wv)             # apply Wv per head
        h = F.leaky_relu(h, negative_slope=self.negative_slope)    # nonlinearity

        # attention logits: e = a^T h -> [E, H]
        e = torch.einsum('eha,ha->eh', h, self.a)                  # dot with per-head a

        # attention weights: alpha = softmax over incoming edges per node
        alpha = softmax(e, index=index)                            # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # dropout

        # cache alpha
        self._alpha = alpha                                        # keep grad version for loss
        self._cached_alpha = alpha.detach()                        # detach version for logging

        return alpha.unsqueeze(-1) * x_j                           # [E, H, C]


class _FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.2):
        super().__init__()
        hid = dim * mult
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x