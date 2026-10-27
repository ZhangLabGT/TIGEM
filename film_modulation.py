import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FiLMHead(nn.Module):
    """
    Gene-Conditioned FiLM Decoder Head
    ----------------------------------
    将基因的 ESM2 向量 e_g 映射为通道级缩放/平移 (gamma_g, beta_g)，
    再用共享读出向量 u 对每个基因的调制后节点表示做回归：

        \tilde h_{i,g} = gamma_g ⊙ h_i + beta_g
        \hat y_{i,g}  = u^T \tilde h_{i,g}

    向量化实现：
        H_u    = H ⊙ u           # [N, d]
        Gamma  = Theta_phi(E)[:,:d]      # [G, d]
        Beta   = Theta_phi(E)[:, d:2d]   # [G, d]
        Y      = H_u @ Gamma.T + 1 @ (Beta @ u).T   # [N, G]

    备注：若你的监督是 log1p(count)，可以直接回归线性输出；
         若回归原始非负表达，建议在 forward 末尾包一层 Softplus。
    """
    def __init__(
        self,
        d_model: int,                 # 节点表示维度 d（来自 GAT 编码器）
        d_gene: int,                  # 基因嵌入维度 d_g（ESM2 向量维度）
        hidden_gene: Optional[int] = None,  # 生成 (gamma,beta) 的中间维度，默认 2*d_model
        nonneg_output: bool = False,  # 是否对输出做 Softplus 保非负
        gamma_init: float = 1.0,      # gamma 的初始缩放中心（常取 1.0）
        beta_init: float = 0.0,       # beta 的初始平移中心（常取 0.0）
        use_tanh_gate: bool = True,   # 是否用 tanh 将偏移限制在稳定范围
        beta_scale: float = 1.0,      # 控制 beta 的幅度（防止过大偏移）
    ) -> None:
        super().__init__()

        # 保存核心超参：模型通道数与基因嵌入维度
        self.d_model = d_model               # 用于对齐 H 与 FiLM 参数的通道数
        self.d_gene  = d_gene                # 输入基因嵌入的维度
        self.nonneg_output = nonneg_output   # 输出是否保证非负
        self.use_tanh_gate = use_tanh_gate   # 是否在 gamma/beta 前加 tanh 门
        self.beta_scale = beta_scale         # beta 的幅度控制

        # 共享读出向量 u ∈ R^d（回归头），初始化为均匀小数
        self.u = nn.Parameter(torch.empty(d_model))
        nn.init.uniform_(self.u, a=-0.02, b=0.02)

        # 将基因嵌入从 d_g 投到 d，并做层归一化以稳训练
        self.gene_ln = nn.LayerNorm(d_gene, elementwise_affine=True)  # 对 E 做归一化
        self.W_e = nn.Linear(d_gene, d_model, bias=False)             # W_e：对齐到 d
        nn.init.xavier_uniform_(self.W_e.weight)

        # 生成 (gamma, beta) 的小型 MLP：Theta: R^d -> R^{2d}
        hg = hidden_gene if hidden_gene is not None else 2 * d_model  # MLP 隐藏维度
        self.theta = nn.Sequential(
            nn.Linear(d_model, hg, bias=True),
            nn.GELU(),
            nn.Linear(hg, 2 * d_model, bias=True),
        )
        # 初始化最后一层，使得初始 gamma≈gamma_init，beta≈beta_init
        nn.init.zeros_(self.theta[-1].weight)  # 让初值更接近常数门
        nn.init.zeros_(self.theta[-1].bias)

        # gamma/beta 中心值作为缓冲（不训练），便于形成 1+Δ 和 0+Δ 的稳定偏置
        self.register_buffer("gamma_center", torch.tensor(float(gamma_init)))
        self.register_buffer("beta_center",  torch.tensor(float(beta_init)))

        # 缓存对齐后的基因嵌入 Phi(E) 以避免每 batch 重复计算
        self._phi_cache: Optional[torch.Tensor] = None  # [G, d]

    @torch.no_grad()
    def set_gene_embeddings(self, E: torch.Tensor) -> None:
        """
        预先设置/缓存对齐后的基因嵌入：Phi(E) = LN(E) @ W_e
        E: [G, d_g]
        """
        phi = self.W_e(self.gene_ln(E))            # [G, d]
        self._phi_cache = phi.detach().clone()     # 缓存到模块内部

    def clear_gene_cache(self) -> None:
        """清空缓存（若基因集或投影参数被更改后需要重算）。"""
        self._phi_cache = None

    def film_params(self, Phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从对齐后的基因嵌入 Phi 生成 (Gamma, Beta)
        Phi: [G, d]
        返回：Gamma, Beta 形状均为 [G, d]
        """
        G = Phi.size(0)
        gb = self.theta(Phi)                    # [G, 2d]
        gamma_raw, beta_raw = gb[:, :self.d_model], gb[:, self.d_model:]

        if self.use_tanh_gate:
            # 将偏移限制在 [-1,1]，再加上中心值，避免数值爆炸
            gamma = self.gamma_center + torch.tanh(gamma_raw)
            beta  = self.beta_center  + self.beta_scale * torch.tanh(beta_raw)
        else:
            # 不做范围限制：更自由但可能不稳
            gamma = self.gamma_center + gamma_raw
            beta  = self.beta_center  + beta_raw * self.beta_scale
        return gamma, beta

    def forward(
        self,
        H: torch.Tensor,                   # [N, d]  GAT 编码器输出
        E: Optional[torch.Tensor] = None,  # [G, d_g] 原始基因嵌入（若已缓存可为 None）
        return_film: bool = False,         # 是否返回 (Gamma, Beta) 便于可解释
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        返回：
            Y: [N, G]  预测值（线性或 Softplus 后）
            (Gamma, Beta): 可选返回，均为 [G, d]
        """
        device = H.device
        dtype  = H.dtype
        N, d = H.shape
        assert d == self.d_model, f"H dim {d} != d_model {self.d_model}"

        # 1) 取得对齐后的基因嵌入 Phi(E)
        if self._phi_cache is None:
            assert E is not None, "E is required on first forward or call set_gene_embeddings()"
            Phi = self.W_e(self.gene_ln(E.to(device=device, dtype=dtype)))  # [G, d]
        else:
            Phi = self._phi_cache.to(device=device, dtype=dtype)
        G = Phi.size(0)

        # 2) 生成 FiLM 参数 (Gamma, Beta)
        Gamma, Beta = self.film_params(Phi)            # [G, d], [G, d]
        Gamma = Gamma.to(device=device, dtype=dtype)
        Beta  = Beta.to(device=device, dtype=dtype)

        # 3) 共享读出向量 u，对应 H 的通道做加权
        u = self.u.to(device=device, dtype=dtype)      # [d]
        H_u = H * u                                    # [N, d]

        # 4) 矩阵化回归：Y = (H⊙u) @ Gamma^T + 1 @ (Beta @ u)^T
        Y_main = H_u @ Gamma.t()                       # [N, G]
        gene_bias = (Beta @ u).unsqueeze(0)            # [1, G]
        Y = Y_main + gene_bias                         # [N, G]

        # 5) 非负输出（可选）
        if self.nonneg_output:
            Y = F.softplus(Y)

        if return_film:
            return Y, (Gamma, Beta)
        return Y, None

    # ========== 额外辅助：可解释性与正则 ==========
    def l1_sparsity(self, weight: float = 1e-4, detach_gene: bool = False) -> torch.Tensor:
        """
        对 FiLM 的缩放门 Gamma 施加 L1 稀疏正则，促进通道选择性（可选）。
        若 detach_gene=True，则先用缓存的 Phi 生成一次 (Gamma,Beta) 并对 Gamma 取 .detach()，
        等价于对当前 batch 不回传梯度，仅作为额外约束项。
        """
        if self._phi_cache is None:
            return torch.tensor(0.0, device=self.u.device)
        with torch.no_grad() if detach_gene else torch.enable_grad():
            Gamma, _ = self.film_params(self._phi_cache.to(device=self.u.device, dtype=self.u.dtype))
        return weight * Gamma.abs().mean()
    



class FiLMHead_2x(nn.Module):
    """
    FiLM → MLP → FiLM → Readout
    - 第一次 FiLM 作用在 Z = GELU(H @ W_pre) 上（对齐到 d_mid）
    - 共享 MLP: Z_g -> U_g = GELU(Z_g @ W_post)
    - 第二次 FiLM 作用在 U_g（对齐到 d_hid），再用共享 readout 得到 y
    - 支持 set_gene_embeddings(E) 预缓存 Φ(E) 以加速
    - 支持 use_chunk / gene_chunk 控制显存
    """
    def __init__(self,
                 d_model: int,        # H 的通道数
                 d_gene: int,         # ESM2 维度
                 d_mid: Optional[int] = None,   # 第一次对齐后的维度
                 d_hid: Optional[int] = None,   # MLP 隐藏/第二次对齐维度
                 nonneg_output: bool = False,
                 use_tanh_gate: bool = True,
                 beta_scale: float = 1.0,
                 gene_chunk: int = 64):
        super().__init__()
        d_mid = d_mid or d_model
        d_hid = d_hid or d_mid

        # H → Z
        self.pre = nn.Sequential(nn.Linear(d_model, d_mid, bias=True), nn.GELU())
        # Z_g → U_g
        self.post = nn.Sequential(nn.Linear(d_mid, d_hid, bias=True), nn.GELU())
        # U_g → y_g（共享读出）
        self.readout = nn.Linear(d_hid, 1, bias=False)

        # Gene 对齐：Φ(E) = LN(E) @ W_e
        self.gene_ln = nn.LayerNorm(d_gene, elementwise_affine=True)
        self.W_e     = nn.Linear(d_gene, d_mid, bias=False)   # 先投到 d_mid（便于第1次 FiLM）

        # 第一次 FiLM 产生 (γ1, β1) ∈ R^{d_mid}
        self.theta1  = nn.Sequential(nn.Linear(d_mid, 2*d_mid), nn.GELU(),
                                     nn.Linear(2*d_mid, 2*d_mid))
        # 第二次 FiLM 产生 (γ2, β2) ∈ R^{d_hid}
        self.W_e2    = nn.Linear(d_mid, d_hid, bias=False)     # 复用 Φ 到 d_hid
        self.theta2  = nn.Sequential(nn.Linear(d_hid, 2*d_hid), nn.GELU(),
                                     nn.Linear(2*d_hid, 2*d_hid))

        # 初始化：让 γ≈1, β≈0，避免训练初期扰动过大
        for m in (self.theta1, self.theta2):
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        nn.init.uniform_(self.readout.weight, a=-0.02, b=0.02)

        self.nonneg_output = nonneg_output
        self.use_tanh_gate = use_tanh_gate
        self.beta_scale    = beta_scale
        self.gene_chunk    = gene_chunk

        # γ/β 的中心（常数，不训练）
        self.register_buffer("gamma1_c", torch.tensor(1.0))
        self.register_buffer("beta1_c",  torch.tensor(0.0))
        self.register_buffer("gamma2_c", torch.tensor(1.0))
        self.register_buffer("beta2_c",  torch.tensor(0.0))

        # 缓存 Φ(E)
        self.register_buffer("_phi_cache", None)   # [G, d_mid]

    # ---------- 工具 ----------
    @torch.no_grad()
    def set_gene_embeddings(self, E: torch.Tensor):
        """一次性缓存 Φ(E) = LN(E) @ W_e。E:[G, d_gene]"""
        phi = self.W_e(self.gene_ln(E))
        self._phi_cache = phi.detach().clone()     # [G, d_mid]

    def _film_params(self, X: torch.Tensor, theta: nn.Module,
                     gamma_c: torch.Tensor, beta_c: torch.Tensor,
                     map_beta: float = 1.0):
        """X:[G, d] -> (γ, β) 形状均为 [G, d]"""
        gb = theta(X)
        d  = X.size(1)
        g_raw, b_raw = gb[:, :d], gb[:, d:]
        if self.use_tanh_gate:
            gamma = gamma_c + torch.tanh(g_raw)
            beta  = beta_c  + self.beta_scale * torch.tanh(b_raw) * map_beta
        else:
            gamma = gamma_c + g_raw
            beta  = beta_c  + self.beta_scale * b_raw * map_beta
        return gamma, beta

    # ---------- 前向 ----------
    def forward(self,
                H: torch.Tensor,                   # [N, d_model]
                E: Optional[torch.Tensor] = None,  # [G, d_gene]（若已缓存可 None）
                return_film: bool = False,
                use_chunk: bool = False,
                gene_chunk: Optional[int] = None):
        device, dtype = H.device, H.dtype
        N = H.size(0)
        d_mid = self.pre[0].out_features
        d_hid = self.post[0].out_features
        gene_chunk = gene_chunk or self.gene_chunk

        # 0) 预处理
        Z = self.pre(H)  # [N, d_mid]

        if self._phi_cache is None:
            assert E is not None, "E required on first forward or call set_gene_embeddings()."
            Phi_mid = self.W_e(self.gene_ln(E.to(device=device, dtype=dtype)))  # [G, d_mid]
        else:
            Phi_mid = self._phi_cache.to(device=device, dtype=dtype)            # [G, d_mid]

        G = Phi_mid.size(0)
        Phi_hid = self.W_e2(Phi_mid)                                            # [G, d_hid]

        outs = []
        if return_film:
            gammas1, betas1, gammas2, betas2 = [], [], [], []

        if not use_chunk:
            # ------- 全量路径（显存足够时可用） -------
            gamma1, beta1 = self._film_params(Phi_mid, self.theta1, self.gamma1_c, self.beta1_c)
            Zg = Z.unsqueeze(1) * gamma1.unsqueeze(0) + beta1.unsqueeze(0)      # [N,G,d_mid]
            U  = self.post(Zg)                                                  # [N,G,d_hid]
            gamma2, beta2 = self._film_params(Phi_hid, self.theta2, self.gamma2_c, self.beta2_c)
            U2 = U * gamma2.unsqueeze(0) + beta2.unsqueeze(0)                   # [N,G,d_hid]
            y  = torch.einsum('ngh,h->ng', U2, self.readout.weight.squeeze(0))  # [N,G]
            Y  = F.softplus(y) if self.nonneg_output else y
            if return_film:
                return Y, (gamma1, beta1, gamma2, beta2)
            return Y, None

        # ------- 分块路径（稳） -------
        for s in range(0, G, gene_chunk):
            t = min(s + gene_chunk, G)
            # 第1次 FiLM
            g1, b1 = self._film_params(Phi_mid[s:t], self.theta1, self.gamma1_c, self.beta1_c)
            Zg = Z.unsqueeze(1) * g1.unsqueeze(0) + b1.unsqueeze(0)             # [N,g,d_mid]
            # 共享 MLP
            U  = self.post(Zg)                                                  # [N,g,d_hid]
            # 第2次 FiLM
            g2, b2 = self._film_params(Phi_hid[s:t], self.theta2, self.gamma2_c, self.beta2_c)
            U2 = U * g2.unsqueeze(0) + b2.unsqueeze(0)                          # [N,g,d_hid]
            # 读出
            yg = torch.einsum('ngh,h->ng', U2, self.readout.weight.squeeze(0))  # [N,g]
            outs.append(yg)

            if return_film:
                gammas1.append(g1); betas1.append(b1)
                gammas2.append(g2); betas2.append(b2)

        Y = torch.cat(outs, dim=1)                                              # [N,G]
        if self.nonneg_output:
            Y = F.softplus(Y)

        if return_film:
            return Y, (torch.cat(gammas1,0), torch.cat(betas1,0),
                       torch.cat(gammas2,0), torch.cat(betas2,0))
        return Y, None


class FiLMHead_2x_CA(nn.Module):
    """
    Gene→Node Cross-Attention + FiLM→MLP→FiLM→Readout（不分块）
    - 先做一次基因查询对节点的 cross-attn 得到每基因上下文 C_g
    - 用 [Phi_g || C_g] 生成两次 FiLM 的 (gamma,beta)，
      再按照方案1堆叠：FiLM → MLP → FiLM → 共享读出
    """
    def __init__(self,
                 d_model: int,          # H 的通道数
                 d_gene: int,           # ESM2 维度
                 d_mid: Optional[int] = None,   # FiLM1/MLP 输入宽度
                 d_hid: Optional[int] = None,   # MLP 输出/FiLM2 宽度
                 d_ca:  Optional[int] = None,   # cross-attn 低维
                 nonneg_output: bool = False,
                 attn_drop: float = 0.0):
        super().__init__()
        d_mid = d_mid or d_model
        d_hid = d_hid or d_mid
        d_ca  = d_ca  or d_mid

        # H → Z（共享特征）
        self.pre  = nn.Sequential(nn.Linear(d_model, d_mid, bias=True), nn.GELU())
        self.post = nn.Sequential(nn.Linear(d_mid,  d_hid, bias=True), nn.GELU())
        self.readout = nn.Linear(d_hid, 1, bias=False)

        # Cross-Attn 投影
        self.gene_ln = nn.LayerNorm(d_gene, elementwise_affine=True)
        self.Wq = nn.Linear(d_gene,  d_ca, bias=False)
        self.Wk = nn.Linear(d_model, d_ca, bias=False)
        self.Wv = nn.Linear(d_model, d_ca, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

        # 基因对齐到 FiLM 维度
        self.W_e1 = nn.Linear(d_gene, d_mid, bias=False)
        self.W_e2 = nn.Linear(d_gene, d_hid, bias=False)
        self.ca2mid = nn.Linear(d_ca, d_mid, bias=False)
        self.ca2hid = nn.Linear(d_ca, d_hid, bias=False)

        # 两次 FiLM 的生成器：输入 concat([Phi_g, C_g])
        self.theta1 = nn.Sequential(
            nn.Linear(d_mid + d_mid, 2 * d_mid), nn.GELU(),
            nn.Linear(2 * d_mid,     2 * d_mid)
        )
        self.theta2 = nn.Sequential(
            nn.Linear(d_hid + d_hid, 2 * d_hid), nn.GELU(),
            nn.Linear(2 * d_hid,     2 * d_hid)
        )

        # 让初值 γ≈1, β≈0
        nn.init.zeros_(self.theta1[-1].weight); nn.init.zeros_(self.theta1[-1].bias)
        nn.init.zeros_(self.theta2[-1].weight); nn.init.zeros_(self.theta2[-1].bias)
        nn.init.uniform_(self.readout.weight, a=-0.02, b=0.02)

        self.nonneg_output = nonneg_output

    @staticmethod
    def _film_params(inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """把最后一层线性输出拆成 (gamma, beta)，并用 tanh 做稳定门控。"""
        d2 = inp.size(-1)
        d  = d2 // 2
        g_raw, b_raw = inp[:, :d], inp[:, d:]
        gamma = 1.0 + torch.tanh(g_raw)
        beta  = 0.0 + torch.tanh(b_raw)
        return gamma, beta

    def forward(self, H: torch.Tensor, E: torch.Tensor,
                return_film: bool = False):
        """
        H: [N, d_model], E: [G, d_gene]
        返回 Y: [N, G]
        """
        N, d = H.shape
        G, dg = E.shape
        device, dtype = H.device, H.dtype

        # ===== 1) Cross-Attention：gene queries over node keys/values =====
        En = self.gene_ln(E.to(device=device, dtype=dtype))      # [G, d_g]
        Q  = self.Wq(En)                                         # [G, d_ca]
        K  = self.Wk(H)                                          # [N, d_ca]
        V  = self.Wv(H)                                          # [N, d_ca]
        attn = torch.softmax(Q @ K.t() / (Q.size(-1) ** 0.5), dim=-1)  # [G, N]
        attn = self.attn_drop(attn)
        C = attn @ V                                             # [G, d_ca]

        # 对齐到各自 FiLM 的宽度
        Phi1 = self.W_e1(En)                                     # [G, d_mid]
        C1   = self.ca2mid(C)                                    # [G, d_mid]
        Phi2 = self.W_e2(En)                                     # [G, d_hid]
        C2   = self.ca2hid(C)                                    # [G, d_hid]

        # ===== 2) 生成两次 FiLM 参数 =====
        gamma1, beta1 = self._film_params(self.theta1(torch.cat([Phi1, C1], dim=-1)))  # [G,d_mid]
        gamma2, beta2 = self._film_params(self.theta2(torch.cat([Phi2, C2], dim=-1)))  # [G,d_hid]

        # ===== 3) FiLM→MLP→FiLM→Readout（一次性全量，无分块） =====
        Z  = self.pre(H)                                         # [N, d_mid]
        Zg = Z.unsqueeze(1) * gamma1.unsqueeze(0) + beta1.unsqueeze(0)    # [N,G,d_mid]
        U  = self.post(Zg)                                       # [N,G,d_hid]
        U2 = U * gamma2.unsqueeze(0) + beta2.unsqueeze(0)        # [N,G,d_hid]
        Y  = torch.einsum('ngh,h->ng', U2, self.readout.weight.squeeze(0))  # [N,G]

        if self.nonneg_output:
            Y = F.softplus(Y)

        if return_film:
            return Y, (gamma1, beta1, gamma2, beta2, attn)       # attn 便于可解释
        return Y, None