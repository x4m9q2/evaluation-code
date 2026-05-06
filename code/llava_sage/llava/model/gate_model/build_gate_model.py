import os

import torch
import torch.nn as nn


_DEBUG_NUMERICS_REPORTED = set()


def _maybe_report_nonfinite(tag, tensor):
    if os.environ.get("DEBUG_NUMERICS", "0") != "1":
        return
    if not isinstance(tensor, torch.Tensor):
        return
    tensor_f = tensor.detach().float()
    if torch.isfinite(tensor_f).all():
        return
    if tag in _DEBUG_NUMERICS_REPORTED:
        return
    _DEBUG_NUMERICS_REPORTED.add(tag)
    finite_vals = tensor_f[torch.isfinite(tensor_f)]
    finite_max_abs = float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else None
    print(
        f"[debug-numerics] rank={os.environ.get('RANK', '?')} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
        f"tag={tag} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"nan_count={int(torch.isnan(tensor_f).sum().item())} "
        f"inf_count={int(torch.isinf(tensor_f).sum().item())} "
        f"finite_max_abs={finite_max_abs}"
    )

class DualInputGate(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        # 第一层：输入是两个特征拼接后的维度
        # 如果莉珑酱传进来的特征维度不对，这里直接报错！
        self.fc1 = nn.Linear(visual_dim + text_dim,4 * visual_dim)
        self.gelu = nn.GELU()
        # 第二层：输出和视觉维度一致，准备做逐元素乘法
        self.fc2 = nn.Linear(4 * visual_dim, visual_dim)
        self.sigmoid = nn.Sigmoid()
        self.last_output_abs_mean = None
        self.last_patch_var = None
        self.current_gate_l1_loss = None
        self.last_gate_l1_loss = None
        self.current_gate_patch_activation = None

    def forward(self, visual_features, text_features):
            """
            visual_features: (B, L, D_v)  <-- 每一个 patch 都要被处理
            text_features: (B, D_t)       <-- 刚才拿到的 EOS 特征
            """
            target_dtype = self.fc1.weight.dtype
            visual_features = visual_features.to(dtype=target_dtype)
            text_features = text_features.to(dtype=target_dtype)
            _maybe_report_nonfinite("gate/visual_features_in", visual_features)
            _maybe_report_nonfinite("gate/text_features_in", text_features)

            # 获取 batch size 和 序列长度
            B, L, D_v = visual_features.shape

            # 1. 准备文本特征
            # 把 text_features 从 (B, D_t) 变成 (B, L, D_t)
            # 就像是把那句文本复制了 L 份，给每一个 patch 都发一份“参考资料”
            # 我们不做任何长度检查，B 对不上就让它在 expand 时报错！
            t_expanded = text_features.unsqueeze(1).expand(-1, L, -1)

            # 2. 暴力拼接视觉和文本特征
            # 在最后一维（特征维）拼接，形状变为 (B, L, D_v + D_t)
            combined = torch.cat([visual_features, t_expanded], dim=-1)
            _maybe_report_nonfinite("gate/combined", combined)

            # 3. 通过两层 MLP 算出每一个 patch 的门控权重
            # nn.Linear 作用在 (B, L, D_v + D_t) 时，会自动处理前两维
            gate = self.fc1(combined)
            _maybe_report_nonfinite("gate/fc1_out", gate)
            gate = self.gelu(gate)
            _maybe_report_nonfinite("gate/gelu_out", gate)
            gate = self.fc2(gate)
            _maybe_report_nonfinite("gate/fc2_out", gate)
            gate = self.sigmoid(gate)
            _maybe_report_nonfinite("gate/sigmoid_out", gate)
            self.current_gate_l1_loss = gate.abs().mean()
            self.current_gate_patch_activation = gate.mean(dim=-1)

            # 4. 逐元素特征缩放
            # 每个 patch 都有了自己专属的缩放系数！
            gated_output = visual_features * gate
            _maybe_report_nonfinite("gate/gated_output", gated_output)

            # 记录门控输出统计量，供训练日志上报。
            with torch.no_grad():
                out_f = gated_output.detach().float()
                self.last_output_abs_mean = float(out_f.abs().mean().item())
                # 先在 patch 维做方差，再对 batch/通道取均值，得到标量。
                self.last_patch_var = float(out_f.var(dim=1, unbiased=False).mean().item())
                self.last_gate_l1_loss = float(self.current_gate_l1_loss.detach().float().item())

            return gated_output
