import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# ==================== Rotary Positional Embedding (giữ nguyên) ====================
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE head dimension must be even"
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

# ==================== State Feature Extractor (đã sửa) ====================
class StateFeatureExtractor(nn.Module):
    """
    Trích xuất đặc trưng từ pos_logits, neg_logits (đã được cắt top‑k)
    và các thông tin vị trí.
    """
    def __init__(self, top_k: int = 32, ks: List[int] = [5, 10, 20], normalize_position: bool = False):
        super().__init__()
        self.top_k = top_k
        self.ks = ks
        self.normalize_position = normalize_position
        # Các tham số chuẩn hóa vị trí (có thể set từ bên ngoài hoặc tính theo max)
        self.register_buffer('max_block_pos', torch.tensor(1.0))   # placeholder, sẽ gán giá trị thực tế
        self.register_buffer('max_abs_pos', torch.tensor(1.0))

    def set_position_normalizers(self, max_block_pos: float, max_abs_pos: float):
        """Gọi trước khi training để set giá trị chuẩn hóa."""
        self.max_block_pos.fill_(max_block_pos)
        self.max_abs_pos.fill_(max_abs_pos)

    def forward(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev: Optional[torch.Tensor] = None):
        """
        Args:
            pos_logits, neg_logits: (B, S, top_k) - logits của các token top‑k
            block_pos: (B, S) - vị trí trong block (0..B-1)
            abs_pos: (B, S) - vị trí tuyệt đối trong toàn bộ câu
            alpha_prev: (B, S, 3) hoặc None
        Returns:
            features: (B, S, feature_dim)
        """
        B, S, K = pos_logits.shape
        assert neg_logits.shape == (B, S, K)
        assert block_pos.shape == (B, S) and abs_pos.shape == (B, S)

        # Chuẩn hóa vị trí nếu cần
        if self.normalize_position:
            block_pos_norm = block_pos / self.max_block_pos
            abs_pos_norm = abs_pos / self.max_abs_pos
        else:
            block_pos_norm = block_pos
            abs_pos_norm = abs_pos

        # Tính các phân phối xác suất
        pos_probs = F.softmax(pos_logits, dim=-1)
        pos_log_probs = F.log_softmax(pos_logits, dim=-1)
        neg_probs = F.softmax(neg_logits, dim=-1)
        neg_log_probs = F.log_softmax(neg_logits, dim=-1)

        # Entropy
        pos_entropy = -(pos_probs * pos_log_probs).sum(dim=-1, keepdim=True)
        neg_entropy = -(neg_probs * neg_log_probs).sum(dim=-1, keepdim=True)

        # Top-1 margin (chênh lệch giữa top1 và top2)
        pos_top2 = torch.topk(pos_probs, 2, dim=-1).values
        pos_margin = (pos_top2[..., 0] - pos_top2[..., 1]).unsqueeze(-1)
        neg_top2 = torch.topk(neg_probs, 2, dim=-1).values
        neg_margin = (neg_top2[..., 0] - neg_top2[..., 1]).unsqueeze(-1)

        # Top‑k mass cho các k khác nhau
        pos_mass = []
        neg_mass = []
        for k in self.ks:
            pos_mass.append(pos_probs[..., :k].sum(dim=-1, keepdim=True))
            neg_mass.append(neg_probs[..., :k].sum(dim=-1, keepdim=True))
        pos_mass = torch.cat(pos_mass, dim=-1)   # (B, S, len(ks))
        neg_mass = torch.cat(neg_mass, dim=-1)

        # Cross features
        diff_logits = pos_logits - neg_logits   # (B, S, K)
        kl_div = (pos_probs * (pos_log_probs - neg_log_probs)).sum(dim=-1, keepdim=True)

        # Ghép tất cả đặc trưng
        features = [
            pos_logits, pos_log_probs, pos_probs,
            pos_entropy, pos_margin, pos_mass,
            neg_logits, neg_log_probs, neg_probs,
            neg_entropy, neg_margin, neg_mass,
            diff_logits, kl_div,
            block_pos_norm.unsqueeze(-1), abs_pos_norm.unsqueeze(-1)
        ]
        if alpha_prev is not None:
            features.append(alpha_prev)   # (B, S, 3)

        return torch.cat(features, dim=-1)

# ==================== State Encoder ====================
class StateEncoder(nn.Module):
    """Biến đổi feature vector thành hidden representation."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==================== Transformer với RoPE và causal mask ====================
class TransformerRoPEBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x, cos, sin):
        residual = x
        x = self.norm1(x)
        B, S, _ = x.shape
        # Project and reshape to (B, n_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Causal attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
        x = residual + self.o_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerContext(nn.Module):
    """Stack các Transformer block với RoPE, dùng để mô hình hóa phụ thuộc giữa các token trong block."""
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, max_seq_len: int = 15):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads, max_position_embeddings=max_seq_len)
        self.layers = nn.ModuleList([TransformerRoPEBlock(hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        B, S, _ = x.shape
        cos, sin = self.rotary_emb(S, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x

# ==================== Actor: ContextualBanditAlpha ====================
class ContextualBanditAlpha(nn.Module):
    """
    Mạng RL sinh alpha động cho contrastive decoding.
    Đầu ra: (B, S, num_alpha_buckets) với num_alpha_buckets = 3.
    """
    def __init__(self, top_k: int = 32, hidden_dim: int = 128, num_alpha_buckets: int = 3,
                 max_alpha: float = 2.0, ks: List[int] = [5, 10, 20]):
        super().__init__()
        self.top_k = top_k
        self.num_alpha_buckets = num_alpha_buckets
        self.max_alpha = max_alpha

        # Feature extractor
        self.feature_extractor = StateFeatureExtractor(top_k=top_k, ks=ks)

        # Tính input_dim một cách chính xác
        # Mỗi side (pos/neg): logits(K) + log_probs(K) + probs(K) + entropy(1) + margin(1) + mass(len(ks))
        per_side = 3 * top_k + 2 + len(ks)
        cross = top_k + 1          # diff_logits (K) + kl_div(1)
        pos_feat = 2               # block_pos, abs_pos (mỗi cái 1)
        alpha_prev_dim = 3         # alpha_prev (luôn có, nếu không có thì zero)
        input_dim = 2 * per_side + cross + pos_feat + alpha_prev_dim
        self.input_dim = input_dim

        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.context = TransformerContext(hidden_dim=hidden_dim, num_heads=4, num_layers=2, max_seq_len=15)
        self.alpha_head = nn.Linear(hidden_dim, num_alpha_buckets)

    def forward(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev: Optional[torch.Tensor] = None):
        """
        Args:
            pos_logits, neg_logits: (B, S, top_k) - logits của top‑k token
            block_pos, abs_pos: (B, S) - vị trí (có thể đã chuẩn hóa hoặc chưa, feature_extractor sẽ xử lý)
            alpha_prev: (B, S, 3) - alpha của bước trước (nếu None thì dùng zero tensor)
        Returns:
            alphas: (B, S, 3) - hệ số alpha cho từng bucket
        """
        B, S, _ = pos_logits.shape
        if alpha_prev is None:
            alpha_prev = torch.zeros(B, S, 3, device=pos_logits.device, dtype=pos_logits.dtype)

        state_feats = self.feature_extractor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
        x = self.encoder(state_feats)          # (B, S, hidden_dim)
        x = self.context(x)                    # (B, S, hidden_dim)
        logits = self.alpha_head(x)            # (B, S, num_buckets)
        alphas = self.max_alpha * torch.tanh(logits)   # [-max_alpha, max_alpha]
        return alphas

# ==================== Critic (giá trị value) ====================
class Critic(nn.Module):
    """
    Mạng ước tính value function V(s). Nhận state giống actor, trả về giá trị (B, S).
    """
    def __init__(self, top_k: int = 32, hidden_dim: int = 128, ks: List[int] = [5, 10, 20]):
        super().__init__()
        self.feature_extractor = StateFeatureExtractor(top_k=top_k, ks=ks)
        # Tính input_dim giống actor
        per_side = 3 * top_k + 2 + len(ks)
        cross = top_k + 1
        pos_feat = 2
        alpha_prev_dim = 3
        input_dim = 2 * per_side + cross + pos_feat + alpha_prev_dim
        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev: Optional[torch.Tensor] = None):
        if alpha_prev is None:
            alpha_prev = torch.zeros_like(pos_logits[..., :3])  # giả định (B,S,3)
        state_feats = self.feature_extractor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
        x = self.encoder(state_feats)
        values = self.value_head(x).squeeze(-1)   # (B, S)
        return values

# ==================== Hàm đếm tham số ====================
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}

# ==================== Ví dụ sử dụng (kiểm tra shape) ====================
if __name__ == "__main__":
    # Giả sử batch=2, seq_len=15 (block size -1), top_k=32
    B, S, K = 2, 15, 32
    pos_logits = torch.randn(B, S, K)
    neg_logits = torch.randn(B, S, K)
    block_pos = torch.randint(0, 16, (B, S)).float()
    abs_pos = torch.randint(0, 512, (B, S)).float()

    # Khởi tạo actor
    
    start_time = time.time()
    actor = ContextualBanditAlpha(top_k=K, hidden_dim=128, num_alpha_buckets=3, max_alpha=2.0)
    alphas = actor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev=None)
    print("Actor output shape:", alphas.shape)   # (2, 15, 3)
    print("Actor forward time:", time.time() - start_time)
    # Critic
    start_time = time.time()
    critic = Critic(top_k=K, hidden_dim=128)
    values = critic(pos_logits, neg_logits, block_pos, abs_pos)
    print("Critic output shape:", values.shape)   # (2, 15)
    print("Critic forward time:", time.time() - start_time)

    # Tham số
    print("Actor params:", count_parameters(actor))
    print("Critic params:", count_parameters(critic))