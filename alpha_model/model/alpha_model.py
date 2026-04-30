import torch
import torch.nn as nn
import torch.nn.functional as F

class StateFeatureExtractor(nn.Module):
    """
    Extracts statistical features from raw positive and negative logits:
    - log(softmax), softmax
    - entropy, top1 margin, top-k mass
    - Cross features: diff (pos - neg), KL divergence
    """
    def __init__(self, top_k: int = 32):
        super().__init__()
        self.top_k = top_k

    def forward(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev=None):
        # pos_logits, neg_logits: (batch, seq_len, top_k)
        # block_pos, abs_pos: (batch, seq_len)
        
        pos_probs = F.softmax(pos_logits, dim=-1)
        pos_log_probs = F.log_softmax(pos_logits, dim=-1)
        
        neg_probs = F.softmax(neg_logits, dim=-1)
        neg_log_probs = F.log_softmax(neg_logits, dim=-1)

        # Entropies
        pos_entropy = -(pos_probs * pos_log_probs).sum(dim=-1, keepdim=True)
        neg_entropy = -(neg_probs * neg_log_probs).sum(dim=-1, keepdim=True)

        # Top-1 Margin
        pos_top2 = torch.topk(pos_probs, 2, dim=-1).values
        pos_margin = (pos_top2[..., 0] - pos_top2[..., 1]).unsqueeze(-1)
        
        neg_top2 = torch.topk(neg_probs, 2, dim=-1).values
        neg_margin = (neg_top2[..., 0] - neg_top2[..., 1]).unsqueeze(-1)

        # Top-k mass (since logits are likely already truncated to top-k, this might be 1.0, 
        # but we compute it generally. If pos_probs is strictly top-k, sum is 1.0)
        pos_mass = pos_probs.sum(dim=-1, keepdim=True)
        neg_mass = neg_probs.sum(dim=-1, keepdim=True)

        # Cross features
        diff_logits = pos_logits - neg_logits
        # KL Divergence: sum(P * log(P/Q))
        kl_div = (pos_probs * (pos_log_probs - neg_log_probs)).sum(dim=-1, keepdim=True)

        # Positions
        b_pos = block_pos.unsqueeze(-1)
        a_pos = abs_pos.unsqueeze(-1)

        # Concatenate all features
        features = [
            pos_logits, pos_log_probs, pos_probs, pos_entropy, pos_margin, pos_mass,
            neg_logits, neg_log_probs, neg_probs, neg_entropy, neg_margin, neg_mass,
            diff_logits, kl_div,
            b_pos, a_pos
        ]
        
        if alpha_prev is not None:
             features.append(alpha_prev)
             
        return torch.cat(features, dim=-1)

class StateEncoder(nn.Module):
    """
    Encodes the combined state features into a dense representation.
    """
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

class TransformerContext(nn.Module):
    """
    Lightweight transformer replacing GRU for sequential block dependencies.
    """
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        # Position embedding for the 15 tokens
        self.pos_emb = nn.Parameter(torch.randn(1, 15, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Add position embedding
        x = x + self.pos_emb[:, :x.size(1), :]
        # Casual mask to prevent looking ahead in the diffusion block
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        return self.transformer(x, mask=mask, is_causal=True)

class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_features):
        x = self.encoder(state_features)
        x = self.mlp(x)
        value = self.value_head(x)
        return value.squeeze(-1)

class ContextualBanditAlpha(nn.Module):
    def __init__(self, top_k: int = 32, hidden_dim: int = 128, num_alpha_buckets: int = 3, max_alpha: float = 2.0, has_alpha_prev: bool = True):
        super().__init__()
        self.feature_extractor = StateFeatureExtractor(top_k=top_k)
        
        # Calculate input dimension
        # pos/neg logits, log_probs, probs = 6 * top_k
        # entropies, margins, masses = 6
        # diff_logits, kl_div = top_k + 1
        # block_pos, abs_pos = 2
        # alpha_prev = 3 (if used)
        input_dim = (7 * top_k) + 9
        if has_alpha_prev:
            input_dim += 3
            
        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.context = TransformerContext(hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, num_alpha_buckets)
        self.max_alpha = max_alpha

    def forward(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev=None):
        state_features = self.feature_extractor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
        x = self.encoder(state_features)
        x = self.context(x)
        logits = self.alpha_head(x)
        alphas = self.max_alpha * torch.sigmoid(logits)
        return alphas

if __name__ == "__main__":
    # Test feature extraction and shapes
    batch_size = 4
    seq_len = 15 # block size
    top_k = 32
    
    # Raw inputs
    pos_logits = torch.randn(batch_size, seq_len, top_k)
    neg_logits = torch.randn(batch_size, seq_len, top_k)
    block_pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    abs_pos = block_pos + 10 # dummy absolute position
    alpha_prev = torch.zeros(batch_size, seq_len, 3)
    
    actor = ContextualBanditAlpha(top_k=top_k, has_alpha_prev=True)
    
    # We still need to manually extract features for the Critic in this test
    # since we only embedded the extractor into the actor
    extractor = StateFeatureExtractor(top_k=top_k)
    state_features = extractor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
    input_dim = state_features.size(-1)
    critic = Critic(input_dim=input_dim)
    
    alphas = actor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
    values = critic(state_features)
    
    print("Alphas shape:", alphas.shape) # Expected: (4, 15, 3)
    print("Values shape:", values.shape)   # Expected: (4, 15)