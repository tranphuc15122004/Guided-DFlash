"""
Negative Logit Predictor for Contrastive Decoding.

Instead of learning scalar alpha values to scale the negative distribution,
this model directly predicts the 32 negative logit values for the top-32
positive tokens, giving it 32 degrees of freedom to reshape the negative
distribution.

Two modes:
  - predict_delta=False (default): output directly replaces negative logits.
  - predict_delta=True: output is a delta added to the original negative logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .alpha_model import StateFeatureExtractor, StateEncoder, TransformerContext


class NegativeLogitPredictor(nn.Module):
    """
    RL model that predicts negative logit values for the top-32 positive tokens.

    Architecture is identical to ContextualBanditAlpha except:
      - Output head: Linear(128 → 32) instead of Linear(128 → num_buckets)
      - No tanh bounding (logits are unbounded reals)
      - Optional delta mode: output is added to original neg_logits

    Input:  pos_logits (B,S,32), neg_logits (B,S,32), block_pos (B,S), abs_pos (B,S)
    Output: predicted_neg_logits (B,S,32) — new negative logits for top-32 tokens
    """

    def __init__(
        self,
        top_k: int = 32,
        hidden_dim: int = 128,
        predict_delta: bool = False,
        ks: List[int] = [5, 10, 20],
    ):
        super().__init__()
        self.top_k = top_k
        self.predict_delta = predict_delta

        # Feature extractor — reuses the same logic as ContextualBanditAlpha
        self.feature_extractor = StateFeatureExtractor(
            top_k=top_k,
            ks=ks,
            alpha_prev_dim=None,
        )

        # Compute input_dim the same way as ContextualBanditAlpha
        per_side = 3 * top_k + 2 + len(ks)     # logits + log_probs + probs + entropy + margin + mass
        cross = top_k + 1                        # diff_logits (K) + kl_div(1)
        pos_feat = 2                             # block_pos, abs_pos
        input_dim = 2 * per_side + cross + pos_feat
        self.input_dim = input_dim

        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.context = TransformerContext(hidden_dim=hidden_dim, num_heads=4, num_layers=2, max_seq_len=15)

        # Output head: 32 logit values, one per top-k token
        # Using LayerNorm before the head for stability (logits are unbounded)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, top_k)

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos_logits, neg_logits: (B, S, top_k) — positive/negative logits for top-k tokens
            block_pos, abs_pos: (B, S) — normalized positions
        Returns:
            predicted_neg_logits: (B, S, top_k) — new negative logit values
        """
        # Extract features (reuses StateFeatureExtractor which handles alignment)
        state_feats = self.feature_extractor(
            pos_logits, neg_logits, block_pos, abs_pos,
            alpha_prev=None,
        )

        x = self.encoder(state_feats)          # (B, S, hidden_dim)
        x = self.context(x)                    # (B, S, hidden_dim)
        x = self.output_norm(x)                # (B, S, hidden_dim)
        predicted = self.output_head(x)        # (B, S, top_k)

        if self.predict_delta:
            # Delta mode: add prediction to original neg_logits
            predicted = neg_logits + predicted

        return predicted


class GaussianNegativePolicy(NegativeLogitPredictor):
    """
    Gaussian policy wrapper for NegativeLogitPredictor.

    Outputs mean and learnable log_std for A2C training.
    log_std is a (top_k,)-shaped parameter — one std per token position.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One learnable log_std per top-k dimension
        self.log_std = nn.Parameter(
            torch.full((self.top_k,), -0.5, dtype=torch.float32)
        )

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ):
        mean = super().forward(pos_logits, neg_logits, block_pos, abs_pos)
        # log_std shape: (top_k,) → expand to (B, S, top_k)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std


class NegativePredictorCritic(nn.Module):
    """
    Critic (value function) for the negative predictor.

    Shares the same feature extractor as the actor but uses a simpler
    MLP value head. Outputs a scalar value per position.
    """

    def __init__(
        self,
        top_k: int = 32,
        hidden_dim: int = 128,
        ks: List[int] = [5, 10, 20],
    ):
        super().__init__()
        self.feature_extractor = StateFeatureExtractor(
            top_k=top_k,
            ks=ks,
            alpha_prev_dim=None,
        )

        per_side = 3 * top_k + 2 + len(ks)
        cross = top_k + 1
        pos_feat = 2
        input_dim = 2 * per_side + cross + pos_feat

        self.encoder = StateEncoder(input_dim, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ) -> torch.Tensor:
        state_feats = self.feature_extractor(
            pos_logits, neg_logits, block_pos, abs_pos,
            alpha_prev=None,
        )
        x = self.encoder(state_feats)
        values = self.value_head(x).squeeze(-1)  # (B, S)
        return values


# ==================== MLP (Dense) Variant ====================

class NegativeLogitPredictor_Dense(nn.Module):
    """
    Dense MLP version of NegativeLogitPredictor.

    This variant removes the transformer stack and instead encodes the full
    block with fully connected layers. Each token is first projected into a
    hidden space, the block is padded to a fixed length, and the flattened
    block representation is processed by an MLP to produce all 32 negative
    logit values at once.

    Architecture is analogous to ContextualBanditAlpha_Dense.
    """

    def __init__(
        self,
        top_k: int = 32,
        hidden_dim: int = 512,
        predict_delta: bool = False,
        ks: List[int] = [5, 10, 20],
        max_seq_len: int = 15,
        block_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.predict_delta = predict_delta
        self.max_seq_len = max_seq_len

        # Feature extractor.
        self.feature_extractor = StateFeatureExtractor(
            top_k=top_k,
            ks=ks,
            alpha_prev_dim=None,
        )

        # Compute input_dim the same way as ContextualBanditAlpha_Dense.
        per_side = 3 * top_k + 2 + len(ks)
        cross = top_k + 1
        pos_feat = 2
        input_dim = 2 * per_side + cross + pos_feat
        self.input_dim = input_dim

        self.token_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        block_hidden_dim = block_hidden_dim or hidden_dim * 2
        block_input_dim = max_seq_len * hidden_dim
        self.block_mlp = nn.Sequential(
            nn.Linear(block_input_dim, block_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_hidden_dim, block_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_hidden_dim, max_seq_len * top_k),
        )

    def _pad_to_max_seq_len(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"NegativeLogitPredictor_Dense expects seq_len <= {self.max_seq_len}, got {seq_len}."
            )
        if seq_len == self.max_seq_len:
            return x
        pad = torch.zeros(
            batch_size,
            self.max_seq_len - seq_len,
            hidden_dim,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=1)

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos_logits, neg_logits: (B, S, top_k)
            block_pos, abs_pos: (B, S)
        Returns:
            predicted_neg_logits: (B, S, top_k)
        """
        batch_size, seq_len, _ = pos_logits.shape
        state_feats = self.feature_extractor(
            pos_logits, neg_logits, block_pos, abs_pos,
            alpha_prev=None,
        )
        token_hidden = self.token_projector(state_feats)  # (B, S, hidden_dim)
        token_hidden = self._pad_to_max_seq_len(token_hidden)
        block_hidden = self.block_mlp(token_hidden.reshape(batch_size, -1))
        predicted = block_hidden.view(batch_size, self.max_seq_len, self.top_k)
        predicted = predicted[:, :seq_len, :]  # crop to actual seq_len

        if self.predict_delta:
            predicted = neg_logits + predicted

        return predicted


class GaussianNegativePolicy_Dense(NegativeLogitPredictor_Dense):
    """
    Gaussian policy wrapper for NegativeLogitPredictor_Dense.

    Outputs mean and learnable log_std for A2C training.
    log_std is a (top_k,)-shaped parameter — one std per token position.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std = nn.Parameter(
            torch.full((self.top_k,), -0.5, dtype=torch.float32)
        )

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ):
        mean = super().forward(pos_logits, neg_logits, block_pos, abs_pos)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std


class NegativePredictorCritic_Dense(nn.Module):
    """
    Critic (value function) for the dense negative predictor.

    Uses the same dense block-level architecture as the actor,
    but with a value head instead of the logit output.
    """

    def __init__(
        self,
        top_k: int = 32,
        hidden_dim: int = 128,
        ks: List[int] = [5, 10, 20],
        max_seq_len: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_extractor = StateFeatureExtractor(
            top_k=top_k,
            ks=ks,
            alpha_prev_dim=None,
        )

        per_side = 3 * top_k + 2 + len(ks)
        cross = top_k + 1
        pos_feat = 2
        input_dim = 2 * per_side + cross + pos_feat

        self.token_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.max_seq_len = max_seq_len
        block_hidden_dim = hidden_dim * 2
        block_input_dim = max_seq_len * hidden_dim
        self.block_mlp = nn.Sequential(
            nn.Linear(block_input_dim, block_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_hidden_dim, block_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(block_hidden_dim, max_seq_len),
        )

    def forward(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        block_pos: torch.Tensor,
        abs_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = pos_logits.shape
        state_feats = self.feature_extractor(
            pos_logits, neg_logits, block_pos, abs_pos,
            alpha_prev=None,
        )
        token_hidden = self.token_projector(state_feats)
        # Pad to fixed length
        if seq_len < self.max_seq_len:
            pad = torch.zeros(
                batch_size, self.max_seq_len - seq_len, token_hidden.size(-1),
                device=token_hidden.device, dtype=token_hidden.dtype,
            )
            token_hidden = torch.cat([token_hidden, pad], dim=1)
        block_hidden = self.block_mlp(token_hidden.reshape(batch_size, -1))
        values = block_hidden.view(batch_size, self.max_seq_len)[:, :seq_len]
        return values


def build_full_neg_from_top32(
    predicted_neg_logits: torch.Tensor,   # (B, S, 32)
    top32_token_ids: torch.Tensor,         # (B, S, 32)
    vocab_size: int,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """
    Build a full-vocabulary negative logit tensor from 32 predicted values.

    Only the top-32 positive tokens get the predicted negative logits.
    All other tokens get `fill_value` (≈ zero probability after softmax).
    """
    B, S, K = predicted_neg_logits.shape
    assert K <= top32_token_ids.size(-1), (
        f"predicted_neg_logits shape ({predicted_neg_logits.shape}) "
        f"and top32_token_ids shape ({top32_token_ids.shape}) mismatch"
    )
    device = predicted_neg_logits.device
    dtype = predicted_neg_logits.dtype

    neg_full = torch.full(
        (B, S, vocab_size),
        fill_value,
        device=device,
        dtype=dtype,
    )
    neg_full.scatter_(-1, top32_token_ids, predicted_neg_logits)
    return neg_full


def apply_cd_with_predicted_neg(
    positive_logits: torch.Tensor,      # (B, S, V)
    negative_logits_predicted: torch.Tensor,  # (B, S, V) — full vocab
) -> torch.Tensor:
    """
    Apply Contrastive Decoding with predicted negative logits and fixed alpha=1.

    CD = log_softmax(pos) - log_softmax(neg_predicted)
    """
    log_p1 = F.log_softmax(positive_logits, dim=-1)
    log_p2 = F.log_softmax(negative_logits_predicted, dim=-1)
    return log_p1 - log_p2  # alpha = 1.0 fixed


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'non_trainable': total - trainable}


if __name__ == "__main__":
    # Quick shape test
    B, S, K = 2, 15, 32
    pos_logits = torch.randn(B, S, K)
    neg_logits = torch.randn(B, S, K)
    block_pos = torch.randint(0, 16, (B, S)).float()
    abs_pos = torch.randint(0, 512, (B, S)).float()

    # Test NegativeLogitPredictor (replacement mode)
    model = NegativeLogitPredictor(top_k=K, hidden_dim=128, predict_delta=False)
    out = model(pos_logits, neg_logits, block_pos, abs_pos)
    print(f"NegativeLogitPredictor (replace): {out.shape}  (expected (2,15,32))")
    print(f"  Params: {count_parameters(model)}")

    # Test delta mode
    model_delta = NegativeLogitPredictor(top_k=K, hidden_dim=128, predict_delta=True)
    out_delta = model_delta(pos_logits, neg_logits, block_pos, abs_pos)
    print(f"NegativeLogitPredictor (delta):  {out_delta.shape}  (expected (2,15,32))")

    # Test GaussianNegativePolicy
    policy = GaussianNegativePolicy(top_k=K, hidden_dim=128)
    mean, log_std = policy(pos_logits, neg_logits, block_pos, abs_pos)
    print(f"GaussianNegativePolicy mean:     {mean.shape}  (expected (2,15,32))")
    print(f"GaussianNegativePolicy log_std:  {log_std.shape}  (expected (2,15,32))")
    print(f"  log_std param: {policy.log_std.shape}")

    # Test Critic
    critic = NegativePredictorCritic(top_k=K, hidden_dim=128)
    values = critic(pos_logits, neg_logits, block_pos, abs_pos)
    print(f"NegativePredictorCritic values:  {values.shape}  (expected (2,15))")
    print(f"  Params: {count_parameters(critic)}")

    # Test build_full_neg_from_top32
    top32_ids = torch.randint(0, 128000, (B, S, K))
    full_neg = build_full_neg_from_top32(out, top32_ids, vocab_size=128000)
    print(f"build_full_neg_from_top32:       {full_neg.shape}  (expected (2,15,128000))")

    # Test apply_cd_with_predicted_neg with dummy inputs
    pos_full = torch.randn(B, S, 128000)
    cd_out = apply_cd_with_predicted_neg(pos_full, full_neg)
    print(f"apply_cd_with_predicted_neg:     {cd_out.shape}  (expected (2,15,128000))")

    # Sanity check: when predicted_neg = pos, CD output should be all zeros
    cd_self = apply_cd_with_predicted_neg(pos_full, pos_full)
    max_deviation = cd_self.abs().max().item()
    print(f"Self-CD max deviation:           {max_deviation:.6f}  (expected ~0.0)")
    assert max_deviation < 1e-5, f"Self-CD failed: max deviation = {max_deviation}"
    print("\nAll shape checks passed ✓")
