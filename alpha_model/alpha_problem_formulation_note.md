# Alpha problem formulation note

## 1. Objective

Train an alpha policy that adapts contrastive strength per decode block in DFlash to improve acceptance length while preserving generation quality.

Main contrastive rule per token rank position j at diffusion position i:

z_cd[i, j] = z_pos[i, j] - alpha[i, b(j)] * z_neg_on_draft[i, j]

- z_pos: draft logits on draft top-k tokens
- z_neg_on_draft: negative logits gathered on the same draft token ids
- b(j): rank-bucket mapping (3 buckets)

## 2. Environment and timeline

- Block size B = 16
- Decision positions per block: B-1 = 15
- At each block t:
  1) draft/negative logits are produced
  2) alpha policy outputs alpha_t
  3) contrastive logits are applied
  4) target model verifies proposal and returns acceptance result
  5) environment transitions to next block state

Episode ends at EOS or max token limit.

## 3. MDP definition

M = (S, A, P, R, gamma)

### State S_t

State contains block-local evidence needed to choose alpha:

- draft_topk_token_ids, draft_topk_logits: shape (15, K)
- neg_logits_on_draft_topk_ids: shape (15, K)
- block_position: shape (15,)
- absolute_position: shape (15,)
- optional alpha_prev: shape (15, 3)

Notes:

- Derived statistics (entropy, margins, KL, mass) are optional engineered features computed offline from raw logits.
- Keep raw aligned logits as source of truth.

### Action A_t

alpha_t in R^(15 x 3), typically bounded to [alpha_min, alpha_max].

Interpretation:

- 3 alpha values per diffusion position
- each alpha controls one rank bucket (1..10, 11..20, 21..K)

### Transition P

Given S_t and A_t:

- proposal distribution changes through contrastive logits
- verifier decides accepted prefix length and reject fix token
- committed tokens update context for next block
- therefore S_(t+1) depends on both current state and action

### Reward R_t

Composite design:

- rank improvement reward (target rank moved up)
- top-1 hit bonus
- acceptance-length gain with asymmetric penalty for degradation
- optional KL regularization to prevent excessive distribution drift

Total reward template:

R_t = w1 *r1_t + w2* r2_t + w3 *r3_t - beta* KL_t

Recommended initial weights:

- w1 = 0.1
- w2 = 0.1
- w3 = 1.0
- lambda (downside penalty) = 3.0
- beta = 0.02

## 4. Why this is sequential (not pure independent bandit)

- Accepted/rejected tokens modify future context.
- Future verifier behavior depends on current committed tokens.
- Action at block t can improve or harm later blocks.

Implication:

- Contextual bandit is useful as a fast baseline.
- Full RL (for example PPO style training) is better for long-horizon optimization.

## 5. Practical training strategy

Phase 1: data collection and offline feature construction

- collect raw block records
- compute engineered features and reward targets offline

Phase 2: baseline policy

- train contextual policy with block-local reward
- validate acceptance and quality metrics

Phase 3: sequential fine-tuning

- move to RL objective over full episodes
- keep KL or trust-region style stabilization

## 6. Core assumptions to validate experimentally

1) Improving per-block acceptance generally increases sentence-level acceptance.
2) 3-bucket alpha parameterization is expressive enough for gains.
3) KL regularization prevents quality collapse.
4) Gains transfer across datasets/prompts and not only training distribution.

## 7. Metrics to track

Primary:

- average acceptance length
- decoding speedup versus baseline alpha

Safety/quality:

- degradation rate (blocks with acceptance drop)
- KL drift statistics
- answer quality task metric (if available)

Stability:

- variance of rewards across seeds
- policy entropy or alpha distribution coverage
