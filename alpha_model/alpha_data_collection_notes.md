# Alpha data collection notes

## 1) Problem summary

Goal: train an alpha-policy model to choose adaptive alpha for contrastive decoding in DFlash per decode block.

Current setting:
- block size B = 16 -> B-1 = 15 diffusion positions per block
- action uses 3 alpha buckets per position (rank buckets)
- data collection script should log raw data needed for training and offline feature/reward computation
- derived stats (entropy, KL, margin, reward terms) can be computed later in another script

## 2) Final data fields to collect (per block)

Only keep fields required for alpha training + exact contrastive reconstruction.

### A. Identity and indexing
- sample_id: int
- turn_index: int
- block_index: int

### B. Positive (draft) side
- draft_topk_token_ids: int32 tensor, shape (15, K)
- draft_topk_logits: float16/float32 tensor, shape (15, K)

### C. Negative logits aligned to draft tokens (critical)
- neg_logits_on_draft_topk_ids: float16/float32 tensor, shape (15, K)

Definition:
neg_logits_on_draft_topk_ids[i, j] = neg_full_logits[i, draft_topk_token_ids[i, j]]

Reason:
contrastive decoding is applied on draft token set, so negative logits must be gathered on the same token ids.

### D. Position features used by policy
- block_position: float32 tensor, shape (15,)  (0..14 or normalized)
- absolute_position: float32 tensor, shape (15,) (normalized if used)

### E. Targets and verification outputs
- target_token_id: int32 tensor, shape (15,)
- acceptance_length: int32 scalar (accepted length for this block)

### F. Optional state memory (only if used by policy)
- alpha_prev: float32 tensor, shape (15, 3)

## 3) Fields explicitly skipped for this collection pass

Skipped because not direct training input or can be derived offline:
- prefix_input_ids
- num_input_tokens
- entropy/margin/top-k mass/KL
- rank deltas and reward components
- timing/profiling metrics
- negative model own top-k ids/logits (not needed if C is logged)

## 4) Contrastive formula reminder

For each position i and rank bucket b(j):

z_cd[i, j] = draft_topk_logits[i, j] - alpha[i, b(j)] * neg_logits_on_draft_topk_ids[i, j]

This is the core reason field C is mandatory.

## 5) Suggested storage schema

Per block record (npz/parquet/hdf5 row group):
- sample_id, turn_index, block_index
- draft_topk_token_ids (15, K)
- draft_topk_logits (15, K)
- neg_logits_on_draft_topk_ids (15, K)
- block_position (15,)
- absolute_position (15,)
- target_token_id (15,)
- acceptance_length ()
- alpha_prev (15, 3) [optional]

## 6) Work progress and next tasks

Done:
- clarified MDP framing and sequential dependency impact
- decided to collect raw block-level data only
- finalized mandatory aligned negative-logit field on draft token ids

Next:
1. add logging fields above into alpha_model/data_collecting.py
2. choose output format (npz vs hdf5) and dtype policy
3. run a small pilot collection (10-50 samples)
4. validate shape consistency and missing-field rate
5. build offline feature/reward script from collected raw records
6. start baseline alpha-policy training with fixed K
