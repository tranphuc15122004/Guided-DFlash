import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List, Optional

class HDF5BanditDataset(Dataset):
    def __init__(self, h5_path: str, device: str = "cpu"):
        self.h5_path = Path(h5_path)
        self.device = device

        # Mở HDF5 và load toàn bộ dữ liệu vào RAM ngay lập tức.
        # HDF5 hiện tại có chunking cực kỳ tệ cho random access
        # (ví dụ chunks=(63622,1,1) → 1 record cần 480 chunk reads),
        # nên đọc từ HDF5 mỗi lần __getitem__ là cực chậm (~1000s/batch).
        self.h5 = h5py.File(self.h5_path, 'r')
        self.num_records = self.h5.attrs['num_records']

        # Các field cần cho training
        self.required_fields = [
            "draft_topk_logits", "neg_logits_on_draft_topk_ids",
            "draft_topk_token_ids", "target_token_id",
            "block_position", "absolute_position", "alpha_prev",
            "acceptance_length"
        ]
        missing = [f for f in self.required_fields if f not in self.h5]
        if missing:
            raise ValueError(f"Missing fields in HDF5: {missing}")

        # Đọc toàn bộ dữ liệu vào RAM — ~23GB, nằm trong 48GB budget
        print(f"Loading {self.num_records} records into memory...")
        self.pos_logits = torch.from_numpy(self.h5["draft_topk_logits"][:]).float()
        self.neg_logits = torch.from_numpy(self.h5["neg_logits_on_draft_topk_ids"][:]).float()
        self.topk_token_ids = torch.from_numpy(self.h5["draft_topk_token_ids"][:]).long()
        self.target_token_id = torch.from_numpy(self.h5["target_token_id"][:]).long()
        self.block_pos = torch.from_numpy(self.h5["block_position"][:]).float()
        self.abs_pos = torch.from_numpy(self.h5["absolute_position"][:]).float()
        self.alpha_prev = torch.from_numpy(self.h5["alpha_prev"][:]).float()
        self.baseline_acc_len = torch.from_numpy(self.h5["acceptance_length"][:]).float()
        self.h5.close()  # Đóng file HDF5 — không cần nữa

        self.S = self.pos_logits.shape[1]  # = block_size-1
        self.K = self.pos_logits.shape[2]  # = top_k
        self.num_buckets = self.alpha_prev.shape[2]  # derive from stored alpha width
        print(f"Loaded {self.num_records} records, S={self.S}, K={self.K}, memory=~23GB")

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        return {
            "pos_logits": self.pos_logits[idx],
            "neg_logits": self.neg_logits[idx],
            "topk_token_ids": self.topk_token_ids[idx],
            "target_token_ids": self.target_token_id[idx],
            "block_pos": self.block_pos[idx],
            "abs_pos": self.abs_pos[idx],
            "alpha_prev": self.alpha_prev[idx],
            "baseline_acc_len": self.baseline_acc_len[idx].item(),
        }

def collate_bandit_batch(
    batch: List[Dict[str, Any]],
    augment_neg_std: float = 0.0,
    augment_pos_std: float = 0.0,
    augment_prob: float = 1.0,
) -> Dict[str, Any]:
    """Gộp batch từ các dict."""
    pos_logits = torch.stack([item["pos_logits"] for item in batch], dim=0)
    neg_logits = torch.stack([item["neg_logits"] for item in batch], dim=0)
    topk_token_ids = torch.stack([item["topk_token_ids"] for item in batch], dim=0)
    target_token_ids = torch.stack([item["target_token_ids"] for item in batch], dim=0)
    block_pos = torch.stack([item["block_pos"] for item in batch], dim=0)
    abs_pos = torch.stack([item["abs_pos"] for item in batch], dim=0)
    alpha_prev = torch.stack([item["alpha_prev"] for item in batch], dim=0)
    baseline_acc_len = torch.tensor([item["baseline_acc_len"] for item in batch], dtype=torch.float32)

    if augment_prob > 0.0 and (augment_neg_std > 0.0 or augment_pos_std > 0.0):
        if augment_prob >= 1.0 or torch.rand(()) < augment_prob:
            if augment_pos_std > 0.0:
                pos_logits = pos_logits + augment_pos_std * torch.randn_like(pos_logits)
            if augment_neg_std > 0.0:
                neg_logits = neg_logits + augment_neg_std * torch.randn_like(neg_logits)

    return {
        "pos_logits": pos_logits,
        "neg_logits": neg_logits,
        "topk_token_ids": topk_token_ids,
        "target_token_ids": target_token_ids,
        "block_pos": block_pos,
        "abs_pos": abs_pos,
        "alpha_prev": alpha_prev,
        "baseline_acc_len": baseline_acc_len,
    }

# ========== Kiểm thử ==========
if __name__ == "__main__":
    ds = HDF5BanditDataset("alpha_model/alpha_dataset.h5")
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
    
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_bandit_batch)
    batch = next(iter(dl))
    print("\nBatch shapes:")
    for k, v in batch.items():
        print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")