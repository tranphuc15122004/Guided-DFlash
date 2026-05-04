import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, List

class HDF5BanditDataset(Dataset):
    def __init__(self, h5_path: str, device: str = "cpu"):
        self.h5_path = Path(h5_path)
        self.device = device
        self.h5 = h5py.File(self.h5_path, 'r')
        self.num_records = self.h5.attrs['num_records']
        # Lấy thông tin shape
        self.layout = {}
        for name in self.h5.keys():
            ds = self.h5[name]
            self.layout[name] = {"shape": ds.shape[1:], "dtype": ds.dtype}
        # Kiểm tra các field bắt buộc cho training
        self.required_fields = [
            "draft_topk_logits", "neg_logits_on_draft_topk_ids",
            "draft_topk_token_ids", "target_token_id",
            "block_position", "absolute_position", "alpha_prev",
            "acceptance_length"
        ]
        missing = [f for f in self.required_fields if f not in self.layout]
        if missing:
            raise ValueError(f"Missing fields in HDF5: {missing}")
        self.S = self.layout["draft_topk_logits"]["shape"][0]  # = block_size-1
        self.K = self.layout["draft_topk_logits"]["shape"][1]  # = top_k
        self.num_buckets = self.layout["alpha_prev"]["shape"][1]  # = 3
        print(f"Loaded {self.num_records} records, S={self.S}, K={self.K}")

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        # Đọc từ HDF5 – mỗi field là mảng (record_index, ...)
        pos_logits = torch.tensor(self.h5["draft_topk_logits"][idx], dtype=torch.float32)
        neg_logits = torch.tensor(self.h5["neg_logits_on_draft_topk_ids"][idx], dtype=torch.float32)
        topk_token_ids = torch.tensor(self.h5["draft_topk_token_ids"][idx], dtype=torch.long)
        target_token_ids = torch.tensor(self.h5["target_token_id"][idx], dtype=torch.long)
        block_pos = torch.tensor(self.h5["block_position"][idx], dtype=torch.float32)
        abs_pos = torch.tensor(self.h5["absolute_position"][idx], dtype=torch.float32)
        alpha_prev = torch.tensor(self.h5["alpha_prev"][idx], dtype=torch.float32)
        baseline_acc_len = int(self.h5["acceptance_length"][idx])
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

    def close(self):
        self.h5.close()

def collate_bandit_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gộp batch từ các dict."""
    pos_logits = torch.stack([item["pos_logits"] for item in batch], dim=0)
    neg_logits = torch.stack([item["neg_logits"] for item in batch], dim=0)
    topk_token_ids = torch.stack([item["topk_token_ids"] for item in batch], dim=0)
    target_token_ids = torch.stack([item["target_token_ids"] for item in batch], dim=0)
    block_pos = torch.stack([item["block_pos"] for item in batch], dim=0)
    abs_pos = torch.stack([item["abs_pos"] for item in batch], dim=0)
    alpha_prev = torch.stack([item["alpha_prev"] for item in batch], dim=0)
    baseline_acc_len = torch.tensor([item["baseline_acc_len"] for item in batch], dtype=torch.float32)
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