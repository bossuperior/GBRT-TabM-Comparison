import json
from pathlib import Path
import numpy as np
import torch


def load_california_tabm(data_dir: str = "data/california"):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")

    if not (data_path / "READY").exists():
        raise RuntimeError("Dataset not READY (missing READY file)")

    X_train = np.load(data_path / "X_num_train.npy")
    X_val = np.load(data_path / "X_num_val.npy")
    X_test = np.load(data_path / "X_num_test.npy")

    y_train = np.load(data_path / "Y_train.npy")
    y_val = np.load(data_path / "Y_val.npy")
    y_test = np.load(data_path / "Y_test.npy")

    info = {}
    info_path = data_path / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), info


# เพิ่มฟังก์ชันนี้เพื่อใช้ใน main.py
def get_california_tensors(data_dir: str = "data/california"):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), info = load_california_tabm(data_dir)

    # แปลงเป็น FloatTensor และเปลี่ยนรูปทรง y ให้เป็น (N, 1)
    train_data = (torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).unsqueeze(1))
    val_data = (torch.FloatTensor(X_va), torch.FloatTensor(y_va).unsqueeze(1))
    test_data = (torch.FloatTensor(X_te), torch.FloatTensor(y_te).unsqueeze(1))

    return train_data, val_data, test_data, info
