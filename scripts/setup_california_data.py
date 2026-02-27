import json
from pathlib import Path
import zipfile
import numpy as np
import gdown

REQUIRED_FILES = [
    "X_num_train.npy", "X_num_val.npy", "X_num_test.npy",
    "Y_train.npy", "Y_val.npy", "Y_test.npy",
    "info.json", "READY",
]

def main():
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data" / "california"
    if not data_dir.exists():
        raise RuntimeError("แตกไฟล์แล้วไม่เจอ data/california/ ตรวจสอบว่า zip มีโครงสร้างถูกต้อง")

    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"ไฟล์ใน data/california ขาด: {missing}")

    # Verify shapes + load info
    Xtr = np.load(data_dir / "X_num_train.npy")
    Xva = np.load(data_dir / "X_num_val.npy")
    Xte = np.load(data_dir / "X_num_test.npy")
    ytr = np.load(data_dir / "Y_train.npy")
    yva = np.load(data_dir / "Y_val.npy")
    yte = np.load(data_dir / "Y_test.npy")

    info = json.loads((data_dir / "info.json").read_text(encoding="utf-8"))

    print("[OK] California dataset is ready at:", data_dir)
    print("INFO:", info)
    print("Shapes:")
    print(" train:", Xtr.shape, ytr.shape)
    print(" val  :", Xva.shape, yva.shape)
    print(" test :", Xte.shape, yte.shape)

    # Sanity checks
    assert Xtr.shape[0] == ytr.shape[0]
    assert Xva.shape[0] == yva.shape[0]
    assert Xte.shape[0] == yte.shape[0]
    assert Xtr.shape[1] == Xva.shape[1] == Xte.shape[1]
    print("[PASS] sanity checks")


if __name__ == "__main__":
    main()