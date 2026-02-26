import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
import tabm

# ==========================================
# 0. ปิด Log มาตรฐานของ Optuna เพื่อให้หน้าจอสะอาด
# ==========================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. ตั้งค่าพื้นฐานและโหลดข้อมูล
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ [Tuner Status] จูน TabM บน: {device}")

X_train = torch.tensor(np.load(DATA_DIR / "X_num_train.npy")).float()
y_train = torch.tensor(np.load(DATA_DIR / "Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(DATA_DIR / "X_num_val.npy")).float()
y_val = torch.tensor(np.load(DATA_DIR / "Y_val.npy")).float().view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)


# ==========================================
# 2. ฟังก์ชัน Objective
# ==========================================
def objective(trial):
    # สุ่มพารามิเตอร์
    n_blocks = trial.suggest_int("n_blocks", 1, 5)
    d_block = trial.suggest_int("d_block", 64, 1024, step=16)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    k = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # สร้างโมเดล
    model = nn.Sequential(
        tabm.EnsembleView(k=k),
        tabm.MLPBackboneBatchEnsemble(
            d_in=8,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=k,
            tabm_init=True,
            scaling_init='normal',
            start_scaling_init_chunks=None,
        ),
        tabm.LinearEnsemble(d_block, 1, k=k)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training Loop แบบย่อ (20 Epochs)
    model.train()
    for epoch in range(20):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)

            y_expanded = batch_y.unsqueeze(1).expand(-1, k, -1)
            loss = criterion(outputs, y_expanded)
            loss.backward()
            optimizer.step()

    # Validation Phase
    model.eval()
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_outputs = model(X_val_dev)
        final_val_pred = val_outputs.mean(dim=1).cpu().numpy()
        y_val_np = y_val_dev.cpu().numpy()

        val_rmse = np.sqrt(mean_squared_error(y_val_np, final_val_pred))

    # --- 🌟 แสดงผลแบบเดียวกับ GBRT 🌟 ---
    iteration_count = trial.number + 1
    print(
        f"⏳ รอบ {iteration_count:2d}: layers={n_blocks}, neurons={d_block:4d}, lr={lr:.5f}, drop={dropout:.2f}, wd={weight_decay:.5f}, batch={batch_size} -> RMSE: {val_rmse:.4f}")

    return val_rmse


# ==========================================
# 3. รัน 50 รอบและบันทึกผล
# ==========================================
if __name__ == "__main__":
    print("🚀 เริ่มการจูน TabM ด้วย Optuna จำนวน 50 รอบ...\n")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n🎉 === สรุปผลการจูน TabM ===")
    print(f"🏆 RMSE ที่ดีที่สุด: {study.best_value:.4f}")
    print(f"💡 พารามิเตอร์ที่ชนะเลิศ: {study.best_params}")

    best_params_file = BASE_DIR / "TabM_R2" / "tabm_best_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    print(f"✅ บันทึกสูตรสำเร็จลงใน {best_params_file} เรียบร้อยแล้ว!")