import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path
import json
import rtdl_num_embeddings
import tabm
optuna.logging.set_verbosity(optuna.logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ [Tuner Status] เริ่มจูน TabM Model บน: {device}")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

# ==========================================
# 2. โหลดข้อมูล
# ==========================================
X_train = torch.tensor(np.load(DATA_DIR / "X_num_train.npy")).float()
y_train = torch.tensor(np.load(DATA_DIR / "Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(DATA_DIR / "X_num_val.npy")).float()
y_val = torch.tensor(np.load(DATA_DIR / "Y_val.npy")).float().view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)

# ==========================================
# 2. ฟังก์ชัน Objective (มี Embeddings)
# ==========================================
def objective(trial):
    # --- 1. สุ่มพารามิเตอร์ ---
    n_blocks = trial.suggest_int("n_blocks", 1, 4)
    d_block = trial.suggest_int("d_block", 64, 1024, step=16)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # 🌟 พารามิเตอร์ใหม่สำหรับ Embeddings 🌟
    n_bins = trial.suggest_int("n_bins", 2, 128)
    d_embedding = trial.suggest_int("d_embedding", 8, 32, step=4)

    k = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 2. สร้างโครงสร้าง Piecewise Linear Embeddings ---
    # คำนวณ Bins จาก X_train ก่อน
    bins = rtdl_num_embeddings.compute_bins(X_train, n_bins=n_bins)
    ple_layer = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=d_embedding, activation=nn.GELU(), version="A")

    # ข้อมูล 8 คอลัมน์ จะถูกขยายเป็น (8 * d_embedding) มิติ
    d_in_expanded = 8 * d_embedding

    # --- 3. ประกอบร่างโมเดล ---
    model = nn.Sequential(
        ple_layer,           # แปลง (Batch, 8) -> (Batch, 8, d_embedding)
        nn.Flatten(1),       # Flatten แบบ 2 มิติ -> (Batch, 8 * d_embedding)
        tabm.EnsembleView(k=k),
        tabm.MLPBackboneBatchEnsemble(
            d_in=d_in_expanded,
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

    # --- 4. Training Loop (เพิ่มเป็น 30 Epochs สำหรับจูน) ---
    model.train()
    for epoch in range(30):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)

            y_expanded = batch_y.unsqueeze(1).expand(-1, k, -1)
            loss = criterion(outputs, y_expanded)
            loss.backward()
            optimizer.step()

    # --- 5. Validation Phase ---
    model.eval()
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_outputs = model(X_val_dev)
        final_val_pred = val_outputs.mean(dim=1).cpu().numpy()
        y_val_np = y_val_dev.cpu().numpy()
        val_rmse = np.sqrt(mean_squared_error(y_val_np, final_val_pred))

    iteration_count = trial.number + 1
    print(f"⏳ รอบ {iteration_count:2d}: layers={n_blocks}, neurons={d_block:4d}, bins={n_bins}, d_emb={d_embedding}, lr={lr:.5f} -> RMSE: {val_rmse:.4f}")

    return val_rmse

# ==========================================
# 3. รัน Optimize และบันทึก
# ==========================================
if __name__ == "__main__":
    print("🚀 เริ่มการจูน TabM Model (50 รอบ)...\n")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n🎉 === สรุปผลการจูน TabM Model ===")
    print(f"🏆 RMSE ที่ดีที่สุด: {study.best_value:.4f}")

    tabm_dir = BASE_DIR / "TabM_R2"

    best_params_file = tabm_dir / "tabm_best_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    print(f"✅ บันทึก Hyperparameters ({best_params_file}) เรียบร้อยแล้ว!")