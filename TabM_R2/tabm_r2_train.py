import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
import random
import rtdl_num_embeddings
import tabm

# ==========================================
# 0. Train Setting
# ==========================================
MAX_EPOCHS = 500
PATIENCE = 30
epochs_no_improve = 0

# ==========================================
# 1. ตั้งค่าพื้นฐานและโหลด Data
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ [Train Status] สตาร์ทเครื่องยนต์เทรน TabM ร่างทอง ด้วย: {device}")

X_train = torch.tensor(np.load(DATA_DIR / "X_num_train.npy")).float()
y_train = torch.tensor(np.load(DATA_DIR / "Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(DATA_DIR / "X_num_val.npy")).float()
y_val = torch.tensor(np.load(DATA_DIR / "Y_val.npy")).float().view(-1, 1)

X_test = torch.tensor(np.load(DATA_DIR / "X_num_test.npy")).float()
y_test_np = np.load(DATA_DIR / "Y_test.npy")

# ==========================================
# 2. 🌟 โหลดพารามิเตอร์อัตโนมัติจาก JSON 🌟
# ==========================================
json_path = BASE_DIR / "TabM_R2" / "tabm_best_params.json"

try:
    with open(json_path, 'r') as f:
        best_params = json.load(f)
    print(f"✅ โหลด Hyperparameters สำเร็จ: {best_params}")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {json_path} กรุณารัน Tuner ก่อน")

BEST_N_BLOCKS = best_params["n_blocks"]
BEST_D_BLOCK = best_params["d_block"]
BEST_LR = best_params["lr"]
BEST_WEIGHT_DECAY = best_params["weight_decay"]
BEST_DROPOUT = best_params["dropout"]
BEST_BATCH_SIZE = best_params["batch_size"]

# ดึงพารามิเตอร์ Embeddings ออกมา
BEST_N_BINS = best_params["n_bins"]
BEST_D_EMBEDDING = best_params["d_embedding"]

K_ENSEMBLE = 32

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True)

# ==========================================
# 3. ประกอบร่าง TabM + Piecewise Linear Embeddings
# ==========================================
# คำนวณขอบเขต Bin จาก X_train
bins = rtdl_num_embeddings.compute_bins(X_train, n_bins=BEST_N_BINS)
ple_layer = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, d_embedding=BEST_D_EMBEDDING, activation=nn.GELU(), version="A")
d_in_expanded = 8 * BEST_D_EMBEDDING

model = nn.Sequential(
    ple_layer,
    nn.Flatten(1),
    tabm.EnsembleView(k=K_ENSEMBLE),
    tabm.MLPBackboneBatchEnsemble(
        d_in=d_in_expanded,
        n_blocks=BEST_N_BLOCKS,
        d_block=BEST_D_BLOCK,
        dropout=BEST_DROPOUT,
        k=K_ENSEMBLE,
        tabm_init=True,
        scaling_init='normal',
        start_scaling_init_chunks=None,
    ),
    tabm.LinearEnsemble(BEST_D_BLOCK, 1, k=K_ENSEMBLE)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
criterion = nn.MSELoss()

# ==========================================
# 4. Training Loop
# ==========================================

best_val_loss = float('inf')
best_model_path = BASE_DIR / "TabM_R2" / "tabm_model.pt"
print(f"🚀 เริ่มฝึก TabM Model (สูงสุด {MAX_EPOCHS} Epochs)...")

for epoch in range(MAX_EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)

        y_expanded = batch_y.unsqueeze(1).expand(-1, K_ENSEMBLE, -1)
        loss = criterion(outputs, y_expanded)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_outputs = model(X_val_dev)
        final_val_pred = val_outputs.mean(dim=1)
        val_loss = criterion(final_val_pred, y_val_dev).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:3d}/{MAX_EPOCHS}] | Val MSE: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 No score improve Early Stopping ที่รอบ {epoch + 1}")
        break

# ==========================================
# 5. ประเมินผล (Test Set)
# ==========================================
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    X_test_dev = X_test.to(device)
    test_outputs = model(X_test_dev)
    final_test_pred = test_outputs.mean(dim=1).cpu().numpy()

    final_test_rmse = np.sqrt(mean_squared_error(y_test_np, final_test_pred))
    final_test_r2 = r2_score(y_test_np, final_test_pred)

print("\n=========================================")
print(f"🏆 ผลการวัดประสิทธิภาพ TabM Model (TEST SET)")
print(f"RMSE: {final_test_rmse:.4f}")
print(f"R² Score: {final_test_r2:.4f}")
print("=========================================")

results_file = BASE_DIR / "TabM_R2" / "tabm_final_results.json" # เซฟทับชื่อเดิมเพื่อให้ main.py อ่านง่าย
final_results = {
    "model_name": "TabM + Piecewise Linear Embeddings",
    "test_rmse": float(final_test_rmse),
    "test_r2": float(final_test_r2)
}
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)
print(f"✅ บันทึกผลการวัดประสิทธิภาพเรียบร้อย!")