import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
import random

import tabm  # อิมพอร์ตไลบรารีของ Yandex

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
print(f"🖥️ [Train Status] สตาร์ทเครื่องยนต์เทรน TabM ด้วย: {device}")

# โหลดข้อมูล
X_train = torch.tensor(np.load(DATA_DIR / "X_num_train.npy")).float()
y_train = torch.tensor(np.load(DATA_DIR / "Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(DATA_DIR / "X_num_val.npy")).float()
y_val = torch.tensor(np.load(DATA_DIR / "Y_val.npy")).float().view(-1, 1)

X_test = torch.tensor(np.load(DATA_DIR / "X_num_test.npy")).float()
y_test_np = np.load(DATA_DIR / "Y_test.npy")

# ==========================================
# 2. 🌟 โหลดพารามิเตอร์อัตโนมัติจากไฟล์ JSON 🌟
# ==========================================
json_path = BASE_DIR / "TabM_R2" / "tabm_best_params.json"

try:
    with open(json_path, 'r') as f:
        best_params = json.load(f)
    print(f"✅ โหลดพารามิเตอร์ที่จูนแล้วสำเร็จ: {best_params}")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {json_path} (รัน tabm_r2_optuna_tuner.py หรือยัง?)")
    print("⚠️ จะใช้ค่า Default ในการเทรนชั่วคราว...")
    best_params = {
        "n_blocks": 3, "d_block": 256, "lr": 0.001,
        "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 256
    }

# ดึงค่าออกมาใส่ตัวแปร
BEST_N_BLOCKS = best_params["n_blocks"]
BEST_D_BLOCK = best_params["d_block"]
BEST_LR = best_params["lr"]
BEST_WEIGHT_DECAY = best_params["weight_decay"]
BEST_DROPOUT = best_params["dropout"]
BEST_BATCH_SIZE = best_params["batch_size"]

K_ENSEMBLE = 32  # ค่าคงที่ของ TabM

# สร้าง DataLoader ด้วย Batch Size ที่ได้จากการจูน
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True)

# ==========================================
# 3. สร้างโมเดล TabM ด้วยพารามิเตอร์ที่ดีที่สุด
# ==========================================
model = nn.Sequential(
    tabm.EnsembleView(k=K_ENSEMBLE),
    tabm.MLPBackboneBatchEnsemble(
        d_in=8,
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
# 4. Training Loop พร้อม Early Stopping
# ==========================================
MAX_EPOCHS = 200
PATIENCE = 20
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = BASE_DIR / "TabM_R2" / "best_tabm_model.pt"

print("🚀 เริ่มฝึกสอน TabM...")

for epoch in range(MAX_EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)

        # ขยาย Label ให้เท่ากับจำนวน Ensemble
        y_expanded = batch_y.unsqueeze(1).expand(-1, K_ENSEMBLE, -1)
        loss = criterion(outputs, y_expanded)
        loss.backward()
        optimizer.step()

    # --- Validation Phase ---
    model.eval()
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_outputs = model(X_val_dev)

        # หาค่าเฉลี่ยของ 32 ร่างเพื่อวัดผล
        final_val_pred = val_outputs.mean(dim=1)
        val_loss = criterion(final_val_pred, y_val_dev).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:3d}/{MAX_EPOCHS}] | Val MSE: {val_loss:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 หยุดเทรนอัตโนมัติที่รอบ {epoch + 1}! โหลดน้ำหนักที่ดีที่สุดกลับมา...")
        break

# ==========================================
# 5. การประเมินผลสนามจริง (Test Set)
# ==========================================
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    X_test_dev = X_test.to(device)
    test_outputs = model(X_test_dev)

    # รวมพลัง 32 ร่าง หาค่าเฉลี่ยตอนสอบไฟนอล
    final_test_pred = test_outputs.mean(dim=1).cpu().numpy()

    final_test_rmse = np.sqrt(mean_squared_error(y_test_np, final_test_pred))
    final_test_r2 = r2_score(y_test_np, final_test_pred)

print("\n=========================================")
print(f"🏆 ผลลัพธ์สุดท้ายของฝั่ง TabM (บน TEST SET")
print(f"RMSE: {final_test_rmse:.4f}")
print(f"R² Score: {final_test_r2:.4f}")
print("=========================================")

# เซฟผลคะแนนเตรียมส่งให้ main.py ทำกราฟเปรียบเทียบ
results_file = BASE_DIR / "TabM" / "tabm_final_results.json"
final_results = {
    "model_name": "TabM (Optuna Tuned)",
    "test_rmse": float(final_test_rmse),
    "test_r2": float(final_test_r2)
}
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)

print(f"✅ บันทึกคะแนนลงใน {results_file} เรียบร้อยแล้ว")