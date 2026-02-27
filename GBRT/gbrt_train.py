import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from gbrt_model import GBRTModel

# ==========================================
# 0. Train Setting
# ==========================================
MAX_EPOCHS = 500
PATIENCE = 30
best_val_loss = float('inf')
epochs_no_improve = 0

# ==========================================
# 1. โหลดพารามิเตอร์ที่ดีที่สุดจาก JSON
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"
MODEL_DIR = BASE_DIR / "GBRT"
param_path = MODEL_DIR / "gbrt_best_params.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ [Train Status] สตาร์ทเครื่องยนต์เทรน TabM ร่างทอง ด้วย: {device}")
try:
    with open(param_path, 'r') as f:
        best_params = json.load(f)
    print(f"โหลดพารามิเตอร์จาก Tuner สำเร็จ: {best_params}")
except FileNotFoundError:
    print(f"ไม่พบไฟล์ {param_path}! จะใช้ค่า Default แทน")
    best_params = {"n_layers": 2, "n_neurons": 64, "lr": 0.001, "dropout_rate": 0.1, "weight_decay": 1e-4, "batch_size": 128}

# ==========================================
# 2. เตรียมข้อมูลและ Hybrid Feature
# ==========================================
X_train_raw = np.load(DATA_DIR / 'X_num_train.npy')
y_train_raw = np.load(DATA_DIR / 'Y_train.npy').ravel()
X_val_raw = np.load(DATA_DIR / 'X_num_val.npy') # โหลด Val มาใช้คุม Early Stopping
y_val_raw = np.load(DATA_DIR / 'Y_val.npy').ravel()
X_test_raw = np.load(DATA_DIR / 'X_num_test.npy')
y_test_raw = np.load(DATA_DIR / 'Y_test.npy').ravel()

# แบ่งส่วน GBRT Extractor
X_tr_gbrt, X_tr_mlp, y_tr_gbrt, y_tr_mlp = train_test_split(X_train_raw, y_train_raw, test_size=0.5, random_state=42)

print("Step 1: Training GBRT Extractor...")
gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbrt.fit(X_tr_gbrt, y_tr_gbrt)

encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(gbrt.apply(X_tr_mlp)).toarray()
X_val_encoded = encoder.transform(gbrt.apply(X_val_raw)).toarray() # แปลง Val
X_test_encoded = encoder.transform(gbrt.apply(X_test_raw)).toarray()

# แปลงเป็น Tensors
X_train_t = torch.tensor(X_train_encoded).float()
y_train_t = torch.tensor(y_tr_mlp).float().view(-1, 1)
X_val_t = torch.tensor(X_val_encoded).float()
y_val_t = torch.tensor(y_val_raw).float().view(-1, 1)
X_test_t = torch.tensor(X_test_encoded).float()

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=best_params["batch_size"], shuffle=True)

# ==========================================
# 2. สร้างโมเดล
# ==========================================
model = GBRTModel(
    input_dim=X_train_encoded.shape[1],
    n_layers=best_params["n_layers"],
    n_neurons=best_params["n_neurons"],
    dropout_rate=best_params["dropout_rate"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
criterion = nn.MSELoss()

# ==========================================
# 3. เทรนจริงพร้อม Early Stopping (Fairness Version)
# ==========================================
best_model_path = MODEL_DIR / "mlp_gbrt_model.pt"

print(f"Step 2: Training MLP with Early Stopping (Max {MAX_EPOCHS} Epochs)...")
for epoch in range(MAX_EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X.to(device)), batch_y.to(device))
        loss.backward()
        optimizer.step()

    # --- Validation Check ---
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t.to(device))
        val_loss = criterion(val_pred, y_val_t.to(device)).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{MAX_EPOCHS}] | Val MSE: {val_loss:.4f}")

    # ระบบ Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path) # เซฟจุดที่ดีที่สุด
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"Early Stopping at epoch {epoch+1}")
        break

# ==========================================
# 4. วัดผล
# ==========================================
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy()

test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
test_r2 = r2_score(y_test_raw, y_pred)

print(f"\n[FINAL] RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")

# บันทึกไฟล์ผลลัพธ์ (แก้ชื่อไฟล์ไม่ให้ทับกับ TabM)
results_file = MODEL_DIR / "mlp_gbrt_final_results.json"
final_results = {
    "model_name": "MLP + GBRT",
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2)
}
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)
print(f"✅ บันทึกผลสำเร็จที่ {results_file}")