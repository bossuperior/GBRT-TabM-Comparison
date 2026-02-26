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
# 1. โหลดพารามิเตอร์ที่ดีที่สุดจาก JSON
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"
MODEL_DIR = BASE_DIR / "GBRT"
param_path = MODEL_DIR / "gbrt_best_params.json"

try:
    with open(param_path, 'r') as f:
        best_params = json.load(f)
    print(f"✅ โหลดพารามิเตอร์จาก Tuner สำเร็จ: {best_params}")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {param_path}! จะใช้ค่า Default แทน")
    best_params = {"n_layers": 2, "n_neurons": 64, "lr": 0.001, "dropout_rate": 0.1, "weight_decay": 1e-4, "batch_size": 128}

# ==========================================
# 2. เตรียมข้อมูลและ Hybrid Feature (เหมือนเดิม)
# ==========================================
X_train = np.load(DATA_DIR / 'X_num_train.npy')
y_train = np.load(DATA_DIR / 'Y_train.npy').ravel()
X_test = np.load(DATA_DIR / 'X_num_test.npy')
y_test = np.load(DATA_DIR / 'Y_test.npy').ravel()

X_tr_gbrt, X_tr_mlp, y_tr_gbrt, y_tr_mlp = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

print("🌲 Step 1: Training GBRT Extractor...")
gbrt = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbrt.fit(X_tr_gbrt, y_tr_gbrt)

encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(gbrt.apply(X_tr_mlp)).toarray()
X_test_encoded = encoder.transform(gbrt.apply(X_test)).toarray()

# แปลงเป็น Tensor และสร้าง DataLoader ตาม Batch Size ที่จูนมา
X_train_t = torch.tensor(X_train_encoded).float()
y_train_t = torch.tensor(y_tr_mlp).float().view(-1, 1)
X_test_t = torch.tensor(X_test_encoded).float()

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=best_params["batch_size"],
    shuffle=True
)

# ==========================================
# 3. สร้างโมเดลแบบ Dynamic ตามค่าที่จูนได้
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GBRTModel(
    input_dim=X_train_encoded.shape[1],
    n_layers=best_params["n_layers"],
    n_neurons=best_params["n_neurons"],
    dropout_rate=best_params["dropout_rate"]
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)
criterion = nn.MSELoss()

# ==========================================
# 4. เทรนจริงพร้อม Early Stopping (ร่างสมบูรณ์)
# ==========================================
print("🧠 Step 2: Training Final MLP...")
model.train()
for epoch in range(200): # เทรนจริงเพิ่มเป็น 200 Epoch
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X.to(device)), batch_y.to(device))
        loss.backward()
        optimizer.step()

# ==========================================
# 5. สรุปผลและเซฟโมเดลให้ครบชุด
# ==========================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t.to(device)).cpu().numpy()

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print(f"\n🏆 [FINAL] RMSE: {test_rmse:.4f} | R2: {test_r2:.4f}")

# เซฟ 3 ทหารเสือ
torch.save(model.state_dict(), MODEL_DIR / "mlp_gbrt_model.pt")
joblib.dump(gbrt, MODEL_DIR / "gbrt_extractor.joblib")
joblib.dump(encoder, MODEL_DIR / "leaf_encoder.joblib")

print(f"✅ บันทึกโมเดลร่างสมบูรณ์ที่ผ่านการจูนแล้วลงใน {MODEL_DIR} เรียบร้อย!")