import numpy as np
import json
import torch
import torch.nn as nn
import sklearn.base
import skopt.optimizer.base
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import gbrt_minimize
from skopt.space import Real, Integer, Categorical
from pathlib import Path
from gbrt_model import GBRTModel
from skopt.learning import GradientBoostingQuantileRegressor

MAX_CALLS = 50
TARGET_RMSE = 0.55

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

X_train_full = np.load(DATA_DIR / "X_num_train.npy")
y_train_full = np.load(DATA_DIR / "Y_train.npy").ravel()
X_val = np.load(DATA_DIR / "X_num_val.npy")
y_val = np.load(DATA_DIR / "Y_val.npy").ravel()

# แบ่งส่วน Hybrid เพื่อป้องกัน Data Leakage
X_tr_gbrt, X_tr_mlp, y_tr_gbrt, y_tr_mlp = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42)

print("กำลังเตรียม Feature Extraction สำหรับ Hybrid MLP...")
gbrt_ext = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gbrt_ext.fit(X_tr_gbrt, y_tr_gbrt)

enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
X_tr_encoded = enc.fit_transform(gbrt_ext.apply(X_tr_mlp)).toarray()
X_val_encoded = enc.transform(gbrt_ext.apply(X_val)).toarray()

X_tr_t = torch.tensor(X_tr_encoded).float()
y_tr_t = torch.tensor(y_tr_mlp).float().view(-1, 1)
X_val_t = torch.tensor(X_val_encoded).float()


class FixedGBQR(GradientBoostingQuantileRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._estimator_type = "regressor"
gbrt_base = FixedGBQR(
    base_estimator=GradientBoostingRegressor(n_estimators=30, loss='quantile')
)

def absolute_bypass(estimator):
    return True
skopt.optimizer.optimizer.is_regressor = absolute_bypass
skopt.optimizer.base.is_regressor = absolute_bypass
sklearn.base.is_regressor = absolute_bypass

iteration_count = 0
def objective(params):
    global iteration_count
    iteration_count += 1
    n_layers, n_neurons, lr, dropout_rate, weight_decay, batch_size = params

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=int(batch_size), shuffle=True)

    # สร้างโมเดล Linear Layer ตามที่โจทย์กำหนด
    model = GBRTModel(
        input_dim=X_tr_encoded.shape[1],
        n_layers=n_layers,
        n_neurons=n_neurons,
        dropout_rate=dropout_rate
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(20):
        for b_X, b_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(b_X), b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).numpy()
        rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(
        f"รอบ {iteration_count:2d}: layers={n_layers}, neurons={n_neurons:3d}, lr={lr:.5f}, batch={batch_size:3d} -> RMSE: {rmse:.4f}")
    return rmse

# --- Search Space & Tuning ---
search_space = [
    Integer(1, 5, name='n_layers'),
    Integer(64, 512, name='n_neurons'),
    Real(1e-4, 5e-3, prior='log-uniform', name='lr'),
    Real(0.0, 0.5, name='dropout_rate'),
    Real(1e-5, 1e-3, prior='log-uniform', name='weight_decay'),
    Categorical([64, 128, 256, 512], name='batch_size')
]

print(f"เริ่มต้น GBRT Tuning...")
# ใช้ gbrt_minimize เพื่อให้ตรงตามคำสั่ง "Gradient Based Tuning"
result = gbrt_minimize(objective, search_space, n_calls=MAX_CALLS, base_estimator=gbrt_base, random_state=42)

best_params = {
    "n_layers": int(result.x[0]), "n_neurons": int(result.x[1]),
    "lr": float(result.x[2]), "dropout_rate": float(result.x[3]),
    "weight_decay": float(result.x[4]), "batch_size": int(result.x[5])
}

param_path = BASE_DIR / "GBRT" / "gbrt_best_params.json"
with open(param_path, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"\nสำเร็จ! RMSE ต่ำสุดจากการจูน: {result.fun:.4f}")