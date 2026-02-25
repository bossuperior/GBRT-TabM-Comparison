from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import EarlyStopper
import numpy as np
from sklearn.metrics import mean_squared_error
from gbrt_model import FlexibleMLP
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#ใส่เงื่อนไขจำนวนรอบและ RMSE ในการจูน
MAX_CALLS = 50
TARGET_RMSE = 0.65

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

X_train = np.load(DATA_DIR / "X_num_train.npy")
y_train = np.load(DATA_DIR / "Y_train.npy")
X_val = np.load(DATA_DIR / "X_num_val.npy")
y_val = np.load(DATA_DIR / "Y_val.npy")

# --- แปลงข้อมูลเป็น Tensor เอาไว้ล่วงหน้า ---
X_train_t = torch.tensor(X_train).float()
y_train_t = torch.tensor(y_train).float().view(-1, 1)
X_val_t = torch.tensor(X_val).float()

# ==========================================
# 1. สร้าง Dataset เตรียมไว้ให้ DataLoader
# ==========================================
train_dataset = TensorDataset(X_train_t, y_train_t)

iteration_count = 0


def objective(params):
    global iteration_count
    iteration_count += 1

    # 2. แตกตัวแปร 6 ตัว (รับ batch_size เพิ่มเข้ามา)
    n_layers, n_neurons, lr, dropout_rate, weight_decay, batch_size = params

    # 3. สร้าง DataLoader เพื่อสับข้อมูลเป็นก้อนๆ ตามขนาด batch_size ที่ GBRT สุ่มมา
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)

    model = FlexibleMLP(n_layers=n_layers, n_neurons=n_neurons, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(20):
        # ==========================================
        # 4. วนลูปย่อยเพื่อดึงข้อมูลทีละ Batch มาเทรน
        # ==========================================
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).numpy()
        rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(
        f"⏳ รอบ {iteration_count}: layers={n_layers}, neurons={n_neurons}, lr={lr:.5f}, drop={dropout_rate:.2f}, wd={weight_decay:.5f}, batch={batch_size} -> RMSE: {rmse:.4f}")
    return rmse


class TargetScoreStopper(EarlyStopper):
    def __init__(self, target_score):
        self.target_score = target_score

    def _criterion(self, result):
        if result.fun <= self.target_score:
            print(
                f"\n🎉 หยุดการจูนก่อนกำหนด! พบค่า RMSE ({result.fun:.4f}) ซึ่งผ่านเกณฑ์เป้าหมาย ({self.target_score}) แล้ว!")
            return True
        return False


# ==========================================
# 5. กำหนดขอบเขตการค้นหา (เพิ่ม Batch Size แบบเลขฐาน 2)
# ==========================================
search_space = [
    Integer(1, 5, name='n_layers'),
    Integer(32, 256, name='n_neurons'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
    Real(0.0, 0.5, name='dropout_rate'),
    Real(1e-5, 1e-3, prior='log-uniform', name='weight_decay'),
    Categorical([32, 64, 128, 256, 512], name='batch_size')  # ใช้ Categorical บังคับให้สุ่มเฉพาะเลขกลุ่มนี้
]

stopper = TargetScoreStopper(target_score=TARGET_RMSE)

print(f"กำลังเริ่มให้ GBRT จูน 6 พารามิเตอร์... (รอบสูงสุด: {MAX_CALLS}, เป้าหมาย RMSE: <= {TARGET_RMSE})")
result = forest_minimize(objective, search_space, n_calls=MAX_CALLS, callback=[stopper], random_state=42)

print("\n=== ผลการจูนเสร็จสิ้น ===")
print(f"ค่าที่ดีที่สุด (Layers, Neurons, LR, Dropout, WeightDecay, BatchSize): {result.x}")
print(f"ค่า RMSE ที่ต่ำที่สุดที่ทำได้: {result.fun:.4f}")