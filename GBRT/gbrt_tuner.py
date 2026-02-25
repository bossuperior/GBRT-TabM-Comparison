from skopt import forest_minimize
from skopt.space import Real, Integer
from skopt.callbacks import EarlyStopper # <-- นำเข้า EarlyStopper
import numpy as np
from sklearn.metrics import mean_squared_error
from gbrt_model import FlexibleMLP
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

X_train = np.load(DATA_DIR / "X_num_train.npy")
y_train = np.load(DATA_DIR / "Y_train.npy")
X_val = np.load(DATA_DIR / "X_num_val.npy")
y_val = np.load(DATA_DIR / "Y_val.npy")

# --- แปลงข้อมูลเป็น Tensor เอาไว้ล่วงหน้าเพื่อให้ทำงานเร็วขึ้น ---
X_train_t = torch.tensor(X_train).float()
y_train_t = torch.tensor(y_train).float().view(-1, 1)
X_val_t = torch.tensor(X_val).float()
# -----------------------------------------------------------
iteration_count = 0

def objective(params):
    global iteration_count  # เรียกใช้ตัวแปรนับรอบจากด้านนอก
    iteration_count += 1  # บวกเพิ่ม 1 ทุกครั้งที่เริ่มรันรอบใหม่
    n_layers, n_neurons, lr = params

    model = FlexibleMLP(n_layers=n_layers, n_neurons=n_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    # เทรนรอบสั้นๆ ให้โมเดลเห็นภาพ (ปรับเป็นสัก 30-50 รอบเพื่อให้มันเรียนรู้ได้ดีขึ้นหน่อยก็ได้ครับ)
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).numpy()
        rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"⏳ รอบที่ {iteration_count}: ลองพารามิเตอร์ layers={n_layers}, neurons={n_neurons}, lr={lr:.5f} -> RMSE: {rmse:.4f}")
    return rmse

# ==========================================
# ส่วนที่เพิ่มเข้ามา: สร้าง Callback สำหรับหยุดเมื่อถึงเป้าหมาย
# ==========================================
class TargetScoreStopper(EarlyStopper):
    def __init__(self, target_score):
        self.target_score = target_score

    def _criterion(self, result):
        # ถ้าคะแนนที่ดีที่สุด (result.fun) ต่ำกว่าหรือเท่ากับเป้าหมาย ให้หยุดการทำงาน (return True)
        if result.fun <= self.target_score:
            print(f"\n🎉 หยุดการจูนก่อนกำหนด! พบค่า RMSE ({result.fun:.4f}) ซึ่งผ่านเกณฑ์เป้าหมาย ({self.target_score}) แล้ว!")
            return True
        return False

# 3. กำหนดขอบเขตการค้นหา
search_space = [
    Integer(1, 5, name='n_layers'),
    Integer(32, 256, name='n_neurons'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr')
]

# ==========================================
# 4. ตั้งค่าเป้าหมายและจำนวนรอบ
# ==========================================
MAX_CALLS = 50           # กำหนดจำนวนรอบสูงสุดที่ยอมให้หาได้ (เช่น 50 รอบ)
TARGET_RMSE = 0.65       # กำหนดคะแนน RMSE ที่พอใจ (ถ้าตัวเลขต่ำกว่าหรือเท่ากับค่านี้ โปรแกรมจะหยุดทันที)

stopper = TargetScoreStopper(target_score=TARGET_RMSE)

print(f"กำลังเริ่มให้ GBRT จูนพารามิเตอร์... (รอบสูงสุด: {MAX_CALLS}, เป้าหมาย RMSE: <= {TARGET_RMSE})")
# สังเกตว่าเราใส่ callback=[stopper] เข้าไปด้วย
result = forest_minimize(objective, search_space, n_calls=MAX_CALLS, callback=[stopper], random_state=42)

# 5. แสดงผลลัพธ์ที่ดีที่สุดเมื่อจูนเสร็จ
print("\n=== ผลการจูนเสร็จสิ้น ===")
print(f"ค่าพารามิเตอร์ที่ดีที่สุด (Layers, Neurons, LR): {result.x}")
print(f"ค่า RMSE ที่ต่ำที่สุดที่ทำได้: {result.fun:.4f}")