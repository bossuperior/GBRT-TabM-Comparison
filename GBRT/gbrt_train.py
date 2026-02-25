import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from gbrt_model import FlexibleMLP
from pathlib import Path

# ==========================================
# 1. ค่าพารามิเตอร์ที่ดีที่สุดจาก GBRT
# ==========================================
BEST_LAYERS = 2
BEST_NEURONS = 35
BEST_LR = 0.008535324065804302
BEST_DROPOUT = 0.03718515424302466
BEST_WEIGHT_DECAY = 1.7294309366607873e-05
BEST_BATCH_SIZE = 64

MAX_EPOCHS = 200  # เทรนสูงสุด 200 รอบ
PATIENCE = 20  # ถ้า Val Loss ไม่ลดลง 20 รอบติด ให้หยุดเทรน (Early Stopping)
# ==========================================

# 2. โหลดข้อมูลทั้งหมด (Train, Val, Test)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "california"

X_train = torch.tensor(np.load(DATA_DIR / "X_num_train.npy")).float()
y_train = torch.tensor(np.load(DATA_DIR / "Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(DATA_DIR / "X_num_val.npy")).float()
y_val = torch.tensor(np.load(DATA_DIR / "Y_val.npy")).float().view(-1, 1)

# ** ไฮไลต์สำคัญ: โหลดชุดข้อสอบจริง (Test Set) มาเตรียมไว้ **
X_test = torch.tensor(np.load(DATA_DIR / "X_num_test.npy")).float()
y_test_np = np.load(DATA_DIR / "Y_test.npy")

# สร้าง DataLoader สำหรับ Batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True)

# 3. สร้างโมเดลตัวจริง
model = FlexibleMLP(n_layers=BEST_LAYERS, n_neurons=BEST_NEURONS, dropout_rate=BEST_DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
criterion = nn.MSELoss()

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = BASE_DIR / "GBRT" / "mlp_gbrt_model.pt"

print(f"🚀 เริ่มเทรน Final Model ด้วยพารามิเตอร์ที่ดีที่สุด...")
print(f"Layers: {BEST_LAYERS}, Neurons: {BEST_NEURONS}, Batch Size: {BEST_BATCH_SIZE}")

# 4. Training Loop แบบเต็มระบบ
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss_accum = 0.0

    # วนลูปย่อยทีละ Batch
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item() * batch_X.size(0)

    avg_train_loss = train_loss_accum / len(train_dataset)

    # วัดผล Validation ทุกรอบ
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:3d}/{MAX_EPOCHS}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # ระบบ Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)  # เซฟน้ำหนักที่ดีที่สุดเก็บไว้
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping ทำงานที่รอบ {epoch + 1}!")
        print(f"โหลดน้ำหนักที่ดีที่สุดกลับมาใช้เพื่อทำข้อสอบจริง...")
        break

# ==========================================
# 5. การสอบครั้งสุดท้าย (Test Evaluation)
# ==========================================
model.load_state_dict(torch.load(best_model_path))  # โหลดน้ำหนักที่ Val Loss ต่ำสุด
model.eval()

with torch.no_grad():
    # ให้โมเดลทำนายข้อสอบ Test Set ที่ไม่เคยเห็นมาก่อนเลยในชีวิต
    test_preds = model(X_test).numpy()

    final_test_rmse = np.sqrt(mean_squared_error(y_test_np, test_preds))
    final_test_r2 = r2_score(y_test_np, test_preds)

print("\n=========================================")
print(f"🏆 ผลลัพธ์สุดท้ายของฝั่ง MLP (บนข้อสอบ TEST SET)")
print(f"RMSE: {final_test_rmse:.4f} (ยิ่งต่ำยิ่งดี)")
print(f"R² Score: {final_test_r2:.4f} (ยิ่งเข้าใกล้ 1.0 ยิ่งดี)")
print("=========================================")

# เซฟผลลัพธ์เป็นไฟล์ txt สำหรับให้กลุ่ม Reporting นำไปใช้ต่อ
results_file = BASE_DIR / "GBRT" / "mlp_gbrt_results.txt"
with open(results_file, "w", encoding="utf-8") as f:
    f.write(f"Model: MLP tuned by GBRT\n")
    f.write(f"Test RMSE: {final_test_rmse:.4f}\n")
    f.write(f"Test R2: {final_test_r2:.4f}\n")
print(f"✅ บันทึกผลลัพธ์ลงในไฟล์ {results_file} เรียบร้อยแล้ว")