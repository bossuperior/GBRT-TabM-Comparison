import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from gbrt_model import FlexibleMLP

# ==========================================
# 1. ใส่ค่าที่ดีที่สุดที่ได้จาก gbrt_tuner.py ลงตรงนี้
# ==========================================
BEST_LAYERS = 3  # สมมติว่า GBRT แนะนำ 3
BEST_NEURONS = 128  # สมมติว่า GBRT แนะนำ 128
BEST_LR = 0.001  # สมมติว่า GBRT แนะนำ 0.001
MAX_EPOCHS = 200  # จำนวนรอบสูงสุดที่จะเทรน
PATIENCE = 20  # จำนวนรอบที่ยอมให้ Val Loss ไม่ลดลงก่อนหยุด (Early Stopping)
# ==========================================

# 2. โหลดข้อมูลครบทั้ง 3 ชุด (Train, Val, Test)
data_dir = "../data/california"  # ปรับ Path ให้ชี้ไปที่โฟลเดอร์ data
X_train = torch.tensor(np.load(f"{data_dir}/X_num_train.npy")).float()
y_train = torch.tensor(np.load(f"{data_dir}/Y_train.npy")).float().view(-1, 1)

X_val = torch.tensor(np.load(f"{data_dir}/X_num_val.npy")).float()
y_val = torch.tensor(np.load(f"{data_dir}/Y_val.npy")).float().view(-1, 1)

# **สำคัญมาก** ต้องใช้ Test Set ในการวัดผลเปรียบเทียบกับ TabM
X_test = torch.tensor(np.load(f"{data_dir}/X_num_test.npy")).float()
y_test_np = np.load(f"{data_dir}/Y_test.npy")  # เก็บเป็น numpy ไว้คำนวณตอนจบ

# 3. สร้างโมเดลและเครื่องมือ
model = FlexibleMLP(n_layers=BEST_LAYERS, n_neurons=BEST_NEURONS)
optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR)
criterion = nn.MSELoss()

# ตัวแปรสำหรับ Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = "best_gbrt_mlp.pt"

print(f"🚀 เริ่มเทรนโมเดล MLP ด้วยพารามิเตอร์ที่ดีที่สุด...")
print(f"Layers: {BEST_LAYERS}, Neurons: {BEST_NEURONS}, LR: {BEST_LR}")

# 4. Training Loop แบบเต็มรูปแบบ
for epoch in range(MAX_EPOCHS):
    # --- Train Mode ---
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # --- Validation Mode ---
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    # พิมพ์ผลทุกๆ 10 Epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # --- Early Stopping Logic ---
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0
        # เซฟน้ำหนักโมเดลที่ดีที่สุดเก็บไว้
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"🛑 Early stopping ทำงานที่รอบ {epoch + 1}! โหลดน้ำหนักที่ดีที่สุดกลับมา...")
        break

# 5. โหลดน้ำหนักที่ดีที่สุดกลับมาเพื่อประเมินผลกับ Test Set
model.load_state_dict(torch.load(best_model_path))
model.eval()

with torch.no_grad():
    # พยากรณ์ข้อมูล Test
    test_preds = model(X_test).numpy()

    # คำนวณ Metrics เชิงวิชาการ
    final_rmse = np.sqrt(mean_squared_error(y_test_np, test_preds))
    final_r2 = r2_score(y_test_np, test_preds)

print("\n=========================================")
print(f"🏆 ผลลัพธ์สุดท้ายของฝั่ง MLP (บน Test Set)")
print(f"RMSE: {final_rmse:.4f} (ยิ่งต่ำยิ่งดี)")
print(f"R² Score: {final_r2:.4f} (ยิ่งเข้าใกล้ 1.0 ยิ่งดี)")
print("=========================================")

# (ตัวเลือก) บันทึกผลลงไฟล์เพื่อส่งให้ทีม Reporting
with open("gbrt_results.txt", "w") as f:
    f.write(f"RMSE,{final_rmse:.4f}\n")
    f.write(f"R2,{final_r2:.4f}\n")