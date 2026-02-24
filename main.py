import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from TabM.tabm_reference import TabM
from utils import get_california_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [Status] Using Device: {device} ---", flush=True)

torch.manual_seed(42)

# 1. โหลดข้อมูลจริง (California Housing) และส่งไปยัง GPU
train_set, val_set, test_set, info = get_california_tensors()

# ย้าย Tensor ไปยัง device (cuda) ทันทีหลังโหลด
X_train, y_train = train_set[0].to(device), train_set[1].to(device)
X_val, y_val = val_set[0].to(device), val_set[1].to(device)

# 2. ตั้งค่าตามมาตรฐาน TabM Paper
d_in = X_train.shape[1]
k = 32  # จำนวน Ensemble

# สร้างโมเดลแล้วส่งไปยัง GPU ด้วย .to(device)
model = TabM(d_in=d_in, d_out=1, n_layers=3, d_block=256, k=k, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Training Loop (100 Epochs เพื่อความแม่นยำสูงสุด)
print(f"Training TabM (Yandex Style) on California Dataset via {device}...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)

    # ขยาย Label ให้เท่ากับจำนวน Ensemble (k)
    # เนื่องจาก y_train อยู่บน GPU แล้ว y_expanded จะอยู่บน GPU โดยอัตโนมัติ
    y_expanded = y_train.unsqueeze(1).expand(-1, k, -1)
    loss = criterion(outputs, y_expanded)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:03d}/100 | Loss: {loss.item():.4f}", flush=True)

# 4. การวัดผลสุดท้าย
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    final_pred = val_outputs.mean(dim=1)
    val_mse = criterion(final_pred, y_val)

print(f"\nFINAL TABM VALIDATION MSE: {val_mse.item():.6f}")