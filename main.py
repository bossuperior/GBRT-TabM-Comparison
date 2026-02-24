import torch
import torch.nn as nn
import torch.optim as optim
import time # สำหรับคำนวณเวลา
from TabM.tabm_reference import TabM
from utils import get_california_tensors

# ตรวจสอบอุปกรณ์ที่ใช้ (Hardware Check)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [Status] Using Device: {device} ---", flush=True)

# 1. โหลดข้อมูล
print("[Step 1/5] Loading data from utils...", end="", flush=True)
train_set, val_set, test_set, info = get_california_tensors()
X_train, y_train = train_set[0].to(device), train_set[1].to(device)
X_val, y_val = val_set[0].to(device), val_set[1].to(device)
print(" DONE", flush=True)
print(f"    - Dataset Name: {info.get('name', 'California')}")
print(f"    - Training samples: {X_train.shape[0]}")

# 2. ตั้งค่า Hyperparameters
d_in = X_train.shape[1]
d_out = 1
k = 32
epochs = 20
print(f"[Step 2/5] Initializing TabM model (k={k})...", end="", flush=True)
model = TabM(d_in=d_in, d_out=d_out, n_layers=3, d_block=256, k=k).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
print(" DONE", flush=True)

# 3. Training Loop พร้อมรายงานผล
print(f"[Step 3/5] Starting Training for {epochs} epochs...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    outputs = model(X_train)
    y_expanded = y_train.unsqueeze(1).expand(-1, k, -1)
    loss = criterion(outputs, y_expanded)

    loss.backward()
    optimizer.step()

    # แสดงผลทุก Epoch เพื่อให้เห็นความเคลื่อนไหว
    print(f"    > Epoch {epoch + 1:02d}/{epochs} | Loss: {loss.item():.6f}", flush=True)

end_time = time.time()
print(f"[Step 4/5] Training Complete! (Time: {end_time - start_time:.2f}s)")

# 4. การวัดผล (Evaluation)
print("[Step 5/5] Running Validation...", end="", flush=True)
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    final_pred = val_outputs.mean(dim=1)
    val_loss = criterion(final_pred, y_val)
    print(" DONE", flush=True)

print("-" * 40)
print(f"TABM FINAL RESULTS")
print(f"Validation MSE: {val_loss.item():.6f}")
print("-" * 40)