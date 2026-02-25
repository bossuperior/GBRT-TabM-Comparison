from skopt import forest_minimize
from skopt.space import Real, Integer
import numpy as np
from sklearn.metrics import mean_squared_error
from gbrt_model import FlexibleMLP
import torch
import torch.nn as nn

X_train = np.load("data/california/X_num_train.npy")
y_train = np.load("data/california/Y_train.npy")
X_val = np.load("data/california/X_num_val.npy")
y_val = np.load("data/california/Y_val.npy")

# --- แปลงข้อมูลเป็น Tensor เอาไว้ล่วงหน้าเพื่อให้ทำงานเร็วขึ้น ---
X_train_t = torch.tensor(X_train).float()
y_train_t = torch.tensor(y_train).float().view(-1, 1) # ต้องแปลงเป็น column vector
X_val_t = torch.tensor(X_val).float()
# -----------------------------------------------------------

# 2. ฟังก์ชันเป้าหมายที่ GBRT จะพยายามทำให้ค่าต่ำที่สุด
def objective(params):
    n_layers, n_neurons, lr = params

    model = FlexibleMLP(n_layers=n_layers, n_neurons=n_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    # 2. เพิ่มกระบวนการเทรน (Training Loop) ตรงนี้!
    for epoch in range(20):  # ลองปรับเป็น 20 รอบเพื่อให้โมเดลพอเห็นภาพ
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # คำนวณ Error เพื่อส่งกลับไปให้ GBRT
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val).float()).numpy()
        rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"ลองพารามิเตอร์: layers={n_layers}, neurons={n_neurons}, lr={lr:.5f} -> RMSE: {rmse:.4f}")
    return rmse  # GBRT จะพยายามหาพารามิเตอร์ที่ทำให้ค่านี้ต่ำสุด


# 3. กำหนดขอบเขตการค้นหา
search_space = [
    Integer(1, 5, name='n_layers'),
    Integer(32, 256, name='n_neurons'),
    Real(1e-4, 1e-2, prior='log-uniform', name='lr')
]

# 4. เริ่มจูนด้วย GBRT
print("กำลังเริ่มให้ GBRT จูนพารามิเตอร์... (อาจใช้เวลาสักครู่)")
result = forest_minimize(objective, search_space, n_calls=20, random_state=42) #

# 3. แสดงผลลัพธ์ที่ดีที่สุดเมื่อจูนเสร็จ
print("\n=== ผลการจูนเสร็จสิ้น ===")
print(f"ค่าพารามิเตอร์ที่ดีที่สุด (Layers, Neurons, LR): {result.x}")
print(f"ค่า RMSE ที่ต่ำที่สุดที่ทำได้: {result.fun:.4f}")