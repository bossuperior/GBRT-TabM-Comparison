import torch
# ดึง Class TabM มาจากไฟล์ที่เราดาวน์โหลดมา
from tabm_reference import TabM

# 1. ตั้งค่าพื้นฐาน (ตามโจทย์ที่คุณ bossk วางไว้)
d_in = 10          # จำนวนฟีเจอร์ของข้อมูล (Input)
d_out = 1          # ผลลัพธ์ (Regression)
batch_size = 256

# 2. สร้างโมเดล TabM (ที่เป็น MLP Ensemble)
model = TabM(
    d_in=d_in,
    d_out=d_out,
    n_layers=3,        # จำนวนเลเยอร์ Linear
    d_block=256,       # จำนวน Neuron ในแต่ละเลเยอร์
    k=32,              # จำนวนสมาชิกใน Ensemble (นี่คือจุดแข็งของ TabM)
    arch_type='tabm'   # ใช้สถาปัตยกรรม TabM แบบเต็มรูปแบบ
)

# 3. จำลองข้อมูลสำหรับทดสอบ (Preprocessing)
x_num = torch.randn(batch_size, d_in) # ข้อมูลตัวเลข

# 4. การทำนาย (Forward Pass)
# TabM จะคืนค่าออกมาเป็น (batch_size, k, d_out)
y_pred = model(x_num)

print(f"ขนาดของผลลัพธ์: {y_pred.shape}")
# ผลลัพธ์ที่ได้คือการทำนายจาก 32 โมเดลย่อยพร้อมกัน