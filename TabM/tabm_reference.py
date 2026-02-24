import torch
import torch.nn as nn
import torch.nn.functional as F


class TabM(nn.Module):
    def __init__(self, d_in, d_out, n_layers=3, d_block=256, k=32):
        super().__init__()
        self.k = k

        # ส่วนกระจาย Input ไปยังสมาชิก k ตัวใน Ensemble
        # d_in คือจำนวนฟีเจอร์ทั้งหมด (Num + Cat)
        self.first_layer = nn.Linear(d_in, d_block * k)

        # Hidden Layers (ใช้หลักการ BatchEnsemble เพื่อความเร็ว)
        self.layers = nn.ModuleList([
            nn.Linear(d_block, d_block) for _ in range(n_layers - 1)
        ])

        # Output Layer สำหรับทำนายผล (Output ต่อ 1 สมาชิก)
        self.head = nn.Linear(d_block, d_out)

    def forward(self, x_num, x_cat=None):
        # รวมข้อมูล Num และ Cat เข้าด้วยกันก่อนเข้าโมเดล
        if x_cat is not None:
            x = torch.cat([x_num, x_cat], dim=1)
        else:
            x = x_num

        # 1. Expand input ไปยัง k members: (Batch, d_in) -> (Batch, k * d_block)
        x = self.first_layer(x)

        # 2. Reshape ให้แยกมิติ k ออกมา: (Batch, k, d_block)
        x = x.view(len(x), self.k, -1)

        # 3. Pass through hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # 4. Final Prediction: (Batch, k, d_out)
        x = self.head(x)
        return x
