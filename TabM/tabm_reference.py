import torch
import torch.nn as nn
import torch.nn.functional as F


class TabM(nn.Module):
    def __init__(self, d_in, d_out, n_layers=3, d_block=256, k=32, dropout=0.1):
        super().__init__()
        self.k = k

        # Layer แรก: ขยายข้อมูลไปยังสมาชิก k ตัว (BatchEnsemble logic)
        self.first_layer = nn.Linear(d_in, d_block * k)

        # Hidden Layers: ใช้ ModuleList เพื่อสร้างชั้นตามจำนวน n_layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(d_block, d_block))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_block, d_out)

    def forward(self, x):
        # 1. กระจายข้อมูล (Batch, d_in) -> (Batch, k, d_block)
        x = self.first_layer(x)
        x = x.view(len(x), self.k, -1)

        # 2. ประมวลผลผ่าน Hidden Layers พร้อม ReLU และ Dropout
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        # 3. ผลลัพธ์สุดท้ายแยกรายสมาชิก (Batch, k, d_out)
        x = self.head(x)
        return x