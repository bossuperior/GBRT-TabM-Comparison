import torch
import torch.nn as nn
from tabm import EnsembleView, make_tabm_backbone, LinearEnsemble


class TabM_California(nn.Module):
    def __init__(self, num_features=8, d_block=256, k=32, num_layers=3, dropout_rate=0.1):
        """
        num_features: จำนวนฟีเจอร์ต้นทาง (California Housing = 8)
        d_block: ขนาดความกว้างของสมองกลในแต่ละเลเยอร์ (ตามเปเปอร์มักใช้ 256 หรือ 512)
        k: จำนวนร่างแยก Ensemble (TabM แนะนำที่ 32)
        """
        super().__init__()

        # ==========================================
        # 1. ส่วนแปลงข้อมูล (handle_input)
        # ==========================================
        # แปลงข้อมูล 8 คอลัมน์ ให้กลายเป็น Vector ขนาด d_block (เช่น 256)
        # เพื่อให้ TabM มีพื้นที่ในการเรียนรู้ความซับซ้อนได้ดีขึ้น
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, d_block),
            nn.ReLU(),
            nn.BatchNorm1d(d_block)  # ใส่ BatchNorm ช่วยให้โมเดลนิ่งและเทรนเร็วขึ้น
        )

        # ==========================================
        # 2. ส่วนประกอบหลักของ TabM
        # ==========================================
        # 2.1 ขยายข้อมูล 1 ก้อน ให้กลายเป็น k ก้อน (สำหรับ 32 โมเดลย่อย)
        self.ensemble_view = EnsembleView(k=k, d_in=d_block)

        # 2.2 กระดูกสันหลัง (Backbone) ที่ใช้พารามิเตอร์ร่วมกัน
        self.backbone = make_tabm_backbone(
            d_in=d_block,
            d_out=d_block,
            d_block=d_block,
            n_layers=num_layers,
            k=k,
            dropout=dropout_rate
        )

        # 2.3 หัวทำนายผล (Prediction Head) ทายราคาบ้าน 1 ค่า
        self.output = LinearEnsemble(
            k=k,
            in_features=d_block,
            out_features=1
        )

    def forward(self, x):
        # x เริ่มต้นคือข้อมูลดิบ -> Shape: (Batch_size, 8)

        # --- ขั้นตอนที่ 1: handle_input ---
        x = self.feature_embedding(x)  # -> Shape: (Batch_size, d_block)

        # --- ขั้นตอนที่ 2: แกนหลักของ TabM ---
        x = self.ensemble_view(x)  # -> Shape: (Batch_size, k, d_block)
        x = self.backbone(x)  # -> Shape: (Batch_size, k, d_block)
        x = self.output(x)  # -> Shape: (Batch_size, k, 1)

        return x