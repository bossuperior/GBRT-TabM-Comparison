import torch
from .tabm_reference import TabM


def create_tabm_model(d_in, d_out=1, k=32):
    """
    ฟังก์ชันสำหรับสร้างโมเดล TabM พร้อมตั้งค่าเริ่มต้น
    d_in: จำนวนฟีเจอร์ทั้งหมดที่ผ่านการ Preprocess แล้ว
    """
    model = TabM(
        d_in=d_in,
        d_out=d_out,
        n_layers=3,
        d_block=256,
        k=k
    )
    return model


def compute_tabm_loss(outputs, targets, k, criterion):
    """
    คำนวณ Loss แบบ Ensemble ตามหลักการของ TabM
    outputs: (Batch, k, d_out)
    targets: (Batch, d_out)
    """
    # ขยาย targets ให้มีมิติ k เท่ากับ outputs
    targets_expanded = targets.unsqueeze(1).expand(-1, k, -1)
    return criterion(outputs, targets_expanded)
