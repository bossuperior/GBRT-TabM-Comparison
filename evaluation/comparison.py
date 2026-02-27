import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import tabm
from tabm import EnsembleView
import joblib
import rtdl_num_embeddings
# =====================================================================
# 1. การจัดการ Path และ System Environment
# =====================================================================
# หาตำแหน่ง Root ของโปรเจกต์ (ถอยจาก evaluation/ ไป 1 ชั้น)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ตั้งค่า Device (ใช้ GPU ถ้ามี ถ้าไม่มีใช้ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 2. Import โมเดลจากโครงสร้างโฟลเดอร์จริง
# =====================================================================
try:
    from GBRT.gbrt_model import GBRTModel 
except ImportError as e:
    print(f"❌ Error: ไม่สามารถ Import โมเดลได้ ตรวจสอบชื่อไฟล์และโฟลเดอร์: {e}")
    sys.exit(1)

# =====================================================================
# 3. ฟังก์ชันคำนวณและบันทึกผล
# =====================================================================
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def save_results_and_plot(gbrt_metrics, tabm_metrics, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    metrics_names = ['RMSE', 'MAE', 'R-squared']
    
    # --- บันทึกผลเป็นไฟล์ Text ---
    txt_path = os.path.join(save_dir, "model_comparison_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("🏠 California Housing: Model Performance Comparison\n")
        f.write("="*50 + "\n\n")
        
        for name, g_val, t_val in zip(metrics_names, gbrt_metrics, tabm_metrics):
            f.write(f"Metric: {name}\n")
            f.write(f"  - GBRT (MLP): {g_val:.4f}\n")
            f.write(f"  - TabM:       {t_val:.4f}\n")
            better = "GBRT" if (g_val < t_val if name != 'R-squared' else g_val > t_val) else "TabM"
            f.write(f"  >> Winner:    {better}\n\n")

    # --- สร้างกราฟเปรียบเทียบ ---
    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, gbrt_metrics, width, label='GBRT (MLP)', color='#3498db', edgecolor='black', alpha=0.8)
    rects2 = ax.bar(x + width/2, tabm_metrics, width, label='TabM', color='#e74c3c', edgecolor='black', alpha=0.8)

    ax.set_ylabel('Scores')
    ax.set_title('Performance Comparison: GBRT(MLP) vs TabM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison_plot.png"), dpi=300)
    plt.close()

# =====================================================================
# 4. Main Execution
# =====================================================================
if __name__ == "__main__":
    print(f"🚀 เริ่มต้นกระบวนการ Evaluation (Device: {device})")

    # --- โหลดข้อมูล (Case Sensitive ตามรูปภาพ) ---
    data_dir = os.path.join(PROJECT_ROOT, "data", "california")
    try:
        X_test = np.load(os.path.join(data_dir, "X_num_test.npy"))
        y_test_real = np.load(os.path.join(data_dir, "Y_test.npy")).ravel()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        print(f"✅ โหลดข้อมูลสำเร็จ: {X_test.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: ไม่พบไฟล์ข้อมูล .npy: {e}")
        sys.exit(1)

    print("-> กำลังโหลด GBRT (Hybrid) Model...")
    gbrt_dir = os.path.join(PROJECT_ROOT, "GBRT")
    
    # 1. โหลดพารามิเตอร์ของ MLP
    gbrt_json_path = os.path.join(gbrt_dir, "gbrt_best_params.json")
    try:
        with open(gbrt_json_path, 'r') as f:
            gbrt_params = json.load(f)
        N_LAYERS = gbrt_params["n_layers"]
        N_NEURONS = gbrt_params["n_neurons"]
        DROPOUT_RATE = gbrt_params.get("dropout_rate", 0.1)
    except FileNotFoundError:
        print("⚠️ ไม่พบไฟล์ JSON ของ GBRT ใช้ค่าที่เห็นจาก Error...")
        N_LAYERS, N_NEURONS, DROPOUT_RATE = 2, 504, 0.1

    # 2. โหลดตัวแปลงข้อมูล (GBRT & Encoder)
    gbrt_extractor = joblib.load(os.path.join(gbrt_dir, "gbrt_extractor.joblib"))
    leaf_encoder = joblib.load(os.path.join(gbrt_dir, "leaf_encoder.joblib"))
    
    # 3. แปลงร่างข้อมูล X_test (8 คอลัมน์ -> 769 คอลัมน์)
    print("   -> กำลังแปลงฟีเจอร์ด้วย GBRT Extractor...")
    X_test_encoded = leaf_encoder.transform(gbrt_extractor.apply(X_test)).toarray()
    INPUT_DIM = X_test_encoded.shape[1] # น่าจะได้ 769
    
    # แปลงเป็น Tensor เพื่อรอเข้า MLP
    X_test_gbrt_tensor = torch.tensor(X_test_encoded, dtype=torch.float32).to(device)

    # 4. โหลดตัว MLP (PyTorch)
    gbrt_model = GBRTModel(
        input_dim=INPUT_DIM, 
        n_layers=N_LAYERS, 
        n_neurons=N_NEURONS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    gbrt_path = os.path.join(gbrt_dir, "mlp_gbrt_model.pt")
    gbrt_model.load_state_dict(torch.load(gbrt_path, map_location=device))
    gbrt_model.eval()

    # --- โหลดและตั้งค่า TabM Model ---
    print("-> กำลังโหลด TabM Model...")
    json_path = os.path.join(PROJECT_ROOT, "TabM_R2", "tabm_best_params.json")
    try:
        with open(json_path, 'r') as f:
            best_params = json.load(f)
        BEST_N_BLOCKS = best_params["n_blocks"]
        BEST_D_BLOCK = best_params["d_block"]
        BEST_DROPOUT = best_params["dropout"]
        BEST_N_BINS = best_params["n_bins"]              # <-- เพิ่มบรรทัดนี้
        BEST_D_EMBEDDING = best_params["d_embedding"]    # <-- เพิ่มบรรทัดนี้
    except FileNotFoundError:
        print("⚠️ ไม่พบไฟล์ JSON ของ TabM ใช้ค่า Default")
        BEST_N_BLOCKS, BEST_D_BLOCK, BEST_DROPOUT = 3, 256, 0.1
        BEST_N_BINS, BEST_D_EMBEDDING = 32, 16           # <-- ตั้งค่าเริ่มต้นเผื่อไว้

    K_ENSEMBLE = 32

    
    # ⚠️ สำคัญมาก: ต้องโหลด X_train เพื่อมาคำนวณ bins ให้เหมือนตอนเทรนเป๊ะๆ
    X_train_raw = np.load(os.path.join(data_dir, "X_num_train.npy"))
    X_train_tensor_for_bins = torch.tensor(X_train_raw, dtype=torch.float32)
    bins = rtdl_num_embeddings.compute_bins(X_train_tensor_for_bins, n_bins=BEST_N_BINS)
    
    ple_layer = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        bins, 
        d_embedding=BEST_D_EMBEDDING, 
        activation=torch.nn.GELU(), 
        version="A"
    )
    d_in_expanded = 8 * BEST_D_EMBEDDING

    # --- สร้างโครงสร้างใหม่ที่มี ple_layer และ Flatten ---
    tabm_model = torch.nn.Sequential(
        ple_layer,
        torch.nn.Flatten(1),
        tabm.EnsembleView(k=K_ENSEMBLE),
        tabm.MLPBackboneBatchEnsemble(
            d_in=d_in_expanded,
            n_blocks=BEST_N_BLOCKS,
            d_block=BEST_D_BLOCK,
            dropout=BEST_DROPOUT,
            k=K_ENSEMBLE,
            tabm_init=True,
            scaling_init='normal',
            start_scaling_init_chunks=None,
        ),
        tabm.LinearEnsemble(BEST_D_BLOCK, 1, k=K_ENSEMBLE)
    ).to(device)

    # ⚠️ โหลดไฟล์โมเดลชื่อใหม่ (tabm_model.pt) ตามที่เซฟในไฟล์เทรน
    tabm_path = os.path.join(PROJECT_ROOT, "TabM_R2", "tabm_model.pt")
    tabm_model.load_state_dict(torch.load(tabm_path, map_location=device))
    tabm_model.eval()

    print("🔮 กำลังทำการพยากรณ์...")
    with torch.no_grad():
        # GBRT Prediction (ใช้ข้อมูลที่แปลงร่างเป็น 769 คอลัมน์แล้ว)
        y_pred_gbrt = gbrt_model(X_test_gbrt_tensor).cpu().numpy().flatten()
        
        # TabM Prediction (ใช้ข้อมูลดิบ 8 คอลัมน์)
        y_pred_tabm_raw = tabm_model(X_test_tensor)
        if y_pred_tabm_raw.dim() > 1 and y_pred_tabm_raw.shape[1] > 1:
            y_pred_tabm = y_pred_tabm_raw.mean(dim=1) 
        else:
            y_pred_tabm = y_pred_tabm_raw
        y_pred_tabm = y_pred_tabm.cpu().numpy().flatten()

    # --- สรุปผลและบันทึก ---
    gbrt_res = calculate_metrics(y_test_real, y_pred_gbrt)
    tabm_res = calculate_metrics(y_test_real, y_pred_tabm)

    eval_dir = os.path.join(PROJECT_ROOT, "evaluation")
    save_results_and_plot(gbrt_res, tabm_res, eval_dir)

    print("\n" + "="*30)
    print(f"✨ เสร็จสมบูรณ์!")
    print(f"📊 ผลลัพธ์ถูกบันทึกไว้ใน: {eval_dir}")
    print("="*30)