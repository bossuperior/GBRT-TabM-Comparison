# GBRT-TabM-Comparison

## Team Members

- 6652300044 นายเพียรวัตร ไคร่นุ่นโพธิ์ 
- 6652300109 นางสาวอรุณฉัตร บุญยัง 
- 6652300184 นายสิทธิกร ชิรงค์ 
- 6652300222 นางสาวดวงกมล พูลเกษม 
- 6652300338 นายสวรุจ สองเมือง 
- 6652300346 นางสาวภัทรลดา สุขเสนา 
- 6652300371 นายเฉลิมพล บรรณารรักษ์ 
- 6652300770 นางสาวอารีรัตน์ อู่ทรัพย์ 
- 6652300931 นายคมกฤษณ์ ตังตติยภัทร์  

## Dataset Setup – California (TabM format split)

โปรเจกต์นี้ใช้ California Housing Dataset (TabM-prepared split) เป็น dataset กลางสำหรับทุกโมเดล เพื่อให้การเปรียบเทียบผลลัพธ์ fair และใช้ train/validation/test split เดียวกันทั้งหมด

### การเตรียม Dataset (ทำครั้งแรกครั้งเดียวต่อเครื่อง)
ก่อนเริ่มทำโมเดล ทุกคนในทีมต้องเตรียม dataset ตามขั้นตอนด้านล่างนี้

#### 1) ติดตั้ง dependency ที่จำเป็น
```bash
pip install -r requirements.txt
```
#### 2) ดาวน์โหลดและแตกไฟล์ Dataset
ที่ root ของโปรเจกต์ ให้รันคำสั่ง:
```bash
python scripts/setup_california_data.py
```
สคริปต์นี้จะ:
ดาวน์โหลดไฟล์ california_tabm.zip จาก Google Drive (ถ้ายังไม่มีในเครื่อง)
แตกไฟล์ไปยังโฟลเดอร์ data/california/
ตรวจสอบว่าไฟล์ที่จำเป็นครบถ้วน
ตรวจสอบความถูกต้องของ shape (sanity check)

#### 3) วิธีโหลด Dataset เข้าโมเดล
ให้ใช้ loader ที่เตรียมไว้ใน utils.py

