# TESA TOP GUN 2026 · KMITL CHUMPHON
## Edge AI Track
ตรวจคุณภาพเมล็ดกาแฟโรบัสต้า บน Raspberry Pi
3 วัน: Classification → Detection / Segmentation → Integration
Jirayu Petchhan, Ph.D.
Dept. of Computer Engineering, School of Engineering, KMITL

*(Image: Coffee beans scattered on a wooden surface)*

---

## ภาพรวมการแข่งขัน
# สาม track ส่งข้อมูลต่อกันจนได้ราคาเมล็ดกาแฟ

* **TRACK 1: Hardware**
  วัดความชื้นของเมล็ดกาแฟด้วยเซนเซอร์
  ผลลัพธ์ → ค่าความชื้น (%)

* **TRACK 2: Edge AI (เราดูแล)**
  ตรวจคุณภาพด้วยภาพ แยกเมล็ดดีกับเมล็ดบกพร่อง บน Raspberry Pi
  ผลลัพธ์ → จำนวนเมล็ดต่อคลาส · สัดส่วนเมล็ดเสีย

* **TRACK 3: Agent-based Bidding**
  Agent ตั้งราคาจากความชื้นและคุณภาพ
  ผลลัพธ์ → ราคาเสนอซื้อ

---

## ตารางเรียน 3 วัน
แต่ละวันขยับความยากขึ้นหนึ่งขั้น

| | Day 1 | Day 2 | Day 3 |
| :--- | :--- | :--- | :--- |
| **งาน** | Classification | Detection / Segmentation | Integration |
| **Train** | จัดเรียง 3x3 / วงกลม / สี่เหลี่ยม · 1 คลาส ต่อภาพ · 2-3 ภาพต่อคลาส | ชุดเดียวกับ Day 1 แต่มีป้ายกำกับ bbox/mask เพิ่มเข้ามา | ทีละเม็ด 3x3 / สามเหลี่ยม / สี่เหลี่ยม · คละ 7 คลาสในภาพเดียว |
| **Test** | วางกองปกติ · 1 ภาพต่อคลาส | วางกองปกติ · 1 ภาพต่อคลาส | วางกองปกติ · คละ 3-5 คลาส หรือครบทุก คลาส |
| **ส่งงาน** | source code + weight (ready to inference แยกคลาส) | source code + weight (ready to draw bbox / mask) | ระบบเต็มบน Raspberry Pi |

---

## ข้อมูลตัวอย่าง
# 7 คลาส: เมล็ดดี 1 คลาส และเมล็ดบกพร่อง 6 คลาส

* **เมล็ดคุณภาพดี (Good)**
  *(Image: Good coffee beans)*
* **เมล็ดหูช้าง (Shell)**
  *(Image: Shell coffee beans)*
* **เมล็ดดำ (Full Black)**
  *(Image: Full Black coffee beans)*
* **เศษกะลา / เปลือกเซอร์รี่แห้ง (Hull / Husk)**
  *(Image: Hull / Husk)*
* **เมล็ดแตกหักจากการสี (Broken / Chipped / Cut)**
  *(Image: Broken / Chipped / Cut beans)*
* **เมล็ดกะลา (Parchment / Pergamino)**
  *(Image: Parchment / Pergamino beans)*
* **เมล็ดดิบ / เมล็ดอ่อน (Immature / Unripe)**
  *(Image: Immature / Unripe beans)*

ภาพตัวอย่างคุณลักษณะของเมล็ดกาแฟที่ใช้ในการแข่งขัน พร้อมป้ายกำกับชื่อประเภท (คลาส)
*** หมายเหตุ: วันเรียนและวันแข่งขันจริงมีการปรับเปลี่ยนชุดข้อมูลทั้งหมด โดยที่แสดงอยู่ ณ ที่นี้เป็นเพียงตัวอย่างคร่าว ๆ เท่านั้น**

---

## ข้อมูลตัวอย่าง
สิ่งที่โมเดลต้องมองให้ออก และคู่ที่สับสนง่าย

| คลาส | ลักษณะที่สังเกตได้จากภาพ | มักสับสนกับ |
| :--- | :--- | :--- |
| **Good · เมล็ดดี** | สีเขียวอมเทา ผิวเรียบ ทรงรีสมบูรณ์ ร่องกลางปิดเรียบ | Immature, Parchment |
| **Shell · หูช้าง** | เปลือกโค้งกลวง เห็นด้านในเป็นแอ่ง ทรงคล้ายใบหู | Broken |
| **Full Black · เมล็ดดำ** | ดำหรือเทาเข้มเกือบทั้งเมล็ด ผิวย่น บางเม็ดมีรู | Hull / Husk |
| **Hull / Husk** | น้ำตาลเข้ม มีเปลือกเชอร์รี่แห้งหุ้ม บางเม็ดแตกเห็นเนื้อในสีครีม | Full Black |
| **Broken · แตกหัก** | ขอบแหว่ง ชิ้นไม่ครบเมล็ด ร่องกลางผิดรูป | Shell, Immature |
| **Parchment · เมล็ดกะลา** | เหลืองครีมสว่าง ผิวเรียบเงา ยังมีกะลาบางหุ้มอยู่ | Good |
| **Immature · ดิบ / อ่อน** | สีซีดอมเหลือง ร่องกลางเปิดกว้างชัด ขอบย่น | Good, Broken |

---

## Classification
เรียนจากภาพเมล็ดกาแฟวางสวยงาม แล้วทำนายภาพที่วางกองตามธรรมชาติ ด้วยโมเดล AI

---

## DAY 1 · ข้อมูล
Train เรียนเป็นรูปแบบ แต่ Test วางเป็นกอง

**Train · 2–3 ภาพต่อคลาส**
*(Image: Examples of 3x3, circle, and rectangle arrangements)*
1 ภาพ = 1 คลาส · จัดวางเป็น 3x3, วงกลม, สี่เหลี่ยม หรือรูปแบบอื่น เม็ดไม่ชนกัน

**Test · 1 ภาพต่อคลาส**
*(Image: Coffee beans piled naturally)*
วางกองตามธรรมชาติ ชิดและช้อนกันบางส่วน

ข้อมูลน้อยมาก: 7 คลาส × 2–3 ภาพ ≈ 14–21 ภาพ — ต้องเปลี่ยนจาก “ภาพ” เป็น “เม็ด” เพื่อให้ได้ตัวอย่างหลายร้อยชิ้น

---

## Detection & Segmentation
จาก “ภาพนี้คือคลาสอะไร” สู่ “แต่ละเม็ดอยู่ตรงไหน และเป็นคลาสอะไร” ด้วยโมเดลเฉพาะงานมากขึ้นและสามารถอนุมานบน อุปกรณ์ edge อย่าง Raspberry Pi

---

## DAY 2 · ประเภทงาน
สีระดับของการ “มองเห็น” เมล็ดกาแฟ

* **Classification**
  1 label ต่อภาพ · Day 1
  *(Image: Classification example)*
* **Detection**
  กรอบ + คลาสต่อเม็ด · นับได้
  *(Image: Detection example with bounding boxes)*
* **Semantic segmentation**
  mask ต่อคลาส · เม็ดติดกันรวมเป็นก้อน
  *(Image: Semantic segmentation example)*
* **Instance segmentation**
  mask แยกทุกเม็ด · เหมาะกับกอง
  *(Image: Instance segmentation example)*

---

## แต่ละโมเดลกิน RAM และเวลาเท่าไร บน Pi 5 และ Pi 4

| โมเดล - รูปแบบ | Params | GFLOPs / ภาพ | RAM สูงสุด | Pi 5 (วัดจริง) | Pi 4 (ประมาณ) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| YOLO26n-seg · NCNN 640 (ไม่โหลด PyTorch) | 3.1 M | 10.4 | 0.15-0.3 GB | 0.26 s | ~0.85 s |
| YOLO26n-seg · NCNN 320 | 3.1 M | 2.6 | 0.1-0.2 GB | 0.08 s | ~0.25 s |
| YOLO26n-seg · ผ่าน Ultralytics + PyTorch | 3.1 M | 10.4 | 0.5-0.8 GB | 0.26-0.45 s | ~0.85-1.0 s |
| YOLOv8n / 11n-seg · NCNN 640 | 3.4 / 2.9 M | 12.0 / 9.9 | 0.15-0.3 GB | 1.28 / 0.27 s | ~0.9 s |
| U-Net · INT8 ONNX Runtime | 6.6 M | ~27 (512) | 0.2-0.35 GB | 0.26 s | ~0.7-1.0 s |
| U-Net · PyTorch | 6.6 M | ~27 (51) | 0.5-0.9 GB | 0.97 s | ~2 s |
| FastSAM-s · ตัดทั้งภาพ 1024 | 11.8 M | ~110 | 0.9-1.5 GB | 1.7 s | ~4-6 s |
| MobileSAM / EdgeSAM · กรอบทีละเม็ด | 10.1 / 9.6 M | ~38 / ~22 | 0.7-1.2 GB | 0.3 s / เม็ด | ~0.7-1 s / เม็ด |
| SAM ViT-B · กรอบทีละเม็ด | 93.7 M | ~490 | 1.8-3 GB | 1 s / เม็ด | ~2-3 s / เม็ด |
| LaMa · inpaint 1 ภาพ | 51 M | ~ | 1.5-3 GB | 35 s | ~80-100 s |

---

## Pi 4 / Pi 5 · รุ่นไหนพอ (ประมาณ)
# โมเดลที่ใช้แข่งขันได้ทุกรุ่น · อย่างน้อยควรมี RAM 4 GB ขึ้นไป

| | Raspberry Pi 4 · Cortex-A72 1.5–1.8 GHz | | | | Raspberry Pi 5 · Cortex-A76 2.4 GHz | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAM** | **1 GB** | **2 GB** | **4 GB** | **8 GB** | **1 GB** | **2 GB** | **4 GB** | **8–16 GB** |
| **YOLO26n-seg NCNN / ONNX (ไม่โหลด PyTorch)**<br>RAM 0.15–0.3 GB | พอ* | พอ | พอ | พอ | พอ* | พอ | พอ | พอ |
| **YOLO-seg ผ่าน Ultralytics + PyTorch**<br>RAM 0.5–0.8 GB | เสี่ยง | พอ | พอ | พอ | เสี่ยง | พอ | พอ | พอ |
| **U-Net INT8 · ONNX Runtime**<br>RAM 0.2–0.35 GB | พอ* | พอ | พอ | พอ | พอ* | พอ | พอ | พอ |
| **U-Net · PyTorch**<br>RAM 0.5–0.9 GB | เสี่ยง | พอ | พอ | พอ | เสี่ยง | พอ | พอ | พอ |
| **FastSAM / MobileSAM / EdgeSAM**<br>RAM 0.7–1.5 GB | ไม่พอ | เสี่ยง | พอ · ช้า | พอ · ช้า | ไม่พอ | เสี่ยง | พอ | พอ |
| **SAM ViT-B / LaMa**<br>RAM 1.5–3 GB | ไม่พอ | ไม่พอ | เสี่ยง | พอ · ช้ามาก | ไม่พอ | ไม่พอ | เสี่ยง | พอ |
| **เวลา YOLO26n NCNN 640 ทั้งชิ้นตอน** | ~0.85 s | ~0.85 s | ~0.85 s | ~0.85 s | 0.26 s | 0.26 s | 0.26 s | 0.26 s |

**พอ\*** = ต้องใช้ Pi OS Lite (ไม่มี desktop เหลือ RAM ~750 MB) · **เสี่ยง** = รับได้ถ้าปิด desktop + เปิด swap แต่อาจช้า / ค้าง · train บนโน๊ตบุ๊ก / Colab เสมอ

*Edge AI Track Pre-Camp Preview · TESA Top Gun 2026*

---

## Integration Day
รวมทุกอย่างจากสองวันแรก ตรวจสอบเมล็ดที่คละคลาสผ่านอุปกรณ์ edge อย่าง Raspberry Pi

---

## DAY 3 · โจทย์
Train ทีละเม็ดคละคลาส แล้วทดสอบกับกองคละคลาส

**Train · 2–3 ภาพต่อรูปแบบ**
*(Images: Beans arranged individually in 3x3, triangle, rectangle shapes, mixed classes)*
ทีละเม็ด จัดเป็น 3x3, สามเหลี่ยม, สี่เหลี่ยม · คละทั้ง 7 คลาสในภาพเดียว

**Test**
*(Image: A natural pile of mixed coffee beans)*
วางกองปกติ คละ 3–5 คลาส หรือครบทุกคลาส

คำใบ: หน้าตาของ test จะคล้ายที่เจอใน Day 1–2 — วางกองตามธรรมชาติ เพียงแต่คราวนี้คละคลาสกัน