# เก็บข้อมูลจาก PuTTY และสอบเทียบ

**ทางเลือกที่ไม่ต้องกดปุ่ม:** อัปโหลด `method1_test.ino` รุ่นล่าสุด เปิด PuTTY พร้อม logging แล้วพิมพ์ `run` + Enter โปรแกรมจะส่ง CAL_DATA และค่า capacitance 15 batches อัตโนมัติ รอ `RUN_DONE,15` แล้วทำขั้นตอน import และ fit ข้อ 3–4 ด้านล่างได้เลย ไม่ต้องใช้ c/q หรือกดสวิตช์ ใช้ `stop` + Enter เพื่อยกเลิกและเริ่มใหม่ด้วย log ไฟล์ใหม่ ดู [คู่มือโค้ดเทส](../method1_test/README.md)

รุ่นนี้คง coefficient ที่ผู้ใช้ใส่ไว้ใน `method1_competition.ino` ไม่ใช้ `calibration_model.h` ที่อยู่ข้างกัน คำนวณเวลาเหมือนกันทั้งโหมดเก็บข้อมูลและโหมดแข่งขัน เปลี่ยนโหมดได้โดยไม่คอมไพล์ใหม่

## 1. อัปโหลดและเปิด PuTTY

Verify/Upload โค้ดแข่งรุ่นนี้ก่อน ปิด Arduino Serial Monitor และ Python capture แล้วเปิด PuTTY: Serial, COM ของบอร์ด, 9600 baud, 8 data bits, 1 stop bit, parity None, flow control None

ตั้ง Session → Logging → All session output เลือก log ใหม่สำหรับ C แต่ละตัว เช่น `putty_5pf.log` ใน `Track1/firmware/method1_test/data` ต้องไม่ append หลายค่า C ลงไฟล์เดียว ใช้ชื่อใหม่เมื่อทดสอบซ้ำ เก็บสาย ตำแหน่งเสียบ และ potentiometer คงเดิม

## 2. กดวัด C ค่าเดียว 15 ครั้ง

พิมพ์ `c` ในหน้าต่าง PuTTY (ไม่จำเป็นต้อง Enter) จะเห็น:

```text
CAL_MODE,on
CAL_HEADER,run,median_us,reference_mv,cpu_mhz,status
```

กดสวิตช์จริง 15 ครั้ง รอให้ข้อความออกครบก่อนกดครั้งต่อไป หนึ่งครั้งกดคือหนึ่ง batch ของ 64 readings แต่ส่งออกเฉพาะมัธยฐานเพื่อใช้สอบเทียบ ไม่ใช่ข้อมูลดิบทั้ง 64 แถว ตัวอย่าง:

```text
TGR-55
Capacitance = 10.00 pF

CAL_DATA,1,8.606250,900,240,ok
```

ค่าในตัวอย่างเป็นรูปแบบเท่านั้น ลำดับฟิลด์คือ run, median_us, reference_mv, cpu_mhz, status การ fit ใช้ **median_us** ไม่ใช้ค่า pF ที่ผ่านสมการเดิม จึงสอบเทียบใหม่ได้แม้สมการเก่าแสดงค่าคลาดเคลื่อน ถ้าสถานะไม่ใช่ ok จะเก็บบันทึกแต่ไม่อนุญาตให้นำไป fit

ครบ 15 ครั้ง พิมพ์ `q` เพื่อปิด logging (ไม่มีข้อความตอบกลับ) ปิด PuTTY เพื่อจบไฟล์ ถอด USB เปลี่ยน C ใช้รูเดิม เปิด session พร้อม log ใหม่ แล้วทำซ้ำกับ 10, 15 และ 20 pF

## 3. นำเข้า log เป็น CSV

รันจากโฟลเดอร์หลัก repository เช่น:

```powershell
.venv/Scripts/python.exe Track1/firmware/method1_competition/putty_calibration.py import --log Track1/firmware/method1_test/data/putty_5pf.log --reference 5 --setup final_layout_01
```

ทำซ้ำสำหรับ 10/15/20 โดยแก้ชื่อ log และ reference แต่ใช้ setup เดียวกันสำหรับ wiring ชุดเดียว สคริปต์สร้าง CSV และสำเนา log ใน `Track1/firmware/method1_test/data/final_layout_01/` ชื่อไฟล์มี timestamp ไม่ทับของเก่า ค่า reference เป็น pF ใช้ค่าที่วัดด้วยเครื่องอ้างอิงถ้ามี ไม่เช่นนั้นเป็นค่าตามป้ายและมี tolerance

ถ้า log มีแต่ Capacitance ไม่มี CAL_DATA จะ fit เวลาไม่ได้ ต้องเก็บใหม่ในโหมด c สคริปต์จะปฏิเสธ run ขาด/ซ้ำ/เริ่มใหม่ในไฟล์เดียว

## 4. คำนวณ calibration

ส่งชื่อ CSV ที่นำเข้าแล้วทั้งสี่ไฟล์อย่างชัดเจน ไม่ใช้ wildcard ปนไฟล์รอบเก่า:

```powershell
.venv/Scripts/python.exe Track1/firmware/method1_competition/putty_calibration.py fit "path/to/5pf.csv" "path/to/10pf.csv" "path/to/15pf.csv" "path/to/20pf.csv"
```

แทน path ตัวอย่างด้วยชื่อจริงที่สคริปต์พิมพ์ออกมา ต้องมี 15 batches ต่อไฟล์และ setup เดียวกัน สร้างเส้นตรงจาก batches 1–10 และตรวจ batches 11–15 แยกไว้ ไม่ใช้ชุดตรวจ fit ซ้ำ แสดงค่า SLOPE/INTERCEPT และบันทึกผลพร้อม hash ต้นทางไว้ใน data/setup เดียวกัน

สคริปต์ไม่ทับ firmware อัตโนมัติ ให้คัดลอกสอง coefficient ไปแก้ **ตัวที่ไม่ถูก comment ใน namespace Method1Calibration ด้านบน `.ino`** แล้ว Verify/Upload และตรวจซ้ำบนบอร์ดจริง ค่าเก่าและไฟล์ header ที่ไม่ได้ include ไม่มีผลต่อโปรแกรม

## 5. กลับโหมดแข่งขัน

พิมพ์ `q` หรือรีเซ็ตบอร์ด โหมดเริ่มต้นจะพิมพ์เฉพาะทีมและ Capacitance พร้อมบรรทัดว่าง ทดสอบกดปุ่มว่าไม่มี CAL_DATA ก่อนส่งกรรมการ คำสั่ง Serial ไม่ได้ใช้เริ่มการวัด

ข้อมูลชุดตรวจใน session เดียวและ C ตามป้ายไม่รับรองความแม่นยำกับ C สุ่ม ผล Error/RMSE ในรายงานใช้ pF คูณ 100 ตามสูตรที่ผู้ใช้ให้ ไม่หารด้วยค่า reference และยังไม่ปัดผลเป็นสองตำแหน่งก่อนประเมิน
