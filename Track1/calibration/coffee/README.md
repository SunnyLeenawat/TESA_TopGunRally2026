# เก็บข้อมูลและ calibration กาแฟ

อ่าน [แผน Day 2 และวงจร](../../Planning_Day2.md) ก่อน ไฟล์ Day 1 เดิมไม่ถูกแก้ ใช้ Python environment เดิมที่มี pyserial 3.5 (ไม่ต้องใช้ NumPy สำหรับสคริปต์นี้)

## 1. Upload โค้ดเทส

เปิด `Track1/firmware/coffee_test/coffee_test.ino` เลือก DOIT ESP32 DEVKIT V1, ESP32 core 3.3.0 แล้ว Upload ไม่มี library เพิ่ม หากแก้ canonical headers ให้รันก่อน:

```powershell
.venv/Scripts/python.exe Track1/firmware/coffee_common/sync_headers.py
```

Serial 9600: `r` + Enter ตรวจ reference ใกล้ 900 mV; `run` + Enter วัด 15 batches; `m` หนึ่ง batch; `stop` ยกเลิก ห้ามเปลี่ยน/จับ sensor ระหว่างวัด

## 2. เก็บกล่องเปล่า

ปิด PuTTY/Serial Monitor แล้วรันจากราก repository:

```powershell
.venv/Scripts/python.exe Track1/calibration/coffee/coffee_data.py capture --port COM3 --setup coffee_layout01 --sample empty --fill empty01
```

เก็บอย่างน้อย empty01 และ empty02 ก่อนชุดกาแฟ โดยไม่ย้ายสาย ค่า baseline หลังทดสอบเอาไว้ดู drift ไม่ปนเข้า baseline ฝึกโดยไม่ได้ตั้งใจ

## 3. เก็บแต่ละ reference

| sample | ความชื้นเริ่มต้นที่สคริปต์ใช้ | มวลถุงที่ผู้ใช้แจ้ง |
| --- | ---: | ---: |
| A | 11.4% | 35.74 g |
| B | 14.4% | 35.10 g |
| C | 6.6% | 70.29 g |

ใช้ระดับเติม/การเคาะเดียวกัน ไม่เททั้งถุงเพียงเพราะ sample มีปริมาณไม่เท่ากัน ตัวอย่าง sample A refill01:

```powershell
.venv/Scripts/python.exe Track1/calibration/coffee/coffee_data.py capture --port COM3 --setup coffee_layout01 --sample A --fill refill01
```

สคริปต์ส่ง run ให้เองและบันทึก 15 batches หนึ่ง refill ไม่ต้องกดปุ่ม หลังเสร็จให้เทออกและเติมใหม่จึงเปลี่ยนเป็น refill02 ทำจนถึง refill05 ของแต่ละ A/B/C ใช้ `--moisture 11.4` หากต้องการกำหนดค่าอ้างอิงชัดเจน/มีค่าที่ผู้จัดแก้ไข ห้ามใส่ค่าที่เดาจากเครื่องเราเองเป็น reference

ไฟล์อยู่ `Track1/calibration/coffee/data/coffee_layout01/` มี log และ CSV หนึ่งแถวต่อ batch (median จาก 64 readings) CSV บันทึก sample, fill, moisture label, setup, reference mV, MAD และ status ไฟล์ดิบไม่ถูกทับ หากผิดพลาดยังมี partial log ต้องเก็บใหม่ ห้ามใส่ sample คนละค่าลง run เดียว

## ทางเลือก: เก็บผ่าน PuTTY

เปิด 9600 baud, 8N1, flow control None ตั้ง All session output เป็นไฟล์ใหม่ แล้วพิมพ์ run + Enter รอ RUN_DONE,15 และปิด PuTTY จากนั้น:

```powershell
.venv/Scripts/python.exe Track1/calibration/coffee/coffee_data.py import --log "C:/path/A_refill01.log" --setup coffee_layout01 --sample A --fill refill01
```

สำหรับกล่องเปล่าใช้ `--sample empty` ไม่ต้องใส่ moisture ทั้ง capture และ PuTTY ต้องใช้ firmware coffee_test เดียวกัน อย่าเปิดสองโปรแกรมจับ COM port พร้อมกัน

## 4. Fit และตรวจ validation

ใช้ไฟล์ empty01/02 เป็น baseline, refill01–03 ของ A/B/C เป็น train, refill04–05 เป็น validation ต้องระบุชื่อไฟล์จริง (ตัวอย่าง path ด้านล่างต้องแทนก่อนรัน):

```powershell
.venv/Scripts/python.exe Track1/calibration/coffee/coffee_data.py fit --empty "empty01.csv" "empty02.csv" --train "A_refill01.csv" "B_refill01.csv" "C_refill01.csv" --validation "A_refill04.csv" "B_refill04.csv" "C_refill04.csv"
```

เพิ่มไฟล์ train/validation ที่เหลือต่อท้าย flag นั้นได้ โดยใช้ path จริงใน data ใช้ sample+fill ID ไม่ซ้ำ และ setup เดียวกัน สคริปต์ปฏิเสธไฟล์ invalid, metadata ปน, ไม่มี reference ครบ 3 ระดับ หรือ refillเดียวอยู่ทั้ง train/validation

ผลอยู่ `results/<timestamp>/calibration.json` และ `moisture_calibration.candidate.h` โมเดลเป็น **M = a*(median_us - empty_us)+b** ไม่ใช่ M จาก pF ของ Day 1 รายงาน R² ฝึกและ RMSE/MAE validation ในหน่วย percentage points ไม่รับรอง absolute accuracy จาก R² อย่างเดียว

## 5. เตรียมโค้ดแข่ง

1. ตรวจ bias แต่ละถุงและแถว validation ที่ in_feature_range=false ถ้าแย่ให้แก้สาเหตุและเก็บใหม่ ไม่ทำให้ตัวเลขตรงด้วยการ clamp
2. คัดลอก candidate header ไปแทน `Track1/firmware/coffee_common/moisture_calibration.h` แล้วเปลี่ยน READY เป็น true หลังยอมรับผล
3. รัน sync_headers.py เพื่อให้ test/competition ใช้ค่าเดียวกัน จากนั้น Verify/Upload `coffee_competition.ino`
4. เปิด PuTTY กดปุ่มจริง 5 ครั้งต่อ known sample ตรวจรูปแบบทีม/Moisture Level/บรรทัดว่าง และ LED 200 ms ก่อนส่ง

ถ้ายังไม่ใส่ calibration รุ่น test จะแสดง UNAVAILABLE ส่วนรุ่นแข่งใช้ 0.00 เป็น failure sentinel ชั่วคราว ห้ามตีความเป็นความชื้นจริงและห้ามส่งแข่งก่อนตั้ง READY=true ผลนอก feature range ไม่ถูก extrapolate เป็นเปอร์เซ็นต์สวย ๆ ให้กลับมาตรวจสาเหตุและช่วงอ้างอิง
