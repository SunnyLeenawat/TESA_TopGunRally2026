# Method 1 รุ่นแข่งขัน — TGR-55

ใช้ DOIT ESP32 DEVKIT V1, Arduino ESP32 core 3.3.0 และวงจร LM393 เดิม กดปุ่มหนึ่งครั้งวัด 64 readings ใช้มัธยฐานเวลาแปลงเป็น pF พร้อม LED GPIO2 ติดประมาณ 200 ms

## ใส่ calibration ด้วยตัวเอง

แก้ coefficient ที่เปิดใช้อยู่ใน **namespace Method1Calibration ด้านบน method1_competition.ino** (ตามการแก้ล่าสุดของผู้ใช้ ไฟล์ header ไม่ได้ถูก include):

```cpp
constexpr double SLOPE = 3.031309;
constexpr double INTERCEPT = -16.087502;
```

สูตรคือ `C_pf = SLOPE * median_elapsed_us + INTERCEPT` เวลาเป็น microseconds (µs) ผลเป็น pF ค่าตั้งต้นคัดลอกจาก coefficient ที่ใช้อยู่ในโค้ดเทสของผู้ใช้ ไม่ใช่การรับรองความแม่นยำของรุ่นแข่งนี้

ทั้งสองโค้ดมี coefficient ใน `.ino` ไม่ซิงค์อัตโนมัติ ต้องแก้ตัวที่ไม่ถูก comment และตรวจสมการที่ใช้งานจริง การแก้ `calibration_model.h` ไม่มีผลในเวอร์ชันนี้

## เก็บข้อมูลจาก PuTTY

ดู [ขั้นตอน PuTTY และ calibration](PUTTY_CALIBRATION.md) อัปโหลดรุ่นนี้ก่อน พิมพ์ `c` ใน PuTTY เพื่อเปิดบันทึกเวลา แล้วกดปุ่ม 15 ครั้งต่อ C เก็บ log แยกแต่ละค่า ใช้ `putty_calibration.py import` เพื่อสร้าง CSV ใน `method1_test/data/<setup>/` และ `fit` เพื่อคำนวณ coefficient จากสี่ไฟล์ พิมพ์ `q` หรือรีเซ็ตเพื่อกลับ output แข่งขัน

## วงจร

ใช้ [คู่มือวงจร](../method1_test/README.md) และคงตำแหน่งสายที่ใช้สอบเทียบ:

| จุดต่อ | การต่อ |
| --- | --- |
| GPIO25 | ผ่าน R 1 MΩ ไปจุด A |
| จุด A | LM393 ขา 2 และขาหนึ่งของ C ที่วัด; อีกขาของ C ไป GND |
| Pot ขากลาง | LM393 ขา 3 และ GPIO34; reference ใกล้ 900 mV |
| LM393 ขา 1 | GPIO27 และ R pull-up 2 kΩ ไป 3.3 V |
| GPIO26 | สวิตช์ลง GND มี 100 nF คร่อมสวิตช์ |
| LM393 ขา 8 / ขา 4 | 3.3 V / GND |
| LM393 ขา 5 / ขา 6 / ขา 7 | GND / reference / ไม่ต่อ |
| GPIO2 | LED บนบอร์ด HIGH = ติด ต้องตรวจบอร์ดจริง |

## วิธีใช้งาน

1. ใช้โค้ดเทสตรวจ reference และหาค่า calibration บนวงจรสุดท้าย
2. ใส่ SLOPE และ INTERCEPT ใน `.ino` ของโค้ดแข่ง แล้วบันทึก
3. เปิด method1_competition.ino เลือก DOIT ESP32 DEVKIT V1 และ COM port ปิด capture ที่ใช้ port อยู่ แล้ว Verify/Upload
4. เปิด Serial Monitor ที่ **9600 baud** รอให้บอร์ดบูตและปล่อยปุ่มก่อนกดครั้งแรก
5. กดหนึ่งครั้งแล้วปล่อย รอผลและ LED ดับก่อนกดครั้งต่อไป การกดค้างไม่สร้างผลซ้ำ ต้องปล่อยปุ่มให้เสถียรอย่างน้อย 30 ms

ตัวอย่างรูปแบบ output:

```text
TGR-55
Capacitance = 10.00 pF

```

รุ่นนี้เริ่มวัดด้วยปุ่มจริงเท่านั้น คำสั่ง `c` เปิด CAL_DATA และ `q` ปิดการบันทึก โหมดเริ่มต้นไม่มีข้อความ calibration ไม่มีคำสั่ง d/m/r ดังนั้น verify_competition_serial.py และ capture_serial.py ใช้สั่งวัดรุ่นนี้ไม่ได้ ให้ใช้ PuTTY logging และสคริปต์ import ข้อความ ROM ตอนบูตไม่ใช่ข้อความของ sketch ให้รอบูตเสร็จก่อนวัด

## เมื่อวัดล้มเหลว

ถ้า reference นอก 700–1100 mV, CPU ไม่ใช่ 240 MHz, มี initial_low/timeout แม้แต่หนึ่งครั้ง หรือผลคำนวณไม่เป็นบวก/ไม่ finite จะส่ง `Capacitance = 0.00 pF` ตามรูปแบบเดิม ค่านี้แจ้งความล้มเหลว ไม่ใช่ค่าที่วัดได้จริง ไม่ใช้ค่าครั้งก่อนแทนและไม่ตัดแถวเสียทิ้ง ต้องยืนยันรูปแบบแจ้งความล้มเหลวกับกรรมการ

การผ่านช่วงตรวจ reference ไม่รับรองความแม่นยำ ให้ใช้แรงดันเดียวกับตอนสอบเทียบ เปลี่ยน breadboard หรือสายแล้วต้องตรวจใหม่ ไม่มีการ clamp หรือปัดผลให้ตรงค่าบนป้าย

## ตรวจครั้งสุดท้าย

รุ่นที่มี PuTTY calibration logging นี้ compile ผ่านด้วย ESP32 core 3.3.0 แล้ว (flash 311,283 bytes, static RAM 21,568 bytes) และทดสอบสคริปต์ import/fit ผ่าน ยังไม่ได้อัปโหลดหรือทดสอบการกดปุ่มบนฮาร์ดแวร์ในรุ่นนี้

ตรวจปุ่มกด/กดค้าง/เปิดเครื่องขณะกดค้าง, LED ประมาณ 200 ms, รูปแบบข้อความ และเวลาตอบกลับต่ำกว่า 3 วินาที ทดสอบตัวอ้างอิงอย่างละ 5 ครั้งและ C ที่ไม่ได้ใช้สร้าง calibration ผล hardware เก่าเป็นของ firmware คนละรุ่น ไม่รับรองรุ่นนี้

```powershell
arduino-cli compile --profile doit_esp32_devkit_v1 Track1/firmware/method1_competition
```
