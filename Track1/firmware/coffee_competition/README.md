# Coffee competition — TGR-55

DOIT ESP32 DEVKIT V1 / core 3.3.0 / 9600 baud / ไม่มี library เพิ่ม

ใช้ [วงจร Day 2](../../Planning_Day2.md) และ [ขั้นตอน calibration](../../calibration/coffee/README.md) โปรแกรมกดปุ่มจริงหนึ่งครั้งวัด 64 readings ด้วย acquisition header เดียวกับโค้ดเทส แล้วเปิด LED GPIO2 ประมาณ 200 ms และส่ง:

```text
TGR-55
Moisture Level = 10.00 %

```

ไม่มี Serial commands/diagnostic/CSV ในรุ่นแข่ง กดค้างไม่วัดซ้ำ ต้องปล่อยปุ่มหลังบูตก่อนกดครั้งแรก ต้องตรวจ LED polarity และเวลาตอบกลับบน hardware จริง

**ยังไม่พร้อมส่งจนกว่าจะสอบเทียบ:** header ตั้ง READY=false และ coefficients เป็น NAN โดยตั้งใจ ไม่ใช้ coefficient ตัวอย่างในสไลด์ หาก calibration ยังไม่พร้อม วัดล้มเหลว หรือนอกช่วงโมเดล จะส่ง 0.00 เป็น failure sentinel ชั่วคราว ไม่ใช่ความชื้นจริง ต้องยืนยัน failure convention กับกรรมการ

แก้ canonical `../coffee_common/moisture_calibration.h` จากผล fit ของ reference bags ของคุณ ตั้ง READY=true หลังตรวจ validation แล้วรัน sync_headers.py ให้ทั้งสอง sketch ตรงกัน เก็บ header ข้าง .ino ก่อน Verify/Upload ทดสอบ known samples ด้วยปุ่มจริงหลังอัปโหลด อย่าเปลี่ยน wiring/fill procedure/pot/baseline หลัง calibration
