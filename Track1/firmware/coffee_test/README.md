# Coffee test

DOIT ESP32 DEVKIT V1 / core 3.3.0 / 9600 baud / ไม่มี library เพิ่ม

อ่าน [แผนวงจร Day 2](../../Planning_Day2.md) และ [วิธีเก็บข้อมูล/สอบเทียบ](../../calibration/coffee/README.md)

คำสั่งผ่าน Serial หรือ PuTTY (ต้อง Enter): run = 15 batches, m = หนึ่ง batch, r = reference mV, stop = ยกเลิก ปุ่มจริงวัดหนึ่ง batch เมื่อไม่มี run ทำงาน ข้อมูล COFFEE_DATA คือมัธยฐานของ 64 readings พร้อม MAD/status ไม่ใช่ค่า moisture ที่รู้จริง

ยังไม่ใส่โมเดลจะแสดง UNAVAILABLE ใช้ labelled bags A=11.4%, B=14.4%, C=6.6% และ empty-box baseline เพื่อ fit ก่อน ไม่ใช้สมการตัวอย่างจากสไลด์

แก้ header หลักใน `../coffee_common/` แล้วรัน sync_headers.py เพื่อคัดลอกไปสอง sketch ต้องมี coffee_sensor.h และ moisture_calibration.h อยู่ข้าง .ino
