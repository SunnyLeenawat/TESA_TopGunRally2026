# Starter agent สำหรับตลาดกาแฟ TESA

โฟลเดอร์นี้เป็น agent ครบชุดสำหรับใช้แข่ง อ้างอิง API ใน `hint/คู่มือ-API.md`
และโครงสร้างจาก `hint/สร้าง-agent.ipynb` / สไลด์วันที่ 18

ปกติทีมแก้แค่ `strategy.py` ไฟล์อื่นเป็นเปลือกเชื่อมต่อ ตรวจตัวเลข และบันทึก log
ให้ agent ไม่ล้มเมื่อตลาดหรือ LLM มีปัญหา

## เริ่มใช้งาน

```powershell
# จาก root ของโปรเจกต์
python -m venv .venv
.\.venv\Scripts\pip.exe install -r starter\requirements.txt
Copy-Item starter\.env.example starter\.env
# แก้ MARKET_URL, TEAM_KEY, LLM_BASE_URL ใน starter\.env
.\.venv\Scripts\python.exe -m starter.main --forever
```

หรือเข้าไปใน `starter/` แล้วรัน `python main.py --forever`

ตลาดฝึก: `--forever` จะเริ่มเกมใหม่ให้เองหลังตลาดปิด (เรียก `/v1/sandbox/reset`)
ตลาดแข่ง: `--forever` จะรอเกมถัดไปจากผู้จัด ไม่พยายาม reset เอง

ตัวเลือกที่มีประโยชน์:

| flag | ความหมาย |
|---|---|
| `--forever` | เล่นต่อหลังจบเกม (ตลาดฝึก reset ให้เอง) |
| `--max-rounds N` | หยุดหลังตัดสินใจครบ N รอบ |
| `--reset` | ตลาดฝึก: เริ่มเกมใหม่ก่อนรัน |
| `--dry-run` | คิดและเขียน log แต่ไม่ POST appraisal/offers |
| `--poll 1` | อ่าน `/v1/state` ทุก 1 วินาที (1–5) |
| `--env-file path` | ใช้ไฟล์ env อื่น |

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | ลูปตาม phase, จัดการ error, ส่ง batch, เขียน `logs/decisions.jsonl` |
| `strategy.py` | `MARKUP`, `spec_ok`, `sell_budget`, `candidate_pairs`, `rule_plan`, `llm_plan`, `guard_offers`, `decide` |
| `appraisal.py` | สูตร TV จาก `rules` ของตลาด (`P0 × G × M × S`) |
| `llm.py` | เรียก gateway แบบ OpenAI-compatible ผ่าน `httpx`; พลาดแล้วคืน `None` |
| `memory.py` | จำผลข้ามรอบใน `logs/memory-*.json` และสรุปให้ LLM |
| `market.py` | client ของ API v1, throttle < 5 req/s, แปลง error เป็น exception |
| `config.py` | อ่าน `.env`, ตรวจ key, ค่า margin/pace |

## ลำดับการทำงาน

1. `SETUP` / `APPRAISAL`: อ่าน `GET /v1/calendar` เอา `rules` แล้วคำนวณ TV
   จากตัวอย่างทุก lot ส่ง `POST /v1/appraisals` ครั้งเดียว
2. `ROUND_OPEN`: ถ้ามี `my_open_offers` อยู่แล้ว (รีสตาร์ตกลางรอบ) จะไม่ส่งซ้ำ
   มิฉะนั้นคำนวณงบขายของรอบ เลือกคู่ RFQ–lot ที่ผ่านสเปก แล้วส่ง
   `POST /v1/offers:batch` ด้วย `replace_existing=true`
3. ถ้า batch ถูกปฏิเสธ (`BATCH_REJECTED`) จะลองส่งทีละรายการและ log
   `offer_rejected` พร้อม `code` / `hint` / `allowed`
4. `ROUND_CLEARING`: รอผล; `CLOSED`: บันทึก `market_closed` และหยุด
   (หรือเล่นเกมใหม่เมื่อ `--forever`)

ทุกค่า TV/กระสอบ/คะแนนมาจาก `rules` ของตลาด ห้าม hard-code ในโค้ด
ถ้าตลาดประกาศตัวเลขใหม่ agent จะใช้ค่านั้นทันที

## ความปลอดภัยของตัวเลข

- `guard_offers()` ตรวจซ้ำก่อนส่งทุกครั้ง: RFQ/lot มีจริง, สเปกผ่าน,
  `kg` เป็นทวีคูณของ `bag_kg`, ไม่เกิน `kg_available`, ไม่เกิน
  `capacity_left_kg`, ราคาอยู่ในกรอบ `LOW..HIGH × TV` และ 0–1000
- LLM เสนอ markup/offers ได้ แต่ Python บีบ markup ให้อยู่ใน `0.85–1.30`
  และตรวจ offers ทุกใบก่อนส่ง
- LLM timeout/429/JSON เสีย → ใช้ `rule_plan()` ทั้งรอบ
- รีสตาร์ตกลางรอบ: ถ้า `my_open_offers` ไม่ว่าง agent จะไม่สร้างชุดใหม่ที่เล็กกว่า
  เพราะ `kg_available` / `capacity_left_kg` ถูกหักข้อเสนอเดิมไปแล้ว

## Log

`logs/decisions.jsonl` เป็น JSON หนึ่งบรรทัดต่อเหตุการณ์:

```text
new_game, appraisal_sent, offers_sent, market_closed,
batch_rejected, offer_rejected, strategy_crashed,
round_already_sent, state_error
```

ตัวอย่าง `offers_sent`:

```json
{"time":"2026-11-17T10:45:12+07:00","event":"offers_sent","round":"R2",
 "source":"llm+rules (1+4)","n":5,"kg":300,"seconds":3.8,
 "offers":[{"rfq_id":"R2-03","lot_id":"LOT-T05-1","kg":120,
            "price_per_kg":121.5,"buyer_type":"COMMERCIAL","markup":1.01}],
 "notes":["llm: ..."],"budget_kg":360}
```

`source` เป็นหนึ่งใน `llm`, `llm+rules (x+y)`, `rules`, `none`
`logs/memory-<game>.json` เก็บ `tv`, `markup`, `sent` และ `rounds`
ใช้วิเคราะห์ย้อนหลังหรือส่งให้ LLM ในรอบถัดไป

## จุดที่ควรปรับ (TODO)

ค้นหา `TODO` ใน `strategy.py` แล้วแก้ทีละข้อ:

1. **MARKUP ต่อประเภทผู้ซื้อ** – เริ่มจาก 1.00 ทุกประเภท แล้วขยับตามผล
   `ACCEPTED` (ขึ้นเล็กน้อย) / `ABOVE_LIMIT`, `OUTBID` (ลง)
2. **spec_ok margins** – เพิ่มระยะจากขอบสเปกถ้าเจอ `SPEC_MISMATCH`
3. **sell_budget** – ใช้ `calendar` จากตลาด ขายมากรอบเทศกาล ถ้ามีตัวเลข
   demand factor ที่ตลาดประกาศ
4. **llm_plan / memory.summary** – เพิ่มข้อมูลที่ LLM ควรเห็น เช่น ผลแยก
   ตามผู้ซื้อหรือปฏิทินรอบถัดไป
5. **estimate_lots** – ดู `reveal` ในตลาดฝึกว่าล็อตแบบไหนประเมินคลาดบ่อย

แก้ทีละข้อ เล่นหลายเกม แล้วเทียบ PR/คะแนน ไม่ควรสรุปจากเกมเดียว

## ทดสอบ

จาก root ของโปรเจกต์:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_starter.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

## ซ้อมกับ demo

ถ้ายังไม่ถึงเวลาแข่งหรือไม่มี gateway ของผู้จัด ให้ใช้ตลาดจำลองใน `demo/`:

```powershell
.\.venv\Scripts\python.exe -m demo.run_demo
.\.venv\Scripts\python.exe -m demo.prompt_preview
```

อ่าน `demo/README_TH.md` และ `demo/PROMPT_DEMO_TH.md`
demo เป็นเครื่องมือฝึกในเครื่อง ไม่ใช่ตลาดของผู้จัด

## หมายเหตุ

- ใช้ LLM ได้เฉพาะ gateway ของผู้จัดในช่วงแข่ง ห้ามสลับไป cloud/ollama อื่น
- ระหว่างแข่ง อย่าเปิดโหมด manual ส่งของหน้าเว็บทับ agent
- ถ้าเจอ `UNAUTHORIZED` ให้ตรวจว่าใช้ key ของตลาดนั้น (ฝึก/แข่งคนละ key)
- ถ้าเจอ `WRONG_PHASE` / `ROUND_CLOSED` แปลว่ารอบปิดแล้ว agent จะรอรอบถัดไป
