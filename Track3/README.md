# 1. Install (PowerShell, in \Track3)
```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install -r starter\requirements.txt
copy starter\.env.example starter\.env
```

That installs only httpx and python-dotenv. Use Python 3.10 or newer.

# 2. Edit starter\.env
-MARKET_URL is the market address the organizer gives you.
-TEAM_KEY is your team key. It must start with sk-T, and the placeholder sk-T00-... will be rejected. -Practice and competition markets use different keys.
-USE_LLM=0 turns the LLM off. With our merged strategy, set it to 0 so that only the rules run.
-If you do use the LLM, LLM_BASE_URL and LLM_MODEL come from the organizer, and only their gateway is allowed.
-Leave PACE, SPEC_MARGIN_* and POLL_SECONDS as they are.
# 3. Run

Try a dry run first, which thinks and writes logs but sends nothing:

```powershell
python -m starter.main --dry-run --max-rounds 2
```
Then the real run:

```powershell
python -m starter.main --forever
```

+ --forever keeps playing: on the practice market it resets and starts a new game, and on the competition market it waits for the organizer’s next game.
+ --reset starts a fresh practice game before running.
+ --max-rounds N stops after N rounds.
+ --poll 1 reads the state every second.

# 4. Check it worked
+ Open starter\logs\decisions.jsonl. You should see new_game, appraisal_sent, then offers_sent each round. The source field shows rules-learned with our strategy.
+ Watch for offer_rejected or strategy_crashed. If either shows up, send me the lines.
+ starter\logs\memory-*.json stores the results the strategy learns from.

### Put in present
Guard rejections: kg below one bag after clamping for LOT-T55-3 in R1 and R2. This is the guard working as intended: that lot had less than one bag available for that RFQ. It isn’t a bug.
R1 learning: 0 buyers known, so R1 used the default markups (1.02 and 0.94), which is expected at the start of a game. It already knew 4 buyers by R2. (no same buyer?)


### Note from lecture, take it with a grain of salt
```
1 lot = same quality
diff lot = diff quality

we get 4 lots
1 bag = 60kg

buyer per round, spec quality amount (bag)

each lot = 3 random specs: % defect, % humidity, num(size)[bigger = better price] 
TV = P0 (base price = **120 baht/kg**) * G * M * S

G = Grade factor
<= 1%
<= 3%
etc.

G = (1-d/100)

M = moisture factor
M = (1-M/100) / 0.875 - 0.02 * (m - 12.5)

dont sell lower than 95.80baht/kg
TV normal = 119.75baht/kg


http://161.246.199.191:8000/trade

key ของทีม
sk-T55-cXCK6yF6JN5PwxFN7VB0ORYf

ชื่อทีม
สวนและบ้าน 2026

มหาวิทยาลัย
มหาวิทยาลัยมหิดล


Lock in
http://161.246.199.191:8001/trade
http://202.29.227.22:8001/trade
sk-T55-QSkOFMYXzHH9KkgpAp7OWn9W

Appraisal = TV
Round 1
offers to every RFQ: 
Commercial 1.10 1.03 0.97
Instant 1.00 0.94 0.88

Round 1-5, check same buyer name, amount sold, other outbidded price
ACCEPTED in full: add 0.03 stay under price other outbidded price
PARTIAL: keep same price
OUTBID: drop to the highest sold price but not below 0.88
ABOVE_LIMIT: drop 0.05

Round 6 = offer everythng at 0.90

Day 18
PR = price realization = diff from TV

```