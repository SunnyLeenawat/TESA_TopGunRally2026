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
