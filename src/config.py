import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ─────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME     = os.getenv("DB_NAME", "diet")

# ── Project users ─────────────────────────────────────────────────────────────
# Run `python scripts/list_users.py` to see all user IDs in the database,
# then set the correct IDs for the non-vegetarian (User 1) and vegetarian (User 2).
USER1_ID = 1    # Non-vegetarian
USER2_ID = 2   # Vegetarian

# ── Nutritional constraint nutrient IDs (from `nutrients` table) ──────────────
# C1=Energy(5), C2=Protein(15), C3=Carbohydrate(8), C4=Fiber(4), C5=Sodium(17)
CONSTRAINT_NUTRIENT_IDS = [5, 15, 8, 4, 17]

NUTRIENT_NAMES = {
    5:  "Energy (kcal)",
    15: "Protein (g)",
    8:  "Carbohydrate (g)",
    4:  "Fiber (g)",
    17: "Sodium (mg)",
}

# ── Chromosome structure ──────────────────────────────────────────────────────
BREAKFAST_SIZE    = 94
LUNCH_DINNER_SIZE = 311
TOTAL_FOODS       = 405

# ── DRI epsilon tolerances ────────────────────────────────────────────────────
EPS_UPPER        = 1.15   # effective RUL = RUL × 1.15  (allow 15% over)
EPS_LOWER        = 0.90   # effective RLL = RLL × 0.90  (allow 10% under)
BREAKFAST_RATIO  = 0.35   # breakfast targets 35% of daily DRI

# ── Penalty & diversity ───────────────────────────────────────────────────────
LAMBDA_PENALTY   = 1.0    # penalty weight (tune experimentally)
ALPHA_DIVERSITY  = 0.5    # diversity penalty weight

# ── Algorithm defaults ────────────────────────────────────────────────────────
POP_SIZE  = 100
N_GEN     = 200
P_CROSS   = 0.9
# P_MUT is computed per part as 1/part_length inside each algorithm
