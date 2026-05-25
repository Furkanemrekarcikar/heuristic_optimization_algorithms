from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connection import DBConnection
from .. import config

# Constraint nutrient IDs in fixed column order used throughout the project.
_C_IDS = config.CONSTRAINT_NUTRIENT_IDS   # [5, 15, 8, 4, 17]


# ── DataStore ─────────────────────────────────────────────────────────────────

@dataclass
class DataStore:
    """
    All data required for the MODP, pre-loaded into NumPy arrays for speed.

    Array indexing convention: position i = the i-th food in `food_ids` order.
    The chromosome permutation uses these 0-based indices, NOT raw DB food IDs.
    """

    user_id: int

    # Per-food arrays — shape (n_foods,)
    food_ids:       np.ndarray   # actual DB ids
    food_names:     list[str]
    food_group_ids: np.ndarray   # for diversity calculation
    preferences:    np.ndarray   # user-specific (user_foods override applied)
    costs:          np.ndarray
    prep_times:     np.ndarray   # preparingTime + cookingTime
    co2s:           np.ndarray

    # Nutrient content — shape (n_foods, 5)
    # Column order = CONSTRAINT_NUTRIENT_IDS = [Energy, Protein, Carb, Fiber, Sodium]
    nutrient_matrix: np.ndarray

    # DRI bounds for this user — shape (5,), same column order
    dri_rll: np.ndarray
    dri_rul: np.ndarray

    # DB food ID → array index (needed for output / debugging)
    id_to_idx: dict[int, int]

    # Food group id → name
    food_groups: dict[int, str]

    # Constants (mirrored from config for convenience)
    n_foods:           int  = config.TOTAL_FOODS
    breakfast_size:    int  = config.BREAKFAST_SIZE
    lunch_dinner_size: int  = config.LUNCH_DINNER_SIZE


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(db: DBConnection, user_id: int) -> DataStore:
    """
    Load all data for a given user.  Raises ValueError if critical data
    (foods, DRI) is missing or incomplete.
    """
    foods_raw    = _load_foods(db, user_id)
    ordered_ids  = [r["id"] for r in foods_raw]
    nutrient_mat = _load_nutrients(db, ordered_ids)
    dri_rll, dri_rul = _load_dri(db, user_id)
    food_groups  = _load_food_groups(db)

    n = len(foods_raw)
    food_ids = np.array(ordered_ids, dtype=np.int32)

    return DataStore(
        user_id       = user_id,
        food_ids      = food_ids,
        food_names    = [r["name"]       for r in foods_raw],
        food_group_ids= np.array([r["foodGroupId"] for r in foods_raw], dtype=np.int32),
        preferences   = np.array([r["preference"]  for r in foods_raw], dtype=np.float64),
        costs         = np.array([r["cost"]         for r in foods_raw], dtype=np.float64),
        prep_times    = np.array([r["prep_time"]    for r in foods_raw], dtype=np.float64),
        co2s          = np.array([r["co2"]          for r in foods_raw], dtype=np.float64),
        nutrient_matrix = nutrient_mat,
        dri_rll       = dri_rll,
        dri_rul       = dri_rul,
        id_to_idx     = {fid: i for i, fid in enumerate(food_ids)},
        food_groups   = food_groups,
        n_foods       = n,
    )


def list_users(db: DBConnection) -> list[dict]:
    """Return all users — helper to identify User 1 / User 2 IDs."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, name, surname, username, age, gender "
            "FROM user ORDER BY id"
        )
        return cur.fetchall()


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_foods(db: DBConnection, user_id: int) -> list[dict]:
    """
    Load 405 food items with user-specific preferences.
    user_foods.preference takes priority over foods.preference (as per handout).
    NULL preference values in user_foods fall back to foods.preference via COALESCE.
    """
    query = """
        SELECT
            f.id,
            f.name,
            f.foodGroupId,
            COALESCE(uf.preference, f.preference) AS preference,
            f.cost,
            (f.preparingTime + COALESCE(f.cookingTime, 0)) AS prep_time,
            f.co2
        FROM foods f
        LEFT JOIN user_foods uf
            ON uf.foodId = f.id AND uf.userId = %s
        ORDER BY f.id
    """
    with db.cursor() as cur:
        cur.execute(query, (user_id,))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"No food rows returned for user_id={user_id}. "
            "Verify the user exists and user_foods contains entries for this user."
        )
    return rows


def _load_nutrients(db: DBConnection, food_ids: list[int]) -> np.ndarray:
    """
    Load nutrient content for the 5 constraint nutrients.
    Returns shape (n_foods, 5).  Missing values default to 0.
    """
    placeholders = ",".join(["%s"] * len(_C_IDS))
    query = f"""
        SELECT foodId, nutrientId, quantity
        FROM food_nutrients
        WHERE nutrientId IN ({placeholders})
    """
    with db.cursor() as cur:
        cur.execute(query, tuple(_C_IDS))
        rows = cur.fetchall()

    # food_id → nutrient_id → quantity
    lookup: dict[int, dict[int, float]] = {}
    for row in rows:
        fid = row["foodId"]
        nid = row["nutrientId"]
        lookup.setdefault(fid, {})[nid] = float(row["quantity"])

    n = len(food_ids)
    matrix = np.zeros((n, 5), dtype=np.float64)
    for i, fid in enumerate(food_ids):
        for j, nid in enumerate(_C_IDS):
            matrix[i, j] = lookup.get(fid, {}).get(nid, 0.0)

    return matrix


def _load_dri(db: DBConnection, user_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load DRI lower/upper bounds for the user's age and gender.

    The `dri` table does NOT store userId directly — it stores age ranges + gender.
    We join with `user` to find the matching DRI rows.

    Returns (rll, rul) each shape (5,), column order = CONSTRAINT_NUTRIENT_IDS.
    """
    placeholders = ",".join(["%s"] * len(_C_IDS))
    query = f"""
        SELECT d.nutrient_id, d.RLL, d.RUL
        FROM dri d
        INNER JOIN user u
            ON u.age BETWEEN d.low_age AND d.up_age
           AND LOWER(d.gender) = LOWER(u.gender)
        WHERE u.id = %s
          AND d.nutrient_id IN ({placeholders})
    """
    with db.cursor() as cur:
        cur.execute(query, (user_id, *_C_IDS))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"No DRI rows found for user_id={user_id}. "
            "Check that the user has a valid age and gender, "
            "and that matching DRI rows exist."
        )

    # If multiple rows match (overlapping ranges), take the narrowest range.
    # We group by nutrient_id and keep the entry with smallest (up_age - low_age).
    # Since the raw query doesn't return age columns, re-query with them.
    query_with_range = f"""
        SELECT d.nutrient_id, d.RLL, d.RUL,
               (d.up_age - d.low_age) AS age_span
        FROM dri d
        INNER JOIN user u
            ON u.age BETWEEN d.low_age AND d.up_age
           AND LOWER(d.gender) = LOWER(u.gender)
        WHERE u.id = %s
          AND d.nutrient_id IN ({placeholders})
        ORDER BY d.nutrient_id, age_span ASC
    """
    with db.cursor() as cur:
        cur.execute(query_with_range, (user_id, *_C_IDS))
        rows = cur.fetchall()

    dri_map: dict[int, tuple[float, float]] = {}
    for row in rows:
        nid = row["nutrient_id"]
        if nid not in dri_map:          # first = narrowest age span
            dri_map[nid] = (float(row["RLL"]), float(row["RUL"]))

    missing = [nid for nid in _C_IDS if nid not in dri_map]
    if missing:
        names = [config.NUTRIENT_NAMES.get(nid, str(nid)) for nid in missing]
        raise ValueError(f"DRI data missing for nutrients: {names}")

    rll = np.array([dri_map[nid][0] for nid in _C_IDS], dtype=np.float64)
    rul = np.array([dri_map[nid][1] for nid in _C_IDS], dtype=np.float64)
    return rll, rul


def _load_food_groups(db: DBConnection) -> dict[int, str]:
    with db.cursor() as cur:
        cur.execute("SELECT id, name FROM food_group")
        return {row["id"]: row["name"] for row in cur.fetchall()}
