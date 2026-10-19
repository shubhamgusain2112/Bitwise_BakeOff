

import json
import re
import sys
from itertools import combinations

INPUT_FILE = "generated_unique_recipes.json"
OUTPUT_FILE = "selected_recipes.json"
N_SELECT = 5

# --- Tunable parameters -----------------------------------------------------

# Penalty weight for sharing an ingredient. The weight of an ingredient is the
# MAX of its class weight and its role weight, so e.g. role "add_in_or_flavor"
# (class "misc") or class "spice_or_flavor" (role "seasoning") are penalised
# heavily, while "structural" (class "dry") is penalised lightly.
CLASS_WEIGHTS = {
    "misc": 3.0,
    "spice_or_flavor": 3.0,
    "dry": 1.0,
    # everything else (sweet, dairy_or_oil, protein, leavening) -> DEFAULT_WEIGHT
}
ROLE_WEIGHTS = {
    "add_in_or_flavor": 3.0,
    "seasoning": 3.0,
    "structural": 1.0,
}
DEFAULT_WEIGHT = 1.5

# How much fitness matters relative to diversity (0 = ignore fitness).
# Fitness is min-max normalised to 0..1 across the input recipes.
FITNESS_WEIGHT = 0.15

# Two ingredient names count as "the same" if their token sets have a
# Jaccard similarity >= this value (1.0 = exact match after normalisation).
NAME_SIMILARITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "for", "to", "in", "with", "into",
    "no", "substitute", "room", "temperature", "around", "fresh", "large",
    "small", "medium", "cup", "cups", "tbsp", "tsp", "unit", "units",
}


def normalize_name(name):
    """Lowercase, drop digits/punctuation/stopwords -> frozenset of tokens."""
    tokens = re.findall(r"[a-z]+", name.lower())
    kept = {t for t in tokens if t not in STOPWORDS and len(t) > 1}
    return frozenset(kept) if kept else frozenset(tokens)


def ingredient_weight(ing):
    cw = CLASS_WEIGHTS.get(ing["class"], DEFAULT_WEIGHT)
    rw = ROLE_WEIGHTS.get(ing["role"], DEFAULT_WEIGHT)
    return max(cw, rw)


def same_ingredient(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return False
    if tokens_a == tokens_b:
        return True
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return jaccard >= NAME_SIMILARITY_THRESHOLD


def prepare(recipe):
    ings = [
        {"tokens": normalize_name(i["name"]), "weight": ingredient_weight(i)}
        for i in recipe["ingredients"]
    ]
    return {
        "id": recipe["generated_id"],
        "fitness": recipe["fitness_score"],
        "ings": ings,
        "total_weight": sum(i["weight"] for i in ings),
    }


def pair_penalty(a, b):
    """Weighted share of overlapping ingredients, in the range 0..1."""
    shared = 0.0
    used = set()
    for ia in a["ings"]:
        for j, ib in enumerate(b["ings"]):
            if j not in used and same_ingredient(ia["tokens"], ib["tokens"]):
                used.add(j)
                shared += max(ia["weight"], ib["weight"])
                break
    denom = min(a["total_weight"], b["total_weight"])
    return min(shared / denom, 1.0) if denom else 0.0


def build_penalty_matrix(prepared):
    n = len(prepared)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = pair_penalty(prepared[i], prepared[j])
    return m


def combo_cost(combo, penalties, fit_norm):
    pairs = list(combinations(combo, 2))
    avg_overlap = sum(penalties[i][j] for i, j in pairs) / len(pairs)
    avg_fitness = sum(fit_norm[i] for i in combo) / len(combo)
    return avg_overlap - FITNESS_WEIGHT * avg_fitness  # lower is better


def greedy_select(n, k, penalties, fit_norm):
    chosen = [max(range(n), key=lambda i: fit_norm[i])]  # fitness head start
    while len(chosen) < k:
        remaining = [i for i in range(n) if i not in chosen]
        best = min(
            remaining,
            key=lambda i: combo_cost(chosen + [i], penalties, fit_norm)
            if len(chosen) + 1 > 1 else 0,
        )
        chosen.append(best)
    return tuple(chosen)


def select_recipes(recipes, k=N_SELECT):
    if len(recipes) < k:
        raise ValueError(f"Need at least {k} recipes, found {len(recipes)}")

    prepared = [prepare(r) for r in recipes]
    penalties = build_penalty_matrix(prepared)

    fits = [p["fitness"] for p in prepared]
    lo, hi = min(fits), max(fits)
    fit_norm = [(f - lo) / (hi - lo) if hi > lo else 1.0 for f in fits]

    n = len(prepared)
    n_combos = 1
    for i in range(k):
        n_combos = n_combos * (n - i) // (i + 1)

    if n_combos <= 2_000_000:
        best = min(
            combinations(range(n), k),
            key=lambda c: combo_cost(c, penalties, fit_norm),
        )
    else:
        best = greedy_select(n, k, penalties, fit_norm)

    best = sorted(best, key=lambda i: -prepared[i]["fitness"])
    return best, prepared, penalties


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE

    with open(in_path, "r") as f:
        recipes = json.load(f)

    best, prepared, penalties = select_recipes(recipes)
    selected = [recipes[i] for i in best]

    with open(out_path, "w") as f:
        json.dump(selected, f, indent=4)

    print(f"Selected {len(selected)} recipes (saved to {out_path}):\n")
    for i in best:
        r = recipes[i]
        print(f"  {r['generated_id']}  fitness={r['fitness_score']}")
        print("     " + ", ".join(ing["name"] for ing in r["ingredients"]))

    print("\nPairwise weighted overlap (0 = nothing shared, 1 = identical):")
    for a, b in combinations(best, 2):
        print(f"  {prepared[a]['id']} <-> {prepared[b]['id']}: {penalties[a][b]:.3f}")


if __name__ == "__main__":
    main()
