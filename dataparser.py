import ast
import json
import os
import re
from collections import Counter
import pandas as pd

CSV_PATH = os.path.join("data", "full_dataset.csv")
KEYWORDS = ["cookie", "cake", "muffin", "brownie", "biscuit"]
MAX_NAME_WORDS = 3
MIN_NAME_COUNT = 20

STANDARD_UNITS = {
    "c", "cup", "cups", "tsp", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons", 
    "g", "gram", "grams", "oz", "ounce", "ounces", "lb", "pound", "pounds", "ml", "liter", 
    "liters", "pinch", "dash", "drop", "drops", "kg", "kilogram", "pt", "pint", "qt", "quart"
}

ROLE_RULES = [
    ("structural", "dry", r"\b(?:flour|starch|cornstarch|cornmeal|oatmeal|meal|oats)\b"),
    ("sweetener", "sweet", r"\b(?:sugar|honey|syrup|molasses|sweetener|agave)\b"),
    ("fat", "dairy_or_oil", r"\b(?:butter|oil|margarine|shortening|lard|ghee)\b"),
    ("binder", "protein", r"\b(?:eggs?|yogurt|milk|cream|water)\b"),
    ("rising_agent", "leavening", r"\b(?:baking powder|baking soda|yeast)\b"),
    ("seasoning", "spice_or_flavor", r"\b(?:salt|vanilla|cinnamon|nutmeg|extract)\b"),
]
ROLE_RULES = [(r, c, re.compile(p)) for r, c, p in ROLE_RULES]

def classify(name):
    for role, cls, pattern in ROLE_RULES:
        if pattern.search(name):
            return {"role": role, "class": cls}
    return {"role": "add_in_or_flavor", "class": "misc"}

def clean_name(name):
    name = str(name).lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .,;:-")

def is_valid_name(name):
    if len(name) < 3 or len(name.split()) > MAX_NAME_WORDS:
        return False
    if re.search(r"[\d.;:!?()/]", name):
        return False
    return True

def find_raw_line(name, raw_lines):
    for line in raw_lines:
        if name in str(line).lower():
            return str(line)
    return ""

def has_standard_unit(raw_line):
    words = set(re.findall(r'[a-z]+', str(raw_line).lower()))
    return bool(words.intersection(STANDARD_UNITS))

df = pd.read_csv(CSV_PATH)
mask = df["title"].fillna("").str.lower().str.contains("|".join(KEYWORDS))
df = df[mask]

parsed = []
name_counts = Counter()
unit_assoc_counts = Counter()

for _, row in df.iterrows():
    title = str(row["title"])
    try:
        raw_lines = ast.literal_eval(row["ingredients"])
        ner = ast.literal_eval(row["NER"])
    except (ValueError, SyntaxError):
        continue

    items, seen = [], set()
    for n in ner:
        name = clean_name(n)
        if name in seen or not is_valid_name(name):
            continue
        raw = find_raw_line(name, raw_lines)
        if not raw:
            continue
            
        seen.add(name)
        name_counts[name] += 1
        if has_standard_unit(raw):
            unit_assoc_counts[name] += 1
            
        items.append({"name": name, "raw_text": raw})

    parsed.append((title, items))

keep = set()
for n, c in name_counts.items():
    if c >= MIN_NAME_COUNT:
        unit_ratio = unit_assoc_counts.get(n, 0) / c
        if unit_ratio >= 0.4:
            keep.add(n)

target_recipes = []
for title, items in parsed:
    structured = [
        {"name": i["name"], **classify(i["name"])}
        for i in items if i["name"] in keep
    ]
    if len(structured) < 3:
        continue

    t = title.lower()
    category = "cookie"
    if "cake" in t: category = "cake"
    elif "muffin" in t: category = "muffin"
    elif "brownie" in t: category = "brownie"

    target_recipes.append({
        "id": f"recipe_{len(target_recipes):03d}",
        "name": title,
        "category": category,
        "ingredients": structured,
    })

knowledgebase = {
    "recipe_categories": ["cookie", "cake", "muffin", "brownie"],
    "recipes": target_recipes,
}

with open("knowledgebase.json", "w") as f:
    json.dump(knowledgebase, f, indent=4)

print(f"{len(target_recipes)} recipes, {len(keep)} distinct ingredient names kept")