"""
Parses the RecipeNLG CSV, keeps cookie/cake/muffin/brownie/biscuit recipes and writes
knowledgebase.json with clean ingredient names (no instructions, no junk), plus the
raw text line for each ingredient so quantities can still be parsed downstream.
"""
 
import ast
import json
import os
import re
from collections import Counter
 
import pandas as pd
 
CSV_PATH = os.path.join("data", "full_dataset.csv")
KEYWORDS = ["cookie", "cake", "muffin", "brownie", "biscuit"]
 
MAX_NAME_WORDS = 4   # "press batter into pans and sprinkle..." is way over this
MIN_NAME_COUNT = 20  # a name must appear in at least this many recipes to be kept
 
# (role, class, regex). Word boundaries avoid "oil" in "boiled", "egg" in "eggplant", etc.
ROLE_RULES = [
    ("structural", "dry", r"\b(?:flour|starch|cornstarch|cornmeal|oatmeal|meal)\b"),
    ("sweetener", "sweet", r"\b(?:sugar|honey|syrup|molasses|sweetener)\b"),
    ("fat", "dairy_or_oil", r"\b(?:butter|oil|margarine|shortening|lard)\b"),
    ("binder", "protein", r"\b(?:eggs?|yogurt)\b"),
    ("rising_agent", "leavening", r"\b(?:baking powder|baking soda|yeast)\b"),
    ("seasoning", "spice_or_flavor", r"\b(?:salt|vanilla|cinnamon|nutmeg)\b"),
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
    if re.search(r"[\d.;:!?()/]", name):  # digits/sentence punctuation = not an ingredient
        return False
    return True
 
 
def find_raw_line(name, raw_lines):
    """Raw ingredient line that this clean name came from (for the quantity), or ''."""
    for line in raw_lines:
        if name in str(line).lower():
            return str(line)
    return ""
 
 
# ---- Pass 1: filter by title, extract (name, raw_text) pairs per recipe ---------------
df = pd.read_csv(CSV_PATH)  # no converters: parsing the lists for 2M rows is very slow
mask = df["title"].fillna("").str.lower().str.contains("|".join(KEYWORDS))
df = df[mask]
 
parsed = []
name_counts = Counter()
 
for _, row in df.iterrows():
    title = str(row["title"])
    try:
        raw_lines = ast.literal_eval(row["ingredients"])
        ner = ast.literal_eval(row["NER"])
    except (ValueError, SyntaxError):
        continue
 
    # NER is the clean ingredient list. It is NOT aligned 1:1 with the raw list,
    # so iterate over NER and look up the matching raw line instead of using index i.
    items, seen = [], set()
    for n in ner:
        name = clean_name(n)
        if name in seen or not is_valid_name(name):
            continue
        raw = find_raw_line(name, raw_lines)
        if not raw:  # name doesn't appear in any raw ingredient line -> not trustworthy
            continue
        seen.add(name)
        items.append({"name": name, "raw_text": raw})
 
    name_counts.update(i["name"] for i in items)
    parsed.append((title, items))
 
# ---- Pass 2: drop rare names (one-off junk like "master"), build the knowledge base ---
keep = {n for n, c in name_counts.items() if c >= MIN_NAME_COUNT}
 
target_recipes = []
for title, items in parsed:
    structured = [
        {"raw_text": i["raw_text"], "name": i["name"], **classify(i["name"])}
        for i in items
        if i["name"] in keep
    ]
    if len(structured) < 3:
        continue
 
    t = title.lower()
    category = "cookie"
    if "cake" in t:
        category = "cake"
    elif "muffin" in t:
        category = "muffin"
    elif "brownie" in t:
        category = "brownie"
 
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