"""
This file parses the full RecipeNLG dataset CSV and separates all the recipies that are 
cookies, cakes, muffins, or brownies. Then those recipies are structured into a JSON file
called knowledgebase.json, which contains the recipies, including the ingredients and ingredient types. 
From the full dataset, around 300,000 recipies are extracted (over 600MB), so I cannot upload that to GitHub :(
"""

import json
import ast
import pandas as pd

df = pd.read_csv("data\\full_dataset.csv",converters={"ingredients": ast.literal_eval,"NER": ast.literal_eval})

target_recipes = []
keywords = ["cookie", "cake", "muffin", "brownie", "biscuit"]

for _, row in df.iterrows():
    title = str(row.get("title", "")).lower()
    if any(kw in title for kw in keywords):
        raw_ingredients = row.get("ingredients", [])
        ner_ingredients = row.get("NER", [])
        
        structured_ingredients = []
        for i, raw_ing in enumerate(raw_ingredients):
            clean_name = ner_ingredients[i] if i < len(ner_ingredients) else str(raw_ing).lower()
            name_lower = clean_name.lower()
            
            if any(w in name_lower for w in ["flour", "starch", "meal"]):
                role_meta = {"role": "structural", "class": "dry"}
            elif any(w in name_lower for w in ["sugar", "honey", "syrup", "sweetener"]):
                role_meta = {"role": "sweetener", "class": "sweet"}
            elif any(w in name_lower for w in ["butter", "oil", "margarine", "shortening"]):
                role_meta = {"role": "fat", "class": "dairy_or_oil"}
            elif any(w in name_lower for w in ["egg", "yogurt"]):
                role_meta = {"role": "binder", "class": "protein"}
            elif any(w in name_lower for w in ["baking powder", "baking soda", "yeast"]):
                role_meta = {"role": "rising_agent", "class": "leavening"}
            elif any(w in name_lower for w in ["salt", "vanilla", "cinnamon", "nutmeg"]):
                role_meta = {"role": "seasoning", "class": "spice_or_flavor"}
            else:
                role_meta = {"role": "add_in_or_flavor", "class": "misc"}
            
            structured_ingredients.append({
                "raw_text": raw_ing,
                "name": name_lower,
                "role": role_meta["role"],
                "class": role_meta["class"]
            })
            
        category = "cookie"
        if "cake" in title:
            category = "cake"
        elif "muffin" in title:
            category = "muffin"
        elif "brownie" in title:
            category = "brownie"

        target_recipes.append({
            "id": f"recipe_{len(target_recipes):03d}",
            "name": row.get("title"),
            "category": category,
            "ingredients": structured_ingredients
        })

knowledgebase = {
    "recipe_categories": ["cookie", "cake", "muffin", "brownie"],
    "recipes": target_recipes
}

with open("knowledgebase.json", "w") as f:
    json.dump(knowledgebase, f, indent=4)