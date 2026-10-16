import json
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

with open("knowledgebase.json", "r") as f:
    kb = json.load(f)

recipes = kb["recipes"]

if len(recipes) > 5000:
    recipes = random.sample(recipes, 5000)

def parse_quantity_and_unit(raw_text):
    if not raw_text:
        return 1.0, "units"
    
    match = re.search(r"^([\d\s\/\.\-]+)?\s*(c|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|g|gram|grams|oz|ounce|ounces|lb|pound|pounds|clove|cloves|pinch|dash)?", str(raw_text).lower())
    
    amount_str = "1.0"
    unit = "units"
    
    if match:
        raw_amt = match.group(1)
        raw_unit = match.group(2)
        
        if raw_amt:
            raw_amt = raw_amt.strip()
            try:
                if "/" in raw_amt:
                    parts = raw_amt.split()
                    val = 0.0
                    for p in parts:
                        if "/" in p:
                            num, denom = p.split("/")
                            val += float(num) / float(denom)
                        else:
                            val += float(p)
                    amount_str = str(val)
                else:
                    amount_str = str(float(raw_amt.split("-")[0].strip()))
            except:
                amount_str = "1.0"
        
        if raw_unit:
            if "c" in raw_unit or "cup" in raw_unit:
                unit = "cups"
            elif "tbsp" in raw_unit or "tablespoon" in raw_unit:
                unit = "tbsp"
            elif "tsp" in raw_unit or "teaspoon" in raw_unit:
                unit = "tsp"
            elif "g" in raw_unit or "gram" in raw_unit:
                unit = "grams"
            elif "oz" in raw_unit or "ounce" in raw_unit:
                unit = "ounces"
            elif "lb" in raw_unit or "pound" in raw_unit:
                unit = "lbs"
            else:
                unit = raw_unit
                
    try:
        amount_val = float(amount_str)
    except:
        amount_val = 1.0
        
    return amount_val, unit

corpus = []
for r in recipes:
    ing_names = [ing["name"] for ing in r["ingredients"]]
    corpus.append(" ".join(ing_names))

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_
ingredient_idf = dict(zip(feature_names, idf_values))

def get_idf(name):
    words = name.split()
    scores = [ingredient_idf.get(w, 2.0) for w in words]
    return max(scores) if scores else 1.0

role_pool = {}
seen_role_ingredients = {}

for r in recipes:
    for ing in r["ingredients"]:
        role = ing["role"]
        name = ing["name"]
        raw_text = ing.get("raw_text", "")
        
        amount, unit = parse_quantity_and_unit(raw_text)
        
        enriched_ing = {
            "name": name,
            "role": role,
            "class": ing["class"],
            "amount": amount,
            "unit": unit
        }
        
        if role not in role_pool:
            role_pool[role] = []
            seen_role_ingredients[role] = set()
            
        if name not in seen_role_ingredients[role]:
            seen_role_ingredients[role].add(name)
            role_pool[role].append(enriched_ing)

required_roles = ["structural", "sweetener", "fat", "binder", "rising_agent"]

def cap_extras(recipe_ingredients, max_extras=5):
    required = [ing for ing in recipe_ingredients if ing["role"] in required_roles]
    extras = [ing for ing in recipe_ingredients if ing["role"] not in required_roles]
    seen_names = set()
    unique_extras = []
    for ex in extras:
        if ex["name"] not in seen_names:
            seen_names.add(ex["name"])
            unique_extras.append(ex)
            
    if len(unique_extras) > max_extras:
        unique_extras = random.sample(unique_extras, max_extras)
    return required + unique_extras

def create_random_recipe():
    recipe_ingredients = []
    for role in required_roles:
        if role in role_pool and role_pool[role]:
            chosen = random.choice(role_pool[role])
            recipe_ingredients.append(chosen.copy())
            
    optional_roles = [k for k in role_pool.keys() if k not in required_roles]
    if optional_roles:
        num_extras = random.randint(1, 3)
        for _ in range(num_extras):
            opt_role = random.choice(optional_roles)
            if role_pool[opt_role]:
                recipe_ingredients.append(random.choice(role_pool[opt_role]).copy())
            
    return cap_extras(recipe_ingredients, max_extras=5)

def evaluate_fitness(recipe_ingredients):
    present_roles = {ing["role"] for ing in recipe_ingredients}
    missing_roles = [r for r in required_roles if r not in present_roles]
    
    if missing_roles:
        return 0.01
    
    rarity_score = sum(get_idf(ing["name"]) for ing in recipe_ingredients)
    diversity_bonus = len(recipe_ingredients) * 0.5
    return rarity_score + diversity_bonus

def crossover(parent1, parent2):
    child = []
    for role in required_roles:
        p1_ings = [ing for ing in parent1 if ing["role"] == role]
        p2_ings = [ing for ing in parent2 if ing["role"] == role]
        pool = p1_ings + p2_ings
        if pool:
            child.append(random.choice(pool).copy())
            
    extras_p1 = [ing for ing in parent1 if ing["role"] not in required_roles]
    extras_p2 = [ing for ing in parent2 if ing["role"] not in required_roles]
    all_extras = extras_p1 + extras_p2
    
    if all_extras:
        sample_size = min(len(all_extras), random.randint(1, 5))
        chosen_extras = random.sample(all_extras, sample_size)
        for ex in chosen_extras:
            child.append(ex.copy())
                
    return cap_extras(child, max_extras=5)

def mutate(recipe_ingredients):
    mutated = [ing.copy() for ing in recipe_ingredients]
    
    if random.random() < 0.4 and mutated:
        idx = random.randrange(len(mutated))
        role = mutated[idx]["role"]
        if role in role_pool and role_pool[role]:
            replacement = random.choice(role_pool[role])
            mutated[idx] = replacement.copy()
            
    elif random.random() < 0.3:
        optional_roles = [k for k in role_pool.keys() if k not in required_roles]
        if optional_roles:
            opt_role = random.choice(optional_roles)
            if role_pool[opt_role]:
                new_ing = random.choice(role_pool[opt_role])
                if not any(i["name"] == new_ing["name"] for i in mutated):
                    mutated.append(new_ing.copy())
                    
    return cap_extras(mutated, max_extras=5)

population_size = 1000
generations = 10
mutation_rate = 0.3

population = [create_random_recipe() for _ in range(population_size)]

for gen in range(generations):
    scored_pop = [(ind, evaluate_fitness(ind)) for ind in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    
    survivors = [ind for ind, score in scored_pop[:population_size // 2]]
    
    next_generation = survivors.copy()
    while len(next_generation) < population_size:
        p1, p2 = random.sample(survivors, 2)
        child = crossover(p1, p2)
        if random.random() < mutation_rate:
            child = mutate(child)
        next_generation.append(child)
        
    population = next_generation

final_scored = [(ind, evaluate_fitness(ind)) for ind in population]
final_scored.sort(key=lambda x: x[1], reverse=True)

unique_recipes = []
seen_signatures = set()

for ind, score in final_scored:
    signature = tuple(sorted([ing["name"] for ing in ind]))
    if signature not in seen_signatures:
        seen_signatures.add(signature)
        unique_recipes.append((ind, score))
    if len(unique_recipes) >= 25:
        break

output_results = []
for idx, (recipe, score) in enumerate(unique_recipes):
    output_results.append({
        "generated_id": f"gen_recipe_{idx+1}",
        "fitness_score": round(score, 2),
        "ingredients": [
            {
                "name": ing["name"],
                "amount": ing["amount"],
                "unit": ing["unit"],
                "role": ing["role"],
                "class": ing["class"]
            }
            for ing in recipe
        ]
    })

with open("generated_unique_recipes.json", "w") as f:
    json.dump(output_results, f, indent=4)