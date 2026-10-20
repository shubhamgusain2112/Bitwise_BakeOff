import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

with open("knowledgebase.json", "r") as f:
    kb = json.load(f)

recipes = kb["recipes"]

corpus = [" ".join([ing["name"] for ing in r["ingredients"]]) for r in recipes]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
ingredient_idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

def get_idf(name):
    scores = [ingredient_idf.get(w, 2.0) for w in name.split()]
    return max(scores) if scores else 1.0

role_pool = {}
for r in recipes:
    for ing in r["ingredients"]:
        role = ing["role"]
        if role not in role_pool:
            role_pool[role] = []
        if not any(i["name"] == ing["name"] for i in role_pool[role]):
            role_pool[role].append(ing)

required_roles = ["structural", "sweetener", "fat", "binder", "rising_agent", "seasoning"]

def get_initial_bp(role):
    if role == "structural": return 100.0
    if role == "sweetener": return random.uniform(50.0, 120.0)
    if role == "fat": return random.uniform(50.0, 90.0)
    if role == "binder": return random.uniform(60.0, 80.0)
    if role == "rising_agent": return random.uniform(1.0, 3.0)
    if role == "seasoning": return random.uniform(0.5, 2.0)
    return random.uniform(2.0, 5.0)

def create_random_recipe():
    recipe_ingredients = []
    for role in required_roles:
        if role in role_pool and role_pool[role]:
            ing = random.choice(role_pool[role]).copy()
            ing["bp"] = get_initial_bp(role)
            recipe_ingredients.append(ing)
            
    optional_roles = [k for k in role_pool.keys() if k not in required_roles]
    for _ in range(random.randint(1, 3)):
        opt_role = random.choice(optional_roles)
        if role_pool[opt_role]:
            ing = random.choice(role_pool[opt_role]).copy()
            ing["bp"] = get_initial_bp(opt_role)
            recipe_ingredients.append(ing)
            
    return recipe_ingredients

def evaluate_fitness(recipe_ingredients):
    bp_totals = {
        "structural": 0.0, "sweetener": 0.0, "fat": 0.0, 
        "binder": 0.0, "rising_agent": 0.0, "seasoning": 0.0
    }
    
    for ing in recipe_ingredients:
        if ing["role"] in bp_totals:
            bp_totals[ing["role"]] += ing["bp"]
            
    if bp_totals["structural"] == 0:
        return 0.01

    penalty = 0.0
    penalty += abs(100.0 - bp_totals["structural"])
    if bp_totals["sweetener"] < 40 or bp_totals["sweetener"] > 150: penalty += 50
    if bp_totals["fat"] < 30 or bp_totals["fat"] > 100: penalty += 50
    if bp_totals["binder"] < 20 or bp_totals["binder"] > 100: penalty += 50
    if bp_totals["rising_agent"] < 1 or bp_totals["rising_agent"] > 4: penalty += 50
    if bp_totals["seasoning"] > 3: penalty += 50

    rarity_score = 0.0
    for ing in recipe_ingredients:
        idf = get_idf(ing["name"])
        if ing["role"] in ["structural", "sweetener", "fat", "binder", "rising_agent"]:
            rarity_score += (5.0 / idf)
        else:
            rarity_score += idf

    diversity_bonus = len(recipe_ingredients) * 2.0
    
    return max(0.01, rarity_score + diversity_bonus - (penalty * 0.5))

def crossover(parent1, parent2):
    child = []
    roles_seen = set([i["role"] for i in parent1 + parent2])
    
    for role in roles_seen:
        pool = [ing for ing in parent1 + parent2 if ing["role"] == role]
        if pool:
            chosen = random.choice(pool).copy()
            chosen["bp"] = sum(i["bp"] for i in pool) / len(pool)
            child.append(chosen)
            
    if len(child) > 10:
        child = random.sample(child, 10)
        
    for ing in child:
        if ing["role"] == "structural":
            ing["bp"] = 100.0
            
    return child

def mutate(recipe_ingredients):
    mutated = [ing.copy() for ing in recipe_ingredients]
    
    if random.random() < 0.5 and mutated:
        idx = random.randrange(len(mutated))
        role = mutated[idx]["role"]
        if random.random() < 0.5:
            if role in role_pool and role_pool[role]:
                new_ing = random.choice(role_pool[role]).copy()
                new_ing["bp"] = mutated[idx]["bp"]
                mutated[idx] = new_ing
        else:
            if role != "structural":
                mutated[idx]["bp"] *= random.uniform(0.8, 1.2)
                
    return mutated

population_size = 10000
generations = 10
mutation_rate = 0.25

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

output_results = [{"generated_id": f"gen_recipe_{idx+1}", "fitness_score": round(score, 2), "ingredients": recipe} for idx, (recipe, score) in enumerate(unique_recipes)]

with open("generated_unique_recipes.json", "w") as f:
    json.dump(output_results, f, indent=4)