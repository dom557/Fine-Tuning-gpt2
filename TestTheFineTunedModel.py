
# ✅ Recipe-specific testing script for your fine-tuned GPT-2 model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging
import os
import json
import re
from datetime import datetime

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory for test results
test_output_dir = f"./model_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(test_output_dir, exist_ok=True)

# ✅ Path to your fine-tuned model
# Replace with the actual path to your saved model
model_path = "./gpt2-finetuned_YYYYMMDD_HHMMSS/final_model"
if not os.path.exists(model_path):
    logger.warning(f"Model path {model_path} not found. Please update with your actual model path.")
    # List available model directories to help the user
    directories = [d for d in os.listdir('./') if d.startswith('gpt2-finetuned_') and os.path.isdir(d)]
    if directories:
        logger.info(f"Available model directories: {directories}")
        model_path = directories[-1] + "/final_model"  # Use the most recent one
        logger.info(f"Using the most recent model: {model_path}")

# ✅ Load tokenizer and model
try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    logger.info(f"Successfully loaded model from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.info("Falling back to original GPT-2 model")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Using device: {device}")

# ✅ Set padding token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# ✅ Testing functions
def generate_recipe(goal_prompt, max_length=300, temperature=0.7):
    """Generate a recipe based on a nutritional goal"""
    # Create the full prompt
    full_prompt = f"### Goal: {goal_prompt}\n### Recipe: "

    # Encode the prompt
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the recipe part (remove the original prompt)
    recipe_part = generated_text[len(full_prompt):]

    # Clean up any incomplete recipes (stop at the next goal if present)
    if "### Goal:" in recipe_part:
        recipe_part = recipe_part.split("### Goal:")[0].strip()

    return full_prompt + recipe_part

def parse_recipe(recipe_text):
    """Parse a recipe to extract structured information"""
    recipe_data = {
        "title": "",
        "ingredients": [],
        "instructions": [],
        "nutrition": {}
    }

    # Extract title
    title_match = re.search(r"Title: (.*?)(?:\n|$)", recipe_text)
    if title_match:
        recipe_data["title"] = title_match.group(1).strip()

    # Extract ingredients
    ingredients_section = re.search(r"Ingredients:(.*?)Instructions:", recipe_text, re.DOTALL)
    if ingredients_section:
        ingredients_text = ingredients_section.group(1).strip()
        ingredients_list = [item.strip().lstrip('- ') for item in ingredients_text.split('\n') if item.strip()]
        recipe_data["ingredients"] = ingredients_list

    # Extract instructions
    instructions_section = re.search(r"Instructions:(.*?)(?:Nutrition:|$)", recipe_text, re.DOTALL)
    if instructions_section:
        instructions_text = instructions_section.group(1).strip()
        instructions_list = []
        for line in instructions_text.split('\n'):
            line = line.strip()
            if line:
                # Remove numbering if present
                if re.match(r"^\d+\.\s", line):
                    line = re.sub(r"^\d+\.\s", "", line)
                instructions_list.append(line.strip())
        recipe_data["instructions"] = instructions_list

    # Extract nutrition
    nutrition_section = re.search(r"Nutrition: (.*?)(?:\n|$)", recipe_text)
    if nutrition_section:
        nutrition_text = nutrition_section.group(1).strip()
        # Parse calories, sugar, protein, fat
        calories_match = re.search(r"(\d+)\s*calories", nutrition_text)
        sugar_match = re.search(r"(\d+)g\s*sugar", nutrition_text)
        protein_match = re.search(r"(\d+)g\s*protein", nutrition_text)
        fat_match = re.search(r"(\d+)g\s*fat", nutrition_text)

        if calories_match:
            recipe_data["nutrition"]["calories"] = int(calories_match.group(1))
        if sugar_match:
            recipe_data["nutrition"]["sugar"] = int(sugar_match.group(1))
        if protein_match:
            recipe_data["nutrition"]["protein"] = int(protein_match.group(1))
        if fat_match:
            recipe_data["nutrition"]["fat"] = int(fat_match.group(1))

    return recipe_data

def evaluate_recipe_quality(recipe_data, goal):
    """Evaluate the quality of a generated recipe"""
    evaluation = {
        "has_title": bool(recipe_data["title"]),
        "ingredient_count": len(recipe_data["ingredients"]),
        "instruction_count": len(recipe_data["instructions"]),
        "has_nutrition": bool(recipe_data["nutrition"]),
        "matches_goal": False,
        "overall_score": 0  # Will be calculated
    }

    # Check if nutrition information matches the goal
    if recipe_data["nutrition"] and "calories" in recipe_data["nutrition"]:
        # Extract target calories from goal
        calories_match = re.search(r"around (\d+) calories", goal)
        if calories_match:
            target_calories = int(calories_match.group(1))
            actual_calories = recipe_data["nutrition"]["calories"]
            # Check if calories are within 20% of target
            if abs(actual_calories - target_calories) <= (target_calories * 0.2):
                evaluation["matches_goal"] = True

    # Calculate overall score (simple heuristic)
    score = 0
    if evaluation["has_title"]:
        score += 1
    if evaluation["ingredient_count"] >= 3:
        score += 1
    if evaluation["instruction_count"] >= 2:
        score += 1
    if evaluation["has_nutrition"]:
        score += 1
    if evaluation["matches_goal"]:
        score += 1

    evaluation["overall_score"] = score
    return evaluation

# ✅ Main testing function
def run_recipe_tests():
    """Run comprehensive tests on the recipe generation model"""
    # Test prompts based on the training data format
    test_goals = [
        "around 500 calories, high protein, moderate fat",
        "around 350 calories, low carb, high fat",
        "around 400 calories, high fiber, low sugar",
        "around 600 calories, high carb, low fat",
        "around 450 calories, balanced macros",
        "around 300 calories, high protein, very low sugar"
    ]

    # Store test results
    all_results = []

    # Run tests for each goal
    for goal in test_goals:
        logger.info(f"\nGenerating recipe for goal: {goal}")

        # Generate recipe
        recipe_text = generate_recipe(goal)
        logger.info(f"Generated recipe:\n{recipe_text}\n")

        # Save to file
        recipe_filename = os.path.join(test_output_dir, f"recipe_{goal.replace(' ', '_')[:20]}.txt")
        with open(recipe_filename, "w") as f:
            f.write(recipe_text)

        # Parse and evaluate recipe
        recipe_data = parse_recipe(recipe_text)
        evaluation = evaluate_recipe_quality(recipe_data, goal)

        # Log evaluation
        logger.info(f"Recipe evaluation: {evaluation}")

        # Add to results
        result = {
            "goal": goal,
            "recipe_text": recipe_text,
            "parsed_recipe": recipe_data,
            "evaluation": evaluation
        }
        all_results.append(result)

    # Save overall results
    with open(os.path.join(test_output_dir, "test_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Calculate average score
    avg_score = sum(r["evaluation"]["overall_score"] for r in all_results) / len(all_results)
    logger.info(f"\nAverage recipe quality score: {avg_score:.2f}/5")

    return all_results

# ✅ Interactive mode for recipe generation
def interactive_recipe_generation():
    """Interactive mode for testing the recipe generator"""
    print("\n" + "="*50)
    print("INTERACTIVE RECIPE GENERATOR")
    print("="*50)
    print("Enter your nutritional goals, or type 'quit' to exit")
    print("Example: 'around 400 calories, high protein, low fat'")

    while True:
        user_goal = input("\nYour nutritional goal: ")
        if user_goal.lower() == 'quit':
            break

        print("\nGenerating recipe...")
        recipe_text = generate_recipe(user_goal)
        print("\n" + "-"*50)
        print(recipe_text)
        print("-"*50)

        # Save the generated recipe
        timestamp = datetime.now().strftime("%H%M%S")
        recipe_filename = os.path.join(test_output_dir, f"interactive_recipe_{timestamp}.txt")
        with open(recipe_filename, "w") as f:
            f.write(recipe_text)
        print(f"Recipe saved to {recipe_filename}")

# ✅ Run tests
if __name__ == "__main__":
    print("\n" + "="*50)
    print("RECIPE MODEL TESTING")
    print("="*50)
    print("1: Run automated tests")
    print("2: Interactive recipe generation")
    print("3: Run both")

    choice = input("\nEnter your choice (1-3): ")

    if choice in ["1", "3"]:
        print("\nRunning automated tests...")
        results = run_recipe_tests()
        print(f"\nTest results saved to {test_output_dir}")

    if choice in ["2", "3"]:
        interactive_recipe_generation()
