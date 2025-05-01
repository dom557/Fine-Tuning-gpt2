# require : Flask pyngrok flask_cors


from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import re
import logging
from datetime import datetime
from pyngrok import ngrok  # Import ngrok for exposing the Flask app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model loading
MODEL_PATH = os.environ.get('MODEL_PATH', '/content/gpt2-finetuned_20250429_124636/final_model')
tokenizer = None
model = None
device = None

# Define the model loader
def load_model():
    global tokenizer, model, device

    logger.info(f"Loading model from {MODEL_PATH}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        logger.info(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Falling back to original GPT-2 model")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

def generate_recipe(goal_prompt, max_length=300, temperature=0.7):
    """Generate a recipe based on a nutritional goal"""
    # Ensure model is loaded
    global tokenizer, model, device
    if tokenizer is None or model is None:
        load_model()

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

@app.route('/', methods=['GET'])
def home():
    """API homepage"""
    return jsonify({
        "name": "Recipe Generation API",
        "version": "1.0.0",
        "endpoints": {
            "/api/healthcheck": "Check API status",
            "/api/generate": "Generate a recipe (POST)",
        }
    })

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    """API endpoint to check if the service is running"""
    return jsonify({
        "status": "ok",
        "message": "Recipe API is operational",
        "timestamp": str(datetime.now())
    })

@app.route('/api/generate', methods=['POST'])
def api_generate_recipe():
    """API endpoint to generate a recipe"""
    data = request.json

    if not data or 'goal' not in data:
        return jsonify({"error": "Missing 'goal' parameter"}), 400

    goal = data['goal']
    max_length = data.get('max_length', 300)  # Optional parameter
    temperature = data.get('temperature', 0.7)  # Optional parameter
    parse = data.get('parse', True)  # Whether to parse the recipe into structured data

    try:
        # Generate recipe
        recipe_text = generate_recipe(goal, max_length, temperature)

        response = {
            "goal": goal,
            "recipe_text": recipe_text,
        }

        return jsonify({"recipe_text": recipe_text})

    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()  # << Explicitly load model here

    # Start ngrok tunnel
    port = int(os.environ.get('PORT', 5000))
    public_url = ngrok.connect(port)
    print(f"Flask app is accessible at {public_url}")

    # Start Flask
    app.run(host='0.0.0.0', port=port)
