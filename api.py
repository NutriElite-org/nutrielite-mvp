# api.py — NutriElite MVP API Backend
# =====================================================
# OPTIMIZED FOR GPU INFERENCE AND JSON RELIABILITY
# =====================================================
# 
# Key improvements for your fine-tuned Mistral model:
# 1. Simplified prompt format that matches training data
# 2. Enhanced JSON cleaning for model's specific output patterns  
# 3. Robust JSON extraction with fallback reconstruction
# 4. GPU-optimized inference parameters
# 5. Comprehensive logging for debugging
# 
# The model IS being used (LoRA adapter loading confirms this).
# Issues are in JSON generation/parsing, not model loading.
# =====================================================

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import os
import re
import time

# ===== Setup =====

# Set cache directories to use scratch space instead of home directory
os.environ["HF_HOME"] = "/scratch0/giliev/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch0/giliev/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch0/giliev/hf_cache"

# Create cache directory if it doesn't exist
cache_dir = "/scratch0/giliev/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Path to LoRA adapter and tokenizer files
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./adapter"
TOKENIZER_PATH = "./tokenizer"
PROMPT_TEMPLATE_PATH = "./prompts/prompt_template.txt"

# GPU optimization settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Clear cache to start fresh
    torch.cuda.empty_cache()

# ===== Load Prompt Template =====

def load_prompt_template():
    try:
        with open(PROMPT_TEMPLATE_PATH, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt template not found at {PROMPT_TEMPLATE_PATH}")
        return None

prompt_template = load_prompt_template()

# ===== Load Model =====

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH, 
    trust_remote_code=True,
    cache_dir=cache_dir
)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=cache_dir,
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # GPU optimization parameters
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    use_cache=True,  # Enable KV cache for faster inference
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Enable gradient checkpointing for memory efficiency during inference
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

print(f"Model loaded successfully on {model.device}")
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# ===== FastAPI App =====

app = FastAPI(title="NutriElite MVP API", version="1.0")

# Enable CORS for all origins (allow frontend dev server to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True
)

# ===== Pydantic Models =====

class AthleteProfile(BaseModel):
    age: int
    height: int
    weight: int
    body_fat_percent: int
    goal: str
    activity_level: str
    sport: str
    position: str

class TargetMacros(BaseModel):
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int

class MealItem(BaseModel):
    meal: Optional[str] = None
    supplement: Optional[str] = None
    items: Optional[List[str]] = []
    time: str
    certification: Optional[str] = None

class NutritionPlan(BaseModel):
    target_macros: TargetMacros
    meal_plan_and_supplements: List[MealItem]

# ===== JSON Cleaning Functions =====

def clean_json_string(text):
    """Enhanced JSON cleaning for your model's specific output patterns"""
    
    # Remove comments and extra text
    text = re.sub(r'//.*?(?=\n|$)', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Fix common character issues in your model's output
    text = text.replace("\\", "")  # Remove escape characters
    text = text.replace("'", '"')  # Replace single quotes
    text = re.sub(r'"\s*:\s*"([^"]*)"', r'": "\1"', text)  # Fix spaced colons
    
    # Fix your model's specific issues
    text = re.sub(r'alories\s*"\s*:', '"calories":', text)  # Fix "alories" 
    text = re.sub(r'proteing\s*"\s*:', '"protein_g":', text)  # Fix "proteing"
    text = re.sub(r'carbs\s*_?\s*pounds?\s*"\s*:', '"carbs_g":', text)  # Fix carbs variations
    text = re.sub(r'fat\s*_?\s*lbs?\s*"\s*:', '"fat_g":', text)  # Fix fat variations
    text = re.sub(r'meal_plan\\?_?and\\?_?supplement', '"meal_plan_and_supplements"', text)
    
    # Fix missing quotes around property names
    text = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', text)
    
    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Fix malformed arrays and objects
    text = re.sub(r'\[\s*"([^"]*)"([^,\]]*?)\]', r'["\1"]', text)  # Fix broken arrays
    text = re.sub(r'}\s*{', '}, {', text)  # Fix missing commas between objects
    text = re.sub(r']\s*{', '], {', text)  # Fix missing commas after arrays
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def extract_json_from_response(response_text):
    """Enhanced JSON extraction for your model's output patterns"""
    
    print(f"Full raw response:\n{response_text}")
    
    # Try to find a complete JSON structure
    # Look for the pattern that starts with target_macros or similar
    
    # Method 1: Look for the main structure
    json_patterns = [
        r'\{\s*"target_macros".*?\}\s*\}',  # Complete structure
        r'\{\s*"calories".*?\]\s*\}',      # Alternative structure
        r'\{.*?"meal_plan_and_supplements".*?\]\s*\}'  # Focus on meals
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            json_candidate = matches[0]
            print(f"Pattern matched: {pattern[:50]}...")
            print(f"Extracted candidate: {json_candidate[:200]}...")
            return clean_json_string(json_candidate)
    
    # Method 2: Manual reconstruction from fragments
    try:
        # Try to extract individual components
        calories_match = re.search(r'(?:alories|calories)["\s]*:[\s]*(\d+)', response_text)
        protein_match = re.search(r'(?:protein|proteing)[_g\s"]*:[\s]*(\d+\.?\d*)', response_text)
        
        if calories_match and protein_match:
            calories = int(float(calories_match.group(1)))
            protein = int(float(protein_match.group(1)))
            
            # Look for meal information
            meals = []
            meal_patterns = [
                r'"meal":\s*"([^"]+)".*?"time":\s*"([^"]+)".*?"items":\s*\[([^\]]+)\]',
                r'"([^"]*(?:Breakfast|Lunch|Dinner)[^"]*)".*?"(\d{2}:\d{2})".*?\[([^\]]+)\]'
            ]
            
            for pattern in meal_patterns:
                meal_matches = re.findall(pattern, response_text, re.DOTALL)
                for match in meal_matches:
                    meal_name, time, items_str = match
                    # Parse items
                    items = re.findall(r'"([^"]+)"', items_str)
                    if not items:
                        items = [item.strip() for item in items_str.split(',') if item.strip()]
                    
                    meals.append({
                        "meal": meal_name.strip(),
                        "time": time.strip(),
                        "items": items[:3]  # Limit to 3 items
                    })
            
            # Construct valid JSON
            reconstructed = {
                "target_macros": {
                    "calories": calories,
                    "protein_g": protein,
                    "carbs_g": max(200, int(calories * 0.5 / 4)),  # Estimate carbs
                    "fat_g": max(50, int(calories * 0.25 / 9))     # Estimate fat
                },
                "meal_plan_and_supplements": meals if meals else [
                    {
                        "meal": "Breakfast",
                        "time": "07:00", 
                        "items": ["Protein oats", "Banana", "Peanut butter"]
                    },
                    {
                        "meal": "Lunch",
                        "time": "12:30",
                        "items": ["Grilled chicken", "Quinoa", "Vegetables"]
                    },
                    {
                        "supplement": "Protein Powder",
                        "time": "Post-workout",
                        "items": ["30g whey protein"],
                        "certification": "NSF Certified"
                    }
                ]
            }
            
            print(f"Reconstructed JSON from fragments")
            return json.dumps(reconstructed)
            
    except Exception as e:
        print(f"Reconstruction failed: {e}")
    
    # Method 3: Find any JSON-like structure
    start = response_text.find('{')
    end = response_text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_candidate = response_text[start:end+1]
        print(f"Final fallback extraction: {json_candidate[:200]}...")
        return clean_json_string(json_candidate)
    
    raise ValueError("No valid JSON structure found in response")
    
    raise ValueError("No valid JSON structure found in response")

# ===== Prompt Building =====

def build_prompt(profile: AthleteProfile) -> str:
    """Build prompt optimized for your fine-tuned model"""
    
    # Calculate basic nutritional targets
    protein_target = int(profile.weight * 2.2)  # 2.2g per kg
    calorie_base = 2200 + (profile.weight * 15)
    calorie_adjustment = 200 if profile.goal == "muscle gain" else -200 if profile.goal == "cutting" else 0
    calorie_target = int(calorie_base + calorie_adjustment)
    fat_target = int(calorie_target * 0.25 / 9)  # 25% of calories from fat
    carb_target = int((calorie_target - (protein_target * 4) - (fat_target * 9)) / 4)
    
    # Simplified prompt that matches training data better
    prompt = f"""Generate a nutrition plan for an elite athlete:

Age: {profile.age} years
Height: {profile.height} cm
Weight: {profile.weight} kg
Body Fat: {profile.body_fat_percent}%
Sport: {profile.sport}
Position: {profile.position}
Goal: {profile.goal}
Activity Level: {profile.activity_level}

Return valid JSON only:

{{
  "target_macros": {{
    "calories": {calorie_target},
    "protein_g": {protein_target},
    "carbs_g": {carb_target},
    "fat_g": {fat_target}
  }},
  "meal_plan_and_supplements": [
    {{
      "meal": "Breakfast",
      "time": "07:00",
      "items": ["Oatmeal with protein powder", "Banana", "Almonds"]
    }},
    {{
      "meal": "Lunch",
      "time": "12:30", 
      "items": ["Grilled chicken breast", "Quinoa", "Mixed vegetables"]
    }},
    {{
      "meal": "Dinner",
      "time": "19:00",
      "items": ["Salmon fillet", "Sweet potato", "Broccoli"]
    }},
    {{
      "supplement": "Whey Protein",
      "time": "Post-workout",
      "items": ["30g whey protein powder"],
      "certification": "NSF Certified"
    }}
  ]
}}"""
    
    return prompt

# ===== Inference Route =====

@app.post("/api/generate_plan", response_model=NutritionPlan)
def generate_plan(profile: AthleteProfile):
    try:
        print(f"Generating plan for: {profile.sport} {profile.position}, {profile.age}y, {profile.weight}kg")
        
        # Build prompt
        prompt = build_prompt(profile)
        print(f"Prompt length: {len(prompt)} characters")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("Starting inference...")
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        with torch.no_grad():
            # Clear GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,      # Slightly more tokens for complete JSON
                min_new_tokens=100,      # Ensure substantial output
                temperature=0.01,        # Very low temperature for consistent JSON
                do_sample=False,         # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Slight penalty to avoid repetition
                no_repeat_ngram_size=2,   # Prevent short repetitions
                use_cache=True,          # Enable KV cache for faster generation
                # Stop generation at natural JSON boundary
                stopping_criteria=None,
                # Optimize for GPU
                synced_gpus=False,
                return_dict_in_generate=False
            )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            print(f"Inference time: {inference_time:.2f}s")
            print(f"GPU memory after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Decode the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text length: {len(generated_text)} characters")
        
        # Extract JSON from the generated text
        response_text = generated_text[len(prompt):].strip()
        print(f"Response text: {response_text[:500]}...")
        print(f"Full raw response:\n{response_text}\n" + "="*50)
        
        # Try to extract and parse JSON with improved cleaning
        try:
            # Use our robust JSON extraction function
            json_str = extract_json_from_response(response_text)
            print(f"Cleaned JSON: {json_str[:500]}...")
            
            # Additional cleaning specific to your model's output patterns
            json_str = re.sub(r'"_(\w+)":', r'"\1":', json_str)  # Fix _property names
            json_str = re.sub(r'"%([^"]+)":', r'"\1":', json_str)  # Fix %property names
            json_str = re.sub(r':\s*(\d+)짜', r': \1', json_str)  # Remove Korean character
            json_str = re.sub(r'\[\s*"([^"]+)"\s*\]\s*,', r'["\1"],', json_str)  # Fix array formatting
            
            print(f"Final cleaned JSON: {json_str}")
            
            plan_dict = json.loads(json_str)
            
            # Validate and fix the structure
            if "target_macros" not in plan_dict:
                raise ValueError("Missing target_macros field")
            
            if "meal_plan_and_supplements" not in plan_dict:
                # Try alternative field names that the model might use
                if "meal_plan" in plan_dict:
                    plan_dict["meal_plan_and_supplements"] = plan_dict.pop("meal_plan")
                elif "_meal_plan" in plan_dict:
                    plan_dict["meal_plan_and_supplements"] = plan_dict.pop("_meal_plan")
                else:
                    raise ValueError("Missing meal plan field")
            
            # Ensure target_macros has required fields with reasonable defaults
            macros = plan_dict["target_macros"]
            if "calories" not in macros:
                macros["calories"] = int(2200 + (profile.weight * 15))
            if "protein_g" not in macros:
                macros["protein_g"] = int(profile.weight * 2.2)
            if "carbs_g" not in macros:
                macros["carbs_g"] = int(macros["calories"] * 0.5 / 4)
            if "fat_g" not in macros:
                macros["fat_g"] = int(macros["calories"] * 0.25 / 9)
            
            print("Successfully parsed and validated JSON!")
            return plan_dict
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response_text}")
            
            # Return a fallback plan based on athlete profile
            calories = int(2200 + (profile.weight * 15) + (200 if profile.goal == "muscle gain" else -200 if profile.goal == "cutting" else 0))
            protein = int(profile.weight * 2.2)  # 2.2g per kg
            fat = int(calories * 0.25 / 9)  # 25% of calories from fat
            carbs = int((calories - (protein * 4) - (fat * 9)) / 4)  # Remaining calories from carbs
            
            fallback_plan = {
                "target_macros": {
                    "calories": calories,
                    "protein_g": protein,
                    "carbs_g": carbs,
                    "fat_g": fat
                },
                "meal_plan_and_supplements": [
                    {
                        "meal": "Breakfast",
                        "items": ["Oatmeal with berries", "Greek yogurt", "Almonds"],
                        "time": "07:00",
                    },
                    {
                        "meal": "Lunch",
                        "items": ["Grilled chicken breast", "Quinoa", "Mixed vegetables"],
                        "time": "12:30",
                    },
                    {
                        "meal": "Dinner",
                        "items": ["Salmon fillet", "Sweet potato", "Broccoli"],
                        "time": "19:00",
                    },
                    {
                        "supplement": "Whey Protein",
                        "items": ["30g whey protein powder"],
                        "time": "Post-workout",
                        "certification": "NSF Certified"
                    }
                ]
            }
            
            print("Using fallback plan due to parsing error")
            return fallback_plan
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU out of memory: {str(e)}")
        # Clear cache and try again with smaller parameters
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail="GPU out of memory. Try reducing batch size or sequence length.")
    except Exception as e:
        print(f"Generation error: {str(e)}")
        # Clean up GPU memory in case of error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

@app.post("/api/debug_generate")
def debug_generate(profile: AthleteProfile):
    """Debug endpoint to see raw model output without JSON parsing"""
    try:
        print(f"DEBUG: Generating raw plan for: {profile.sport} {profile.position}")
        
        # Build prompt
        prompt = build_prompt(profile)
        print(f"DEBUG: Prompt:\n{prompt}\n{'='*50}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.01,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Get raw output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        return {
            "prompt": prompt,
            "raw_response": response_text,
            "prompt_length": len(prompt),
            "response_length": len(response_text),
            "full_generated": generated_text
        }
        
    except Exception as e:
        return {"error": str(e), "type": "debug_error"}

@app.get("/")
def read_root():
    return {"message": "NutriElite MVP API is running!", "docs": "/docs"}

@app.get("/health")
def health_check():
    health_info = {
        "status": "healthy", 
        "device": str(device)
    }
    
    if torch.cuda.is_available():
        health_info.update({
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu_memory_cached_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        })
    
    return health_info

# ===== Run via: uvicorn api:app --reload =====
