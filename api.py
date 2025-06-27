# api.py — NutriElite MVP API Backend

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

# ===== Setup =====

# Set cache directories to use scratch space instead of home directory
os.environ["HF_HOME"] = "/scratch0/giliev/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch0/giliev/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch0/giliev/hf_cache"
os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"

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
    attn_implementation=None,
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

# ===== Prompt Building =====

def build_prompt(profile: AthleteProfile) -> str:
    # Format the prompt to match the training data format
    prompt = f"""<s>[INST] Generate a nutrition plan for an elite athlete with the following profile:

Age: {profile.age} years
Height: {profile.height} cm  
Weight: {profile.weight} kg
Body Fat: {profile.body_fat_percent}%
Sport: {profile.sport}
Position: {profile.position}  
Goal: {profile.goal}
Activity Level: {profile.activity_level}

Provide a complete daily nutrition and supplement plan in JSON format with:
- target_macros: calories, protein_g, carbs_g, fat_g
- meal_plan_and_supplements: array of meals/supplements with time, items, and certification

Ensure protein ≥2g/kg bodyweight and appropriate caloric intake for the goal. [/INST]

""" + """{
  "target_macros": {"""
    
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
                max_new_tokens=256,  # Further reduced for faster generation
                min_new_tokens=50,   
                temperature=0.1,     
                do_sample=False,     # Greedy decoding for speed
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                use_cache=True,  # Enable KV cache for faster generation
                # GPU optimization
                synced_gpus=False,  # Don't sync across GPUs for single GPU setup
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
        
        # Try to extract and parse JSON
        try:
            # First try to find a complete JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: look for JSON boundaries
                json_start = response_text.find('{')
                if json_start == -1:
                    raise ValueError("No JSON block found in response")
                
                json_end = response_text.rfind('}') + 1
                if json_end <= json_start:
                    raise ValueError("Invalid JSON block boundaries")
                
                json_str = response_text[json_start:json_end]
            
            print(f"Extracted JSON: {json_str[:500]}...")
            
            # Clean up the JSON string
            json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            plan_dict = json.loads(json_str)
            
            # Validate the structure
            if "target_macros" not in plan_dict or "meal_plan_and_supplements" not in plan_dict:
                raise ValueError("Missing required fields in generated plan")
            
            # Ensure target_macros has required fields
            if not all(key in plan_dict["target_macros"] for key in ["calories", "protein_g", "carbs_g", "fat_g"]):
                raise ValueError("Missing required macro fields")
            
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
