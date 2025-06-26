# api.py — NutriElite MVP API Backend

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import os

# ===== Setup =====

# Path to LoRA adapter and tokenizer files
ADAPTER_PATH = "./adapter_model.safetensors"
TOKENIZER_PATH = "./tokenizer"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Load Model =====

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

print("Loading model with LoRA adapters...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.load_adapter(ADAPTER_PATH, adapter_name="default")
model.set_adapter("default")
model.eval()

# ===== FastAPI App =====

app = FastAPI(title="NutriElite MVP API", version="1.0")

# ===== Input & Output Schema =====

class AthleteProfile(BaseModel):
    age: int
    height: int
    weight: int
    body_fat_percent: int
    goal: str
    activity_level: str
    sport: str
    position: str

class MealItem(BaseModel):
    meal: Optional[str] = None
    supplement: Optional[str] = None
    items: Optional[List[str]] = []
    time: str
    certification: Optional[str] = None

class NutritionPlan(BaseModel):
    target_macros: dict
    meal_plan_and_supplements: List[MealItem]

# ===== Prompt Template =====

def build_prompt(profile: AthleteProfile) -> str:
    return (
        f"""
        You are a certified sports dietitian assistant. Generate a complete daily meal plan and supplement schedule for the following athlete:

        Age: {profile.age} years
        Height: {profile.height} cm
        Weight: {profile.weight} kg
        Body Fat %: {profile.body_fat_percent}%
        Sport: {profile.sport}
        Position: {profile.position}
        Training Goal: {profile.goal}
        Activity Level: {profile.activity_level}

        The plan should be structured in JSON with:
        - Target macros (calories, protein_g, carbs_g, fat_g)
        - A list of meals and supplements, each with time and optional certification
        Ensure the plan meets ±10% caloric accuracy and ≥2g/kg protein targeting.
        """
    )

# ===== Inference Route =====

@app.post("/generate_plan", response_model=NutritionPlan)
def generate_plan(profile: AthleteProfile):
    prompt = build_prompt(profile)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            return_dict_in_generate=True
        )
    decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Extract JSON block from response
    try:
        start = decoded.index('{')
        end = decoded.rindex('}') + 1
        json_plan = decoded[start:end]
        plan_dict = json.loads(json_plan)
        return plan_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON extraction failed: {str(e)}")

# ===== Run via: uvicorn api:app --reload =====
