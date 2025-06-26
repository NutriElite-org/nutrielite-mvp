# NutriElite MVP — AI-Powered Nutrition Plan Generator for Elite Athletes

## 🚀 Overview
NutriElite is a real-time nutrition planning system powered by a fine-tuned **Mistral-7B** model with **LoRA adapters**, developed to assist sports dietitians working with professional athletes.

## 🧠 Core Features
- Generates structured daily meal plans and supplement schedules
- Ensures ±10% caloric adherence and ≥2g/kg protein targeting
- Validates JSON output and supplement certifications (NSF/Informed)
- CLI or Streamlit interface for usage

## 🛠 Tech Stack
- **Model**: Mistral-7B Instruct (LoRA fine-tuned)
- **API**: FastAPI (`api.py`)
- **Frontend**: Streamlit or CLI (`main.py`)
- **Evaluation**: Nutrition + JSON validation wrapper (`evaluate.py`)

## 🗂 File Structure
```
📁 nutrielite/
├── api.py                  # FastAPI backend endpoint (/generate_plan)
├── main.py                 # CLI + Streamlit launcher
├── evaluate.py             # Wrapper for evaluation script
├── adapter_model.safetensors  # Fine-tuned LoRA adapter
├── tokenizer/              # Tokenizer config, model, special tokens
├── prompt_template.txt     # Editable generation prompt
├── requirements.txt        # Dependencies
├── README.md               # This file
```

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🔄 Run API
```bash
uvicorn api:app --reload
```

## 🧪 Run CLI
```bash
python main.py
```

## 🌐 Run with Streamlit
```bash
python main.py --streamlit
```

## 🧾 Evaluate Generated Plan
```bash
python evaluate.py --plan outputs/plan_example.json
```

## 📬 API Example
```bash
curl -X POST http://localhost:8000/generate_plan \
  -H "Content-Type: application/json" \
  -d '{"age": 26, "height": 198, "weight": 96, "body_fat_percent": 12, "goal": "cutting", "activity_level": "very_active", "sport": "basketball", "position": "forward"}'
```

## 🧩 Integration
- Ensure `adapter_model.safetensors` and `tokenizer/` files are present in root or correct subpaths
- Load adapters using HuggingFace PEFT or `model.load_adapter(...)`

---
© 2025 Georgi Iliev, MSc @ UCL | NutriElite — AI-Powered Sports Nutrition
