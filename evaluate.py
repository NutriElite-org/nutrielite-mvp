# evaluate.py — Wrapper for validation

import argparse
import json
from state_of_art_model_evaluator import evaluate_generated_plan  # assumed to be available in path


def evaluate(plan_path):
    with open(plan_path, "r") as f:
        plan = json.load(f)

    print("\nRunning evaluation on generated nutrition plan...")
    result = evaluate_generated_plan(plan)
    print("\nValidation Results:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated nutrition plan.")
    parser.add_argument("--plan", required=True, help="Path to JSON nutrition plan")
    args = parser.parse_args()
    evaluate(args.plan)
