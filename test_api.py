#!/usr/bin/env python3
"""
Quick test script for the improved NutriElite API
"""

import requests
import json
import sys

def test_api():
    """Test the improved API with a sample athlete profile"""
    
    # Test data
    athlete_data = {
        "age": 25,
        "height": 185,
        "weight": 80,
        "body_fat_percent": 12,
        "goal": "muscle gain",
        "activity_level": "very active", 
        "sport": "basketball",
        "position": "forward"
    }
    
    api_url = "http://127.0.0.1:8000/api/generate_plan"
    
    print("🏀 Testing NutriElite API...")
    print(f"📊 Athlete Profile: {athlete_data}")
    print(f"🌐 API URL: {api_url}")
    print("-" * 50)
    
    try:
        # Make request
        response = requests.post(api_url, json=athlete_data, timeout=60)
        
        # Check response
        if response.status_code == 200:
            plan = response.json()
            print("✅ SUCCESS! Generated plan:")
            print(json.dumps(plan, indent=2))
            
            # Validate structure
            required_fields = ["target_macros", "meal_plan_and_supplements"]
            missing_fields = [field for field in required_fields if field not in plan]
            
            if missing_fields:
                print(f"⚠️  WARNING: Missing fields: {missing_fields}")
            else:
                print("✅ Plan structure is valid!")
                
                # Check macro requirements
                macros = plan["target_macros"]
                protein_requirement = athlete_data["weight"] * 2  # 2g per kg
                
                if macros.get("protein_g", 0) >= protein_requirement:
                    print(f"✅ Protein requirement met: {macros['protein_g']}g >= {protein_requirement}g")
                else:
                    print(f"⚠️  Protein requirement not met: {macros.get('protein_g', 0)}g < {protein_requirement}g")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (60s) - model inference may be too slow")
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - is the API server running?")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("🏥 Health Check:")
            print(json.dumps(health, indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🚀 NutriElite API Test Suite")
    print("=" * 50)
    
    # Test health first
    test_health()
    print()
    
    # Test main functionality
    test_api()
