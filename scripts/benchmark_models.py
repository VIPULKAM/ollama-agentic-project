#!/usr/bin/env python3
"""
Model Benchmarking Script for Qwen2.5-Coder
Tests the model on coding tasks relevant to the AI agent requirements
"""

import subprocess
import time
import json

def test_model(model_name, prompt):
    """Test a model with a prompt and measure response time"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        end_time = time.time()
        response_time = end_time - start_time

        print(f"\n✓ Response Time: {response_time:.2f}s")
        print(f"\nResponse:\n{result.stdout}")

        return {
            "model": model_name,
            "response_time": response_time,
            "response": result.stdout,
            "success": True
        }

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 60s")
        return {
            "model": model_name,
            "response_time": 60,
            "response": "TIMEOUT",
            "success": False
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "model": model_name,
            "error": str(e),
            "success": False
        }

# Test prompts relevant to the AI agent use case
tests = [
    {
        "name": "Python + PostgreSQL",
        "prompt": "Write a Python function to connect to PostgreSQL and execute a parameterized SELECT query with error handling. Keep it concise."
    },
    {
        "name": "TypeScript Function",
        "prompt": "Write a TypeScript function that validates user input data (email, password). Return it as a simple function."
    },
    {
        "name": "SQL Query",
        "prompt": "Write a PostgreSQL query to join users and orders tables, get users who ordered in last 30 days, with pagination."
    },
    {
        "name": "MongoDB Query",
        "prompt": "Write a MongoDB aggregation query to find top 5 products by total sales amount."
    }
]

if __name__ == "__main__":
    models = ["qwen2.5-coder:1.5b"]
    results = []

    print("Model Comparison Test")
    print("=" * 60)
    print("Testing model: qwen2.5-coder:1.5b")
    print("=" * 60)

    for test in tests:
        print(f"\n\n{'#'*60}")
        print(f"TEST: {test['name']}")
        print(f"{'#'*60}")

        for model in models:
            result = test_model(model, test["prompt"])
            result["test_name"] = test["name"]
            results.append(result)
            time.sleep(2)  # Brief pause between tests

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for model in models:
        model_results = [r for r in results if r.get("model") == model and r.get("success")]
        if model_results:
            avg_time = sum(r["response_time"] for r in model_results) / len(model_results)
            print(f"\n{model}:")
            print(f"  - Average response time: {avg_time:.2f}s")
            print(f"  - Successful tests: {len(model_results)}/{len(tests)}")
