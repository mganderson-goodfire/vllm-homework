#!/usr/bin/env python3
"""
Simple vLLM inference test script
Tests that vLLM is working with a small model
"""

import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

print("Importing vLLM...")
from vllm import LLM, SamplingParams
print("✓ vLLM imported successfully")
print()

print("Initializing vLLM with a tiny model...")
print("Model: facebook/opt-125m (125M parameters, ~500MB)")
print("This may take a moment on first run to download the model...")
print()

# Use one of the smallest models available
# Set environment variable for more logging
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

llm = LLM(
    model="facebook/opt-125m",  # 125M parameter model, ~500MB download
    max_model_len=256,           # Short context for quick testing
    trust_remote_code=False,     # For safety
)

# Simple sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=30,
)

# Test prompts
prompts = [
    "The weather today is",
    "Python is a",
    "Hello world",
]

print("Starting inference generation...")
print(f"Number of prompts: {len(prompts)}")
print(f"Sampling params: temperature={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}")
print()

outputs = llm.generate(prompts, sampling_params)
print("✓ Generation complete")

# Display results
print("\n" + "="*50)
print("Results:")
print("="*50)
for output in outputs:
    print(f"\nPrompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")

print("\n✅ vLLM is working!")