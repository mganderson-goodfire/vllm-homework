#!/usr/bin/env python3
"""
Simple vLLM inference test script
Tests that vLLM is working with a small model
"""

from vllm import LLM, SamplingParams

print("Initializing vLLM with a tiny model...")

# Use one of the smallest models available
llm = LLM(
    model="facebook/opt-125m",  # 125M parameter model, ~500MB download
    max_model_len=256,           # Short context for quick testing
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

print("Running inference...")
outputs = llm.generate(prompts, sampling_params)

# Display results
print("\n" + "="*50)
print("Results:")
print("="*50)
for output in outputs:
    print(f"\nPrompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")

print("\n✅ vLLM is working!")