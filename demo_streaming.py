#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of streaming support with curl / curl 流式支持演示
"""

import subprocess
import json


def test_non_streaming():
    """Test non-streaming request / 测试非流式请求"""
    print("=" * 60)
    print("Testing non-streaming request")
    print("=" * 60)
    
    payload = {
        "model": "mcp-bridge-demo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False
    }
    
    cmd = [
        "curl", "-X", "POST", "http://127.0.0.1:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\nPayload:\n{json.dumps(payload, indent=2)}\n")
    print("Note: Make sure the server is running on localhost:8000")


def test_streaming():
    """Test streaming request / 测试流式请求"""
    print("\n" + "=" * 60)
    print("Testing streaming request")
    print("=" * 60)
    
    payload = {
        "model": "mcp-bridge-demo",
        "messages": [{"role": "user", "content": "Tell me a short story"}],
        "stream": True
    }
    
    cmd = [
        "curl", "-X", "POST", "http://127.0.0.1:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\nPayload:\n{json.dumps(payload, indent=2)}\n")
    print("Expected response format:")
    print("  data: {chunk1}\n")
    print("  data: {chunk2}\n")
    print("  ...\n")
    print("  data: [DONE]\n")
    print("\nNote: Make sure the server is running on localhost:8000")


def test_with_litellm():
    """Show how to use with litellm / 展示如何与 litellm 一起使用"""
    print("\n" + "=" * 60)
    print("Using with litellm")
    print("=" * 60)
    
    code = '''
import litellm

# Configure litellm to use the bridge endpoint
litellm.api_base = "http://127.0.0.1:8000/v1"
litellm.api_key = "test-key"

# Non-streaming request
response = litellm.completion(
    model="openai/mcp-bridge-demo",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False
)
print("Non-streaming response:", response)

# Streaming request
response = litellm.completion(
    model="openai/mcp-bridge-demo",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in response:
    print("Stream chunk:", chunk)
'''
    
    print("Python code example:")
    print(code)


if __name__ == "__main__":
    test_non_streaming()
    test_streaming()
    test_with_litellm()
    print("\n" + "=" * 60)
    print("Streaming implementation complete!")
    print("=" * 60)
