#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of MCPTokenBridge streaming fix
快速测试 MCPTokenBridge 流式修复
"""

import json
import asyncio
import httpx
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def main():
    base_url = "http://192.168.50.137:8000/v1/chat/completions"
    
    # Test 1: Non-streaming
    print("\n" + "="*60)
    print("Test 1: Non-streaming request")
    print("="*60)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                base_url,
                json={
                    "model": "local-vscode-copilot",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
            )
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"Response: {content[:100]}")
                print("✓ Non-streaming test PASSED")
            else:
                print(f"Error: {resp.text}")
                print("✗ Non-streaming test FAILED")
    except Exception as e:
        logger.error(f"Error: {e}")
        print("✗ Non-streaming test FAILED")
    
    await asyncio.sleep(1)
    
    # Test 2: Streaming
    print("\n" + "="*60)
    print("Test 2: Streaming request")
    print("="*60)
    try:
        chunk_count = 0
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST",
                base_url,
                json={
                    "model": "local-vscode-copilot",
                    "messages": [{"role": "user", "content": "Hello streaming"}],
                    "stream": True
                }
            ) as resp:
                print(f"Status: {resp.status_code}")
                if resp.status_code == 200:
                    full_response = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk_count += 1
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if chunk_count <= 3:
                                    print(f"Chunk {chunk_count}: {chunk}")
                                # Extract content
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    full_response += content
                            except json.JSONDecodeError:
                                pass
                    
                    print(f"\nTotal chunks: {chunk_count}")
                    print(f"Full response: {full_response[:100]}")
                    if chunk_count > 0:
                        print("✓ Streaming test PASSED")
                    else:
                        print("✗ Streaming test FAILED - no chunks")
                else:
                    print(f"Error: {resp.text}")
                    print("✗ Streaming test FAILED")
    except Exception as e:
        logger.error(f"Error: {e}")
        print("✗ Streaming test FAILED")

if __name__ == "__main__":
    asyncio.run(main())
