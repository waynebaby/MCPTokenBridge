#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本 / Diagnostic script for MCPTokenBridge streaming issues

这个脚本有助于识别流式响应中的具体问题。
"""

import json
import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger("diagnostic")

async def test_direct_non_stream():
    """Test non-streaming directly against mcptb.py"""
    import httpx
    
    logger.info("=" * 60)
    logger.info("Test 1: Non-streaming direct to mcptb.py (port 8000)")
    logger.info("=" * 60)
    
    url = "http://192.168.50.137:8000/v1/chat/completions"
    payload = {
        "model": "local-vscode-copilot",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            logger.info(f"Status: {resp.status_code}")
            data = resp.json()
            logger.info(f"Response: {json.dumps(data, ensure_ascii=False)[:300]}")
            return True
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False

async def test_direct_stream():
    """Test streaming directly against mcptb.py"""
    import httpx
    
    logger.info("=" * 60)
    logger.info("Test 2: Streaming direct to mcptb.py (port 8000)")
    logger.info("=" * 60)
    
    url = "http://192.168.50.137:8000/v1/chat/completions"
    payload = {
        "model": "local-vscode-copilot",
        "messages": [{"role": "user", "content": "Hello streaming"}],
        "stream": True
    }
    
    chunk_count = 0
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                logger.info(f"Status: {resp.status_code}")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk_count += 1
                        data = line[6:]
                        if chunk_count <= 3:
                            logger.info(f"Chunk {chunk_count}: {data[:100]}")
                        if data == "[DONE]":
                            break
        logger.info(f"Total chunks: {chunk_count}")
        return chunk_count > 0
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False

async def test_via_litellm():
    """Test via LiteLLM proxy"""
    import httpx
    
    logger.info("=" * 60)
    logger.info("Test 3: Via LiteLLM proxy (port 4001)")
    logger.info("=" * 60)
    
    url = "http://192.168.50.137:4001/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-litellm-123",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "local-vscode-copilot",
        "messages": [{"role": "user", "content": "Hello via LiteLLM"}],
        "stream": True
    }
    
    chunk_count = 0
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                logger.info(f"Status: {resp.status_code}")
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk_count += 1
                        data = line[6:]
                        if chunk_count <= 3:
                            logger.info(f"Chunk {chunk_count}: {data[:100]}")
                        if data == "[DONE]":
                            break
        logger.info(f"Total chunks: {chunk_count}")
        return chunk_count > 0
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False

async def main():
    logger.info("MCPTokenBridge Streaming Diagnostics")
    logger.info(f"Python: {sys.version}")
    
    # Check imports
    try:
        import httpx
        logger.info(f"httpx: available")
    except ImportError:
        logger.error("httpx not available - install with: pip install httpx")
        return
    
    results = []
    
    # Run tests
    logger.info("\nStarting diagnostic tests...\n")
    
    results.append(("Direct non-stream", await test_direct_non_stream()))
    await asyncio.sleep(1)
    
    results.append(("Direct stream", await test_direct_stream()))
    await asyncio.sleep(1)
    
    results.append(("Via LiteLLM", await test_via_litellm()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Diagnostic Summary")
    logger.info("=" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        logger.info("\n[SUCCESS] All tests passed!")
    else:
        logger.info("\n[FAILURE] Some tests failed. Check logs above.")
    
    return all_passed

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
