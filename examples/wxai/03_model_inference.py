"""
Example 03: Test All ModelInference Methods
Demonstrates all invoke and stream methods available in ModelInference class.
Based on: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/fm_model_inference.html
"""

import os
import time
import asyncio
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# %% Load Environment
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

# Model configuration
MODEL_NAME = "ibm/granite-4-h-small"

# Create credentials
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

# %% Test 1: Synchronous generate() method
def test_generate():
    print("="*60)
    print("TEST 1: Synchronous generate()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "Write a haiku about artificial intelligence"
        
        response = model.generate(
            prompt=prompt,
            params={
                "max_new_tokens": 100,
                "temperature": 0.7
            }
        )
        
        print(f"Prompt: {prompt}")
        print(f"\nResponse: {response}")
        print("✓ Test 1 passed\n")
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")

# %% Test 2: Streaming generate_text_stream() method
def test_generate_text_stream_basic():
    print("="*60)
    print("TEST 2: Streaming generate_text_stream()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "Tell me a short story about a robot learning to paint"
        
        print(f"Prompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 60)
        
        for chunk in model.generate_text_stream(
            prompt=prompt,
            params={
                "max_new_tokens": 150,
                "temperature": 0.8
            }
        ):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Test 2 passed\n")
        
    except Exception as e:
        print(f"✗ Test 2 failed: {e}\n")

# %% Test 3: Synchronous generate_text() method
def test_generate_text():
    print("="*60)
    print("TEST 3: Synchronous generate_text()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "Explain quantum computing in one sentence"
        
        response = model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 100,
                "temperature": 0.6
            }
        )
        
        print(f"Prompt: {prompt}")
        print(f"\nResponse:\n{response}")
        print("✓ Test 3 passed\n")
        
    except Exception as e:
        print(f"✗ Test 3 failed: {e}\n")

# %% Test 4: Additional generate_text_stream() test
def test_generate_text_stream_advanced():
    print("="*60)
    print("TEST 4: Additional generate_text_stream()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "Describe the future of AI in healthcare"
        
        print(f"Prompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 60)
        
        for chunk in model.generate_text_stream(
            prompt=prompt,
            params={
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9
            }
        ):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Test 4 passed\n")
        
    except Exception as e:
        print(f"✗ Test 4 failed: {e}\n")

# %% Test 5: Synchronous chat() method
def test_chat():
    print("="*60)
    print("TEST 5: Synchronous chat()")
    print("="*60)
    
    try:
        # For chat, we need a chat model
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        messages = [
            {"role": "user", "content": "What is machine learning?"}
        ]
        
        response = model.chat(
            messages=messages,
            params={
                "max_new_tokens": 150,
                "temperature": 0.7
            }
        )
        
        print(f"Messages: {messages}")
        print(f"\nResponse: {response}")
        print("✓ Test 5 passed\n")
        
    except Exception as e:
        print(f"✗ Test 5 failed: {e}\n")

# %% Test 6: Streaming chat_stream() method
def test_chat_stream():
    print("="*60)
    print("TEST 6: Streaming chat_stream()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        messages = [
            {"role": "user", "content": "Write a poem about space exploration"}
        ]
        
        print(f"Messages: {messages}")
        print("\nStreaming response:")
        print("-" * 60)
        
        for chunk in model.chat_stream(
            messages=messages,
            params={
                "max_new_tokens": 150,
                "temperature": 0.8
            }
        ):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Test 6 passed\n")
        
    except Exception as e:
        print(f"✗ Test 6 failed: {e}\n")

# %% Test 7: tokenize() method
def test_tokenize():
    print("="*60)
    print("TEST 7: tokenize()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "This is a test prompt for tokenization"
        
        # Tokenize without returning tokens
        result = model.tokenize(prompt=prompt, return_tokens=False)
        print(f"Prompt: {prompt}")
        print(f"\nTokenization result (without tokens): {result}")
        
        # Tokenize with returning tokens
        result_with_tokens = model.tokenize(prompt=prompt, return_tokens=True)
        print(f"\nTokenization result (with tokens): {result_with_tokens}")
        
        print("✓ Test 7 passed\n")
        
    except Exception as e:
        print(f"✗ Test 7 failed: {e}\n")

# %% Test 8: get_details() method
def test_get_details():
    print("="*60)
    print("TEST 8: get_details()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        details = model.get_details()
        
        print(f"Model Details:")
        print(f"{details}")
        
        print("✓ Test 8 passed\n")
        
    except Exception as e:
        print(f"✗ Test 8 failed: {e}\n")

# %% Test 9: get_identifying_params() method
def test_get_identifying_params():
    print("="*60)
    print("TEST 9: get_identifying_params()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        params = model.get_identifying_params()
        
        print(f"Identifying Parameters:")
        print(f"{params}")
        
        print("✓ Test 9 passed\n")
        
    except Exception as e:
        print(f"✗ Test 9 failed: {e}\n")

# %% Test 10: Async generate() method
async def test_agenerate():
    print("="*60)
    print("TEST 10: Async generate()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "What is the meaning of life in one sentence"
        
        response = await model.agenerate(
            prompt=prompt,
            params={
                "max_new_tokens": 100,
                "temperature": 0.7
            }
        )
        
        print(f"Prompt: {prompt}")
        print(f"\nResponse: {response}")
        print("✓ Test 10 passed\n")
        
    except Exception as e:
        print(f"✗ Test 10 failed: {e}\n")

# %% Test 11: Async generate_stream() method
async def test_agenerate_stream():
    print("="*60)
    print("TEST 11: Async generate_stream()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        prompt = "Write a limerick about programming"
        
        print(f"Prompt: {prompt}")
        print("\nStreaming response:")
        print("-" * 60)
        
        stream = await model.agenerate_stream(
            prompt=prompt,
            params={
                "max_new_tokens": 100,
                "temperature": 0.8
            }
        )
        
        async for chunk in stream:
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Test 11 passed\n")
        
    except Exception as e:
        print(f"✗ Test 11 failed: {e}\n")

# %% Test 12: Async chat() method
async def test_achat():
    print("="*60)
    print("TEST 12: Async chat()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        messages = [
            {"role": "user", "content": "What are the benefits of cloud computing?"}
        ]
        
        response = await model.achat(
            messages=messages,
            params={
                "max_new_tokens": 150,
                "temperature": 0.7
            }
        )
        
        print(f"Messages: {messages}")
        print(f"\nResponse: {response}")
        print("✓ Test 12 passed\n")
        
    except Exception as e:
        print(f"✗ Test 12 failed: {e}\n")

# %% Test 13: Async chat_stream() method
async def test_achat_stream():
    print("="*60)
    print("TEST 13: Async chat_stream()")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        messages = [
            {"role": "user", "content": "Explain blockchain technology"}
        ]
        
        print(f"Messages: {messages}")
        print("\nStreaming response:")
        print("-" * 60)
        
        stream = await model.achat_stream(
            messages=messages,
            params={
                "max_new_tokens": 150,
                "temperature": 0.7
            }
        )
        
        async for chunk in stream:
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("✓ Test 13 passed\n")
        
    except Exception as e:
        print(f"✗ Test 13 failed: {e}\n")

# %% Test 14: Persistent connection with generate()
def test_persistent_connection():
    print("="*60)
    print("TEST 14: Persistent Connection")
    print("="*60)
    
    try:
        model = ModelInference(
            model_id=MODEL_NAME,
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            persistent_connection=True
        )
        
        prompts = [
            "First prompt",
            "Second prompt",
            "Third prompt"
        ]
        
        print("Sending multiple prompts with persistent connection:")
        print("-" * 60)
        
        for prompt in prompts:
            response = model.generate(
                prompt=prompt,
                params={"max_new_tokens": 50}
            )
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        
        # Close the persistent connection
        model.close_persistent_connection()
        
        print("\n" + "-" * 60)
        print("✓ Test 14 passed\n")
        
    except Exception as e:
        print(f"✗ Test 14 failed: {e}\n")

# %% Run all async tests in one event loop
async def run_all_async_tests():
    """Run all async tests in a single event loop"""
    await test_agenerate()
    await asyncio.sleep(1)
    
    await test_agenerate_stream()
    await asyncio.sleep(1)
    
    await test_achat()
    await asyncio.sleep(1)
    
    await test_achat_stream()

# %% Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL INFERENCE TEST SUITE")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Project ID: {WATSONX_PROJECT_ID}")
    print("="*60 + "\n")
    
    # Run all tests
    test_generate()
    time.sleep(1)
    
    test_generate_text_stream_basic()
    time.sleep(1)
    
    test_generate_text()
    time.sleep(1)
    
    test_generate_text_stream_advanced()
    time.sleep(1)
    
    test_chat()
    time.sleep(1)
    
    test_chat_stream()
    time.sleep(1)
    
    test_tokenize()
    time.sleep(1)
    
    test_get_details()
    time.sleep(1)
    
    test_get_identifying_params()
    time.sleep(1)
    
    # Run all async tests in a single event loop
    print("\n" + "="*60)
    print("RUNNING ASYNC TESTS")
    print("="*60)
    asyncio.run(run_all_async_tests())

    test_persistent_connection()
    time.sleep(1)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

