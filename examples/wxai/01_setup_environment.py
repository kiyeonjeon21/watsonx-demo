"""
Example 01: Setup Environment
Demonstrates basic configuration and environment setup using IBM watsonx SDK.
Based on: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/setup_cloud.html

Self-Contained Design:
- No local package imports
- Can be run as Python script or imported into Jupyter/Studio
- Use # %% cell markers for Jupyter notebook conversion
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# %% [markdown]
# # Example 01: Setup Environment
# This script demonstrates how to set up the watsonx demo environment.

# %% Cell 1: Load Environment Variables
# Load environment variables from .env file
load_dotenv()

# Get credentials
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_SPACE_ID = os.getenv('WATSONX_SPACE_ID_DEV')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

print("="*60)
print("EXAMPLE 01: Setup Environment")
print("="*60)
print("\n1. Environment Variables:")
print(f"  API Key: {WATSONX_API_KEY[:10]}..." if WATSONX_API_KEY else "  API Key: Not set")
print(f"  Project ID: {WATSONX_PROJECT_ID}")
print(f"  Space ID: {WATSONX_SPACE_ID}")
print(f"  URL: {WATSONX_URL}")

# %% Cell 2: Model Configuration (Inline)
# Model configuration - all defined inline for self-contained use
MODEL_NAME = os.getenv('MODEL_NAME', "mistralai/mistral-small-3-1-24b-instruct-2503")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'ibm/granite-embedding-107m-multilingual')

# Model parameters
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.0))
# TOP_P = float(os.getenv('TOP_P', 0.9))
# TOP_K = int(os.getenv('TOP_K', 50))
VECTOR_DB_TYPE = os.getenv('VECTOR_DB_TYPE', 'milvus')


# Vector store configuration
print("\n2. Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Embedding Model: {EMBEDDING_MODEL}")
print(f"  Max Tokens: {MAX_NEW_TOKENS}")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Vector DB: {VECTOR_DB_TYPE}")

# Milvus configuration
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_API_KEY = os.getenv('MILVUS_API_KEY')


print(f"\n  Milvus Configuration:")
print(f"    Host: {MILVUS_HOST}")
print(f"    Port: {MILVUS_PORT}")
print(f"    API Key: {'configured' if MILVUS_API_KEY else 'not set'}")


# %% Cell 3: Initialize IBM watsonx AI SDK Client
print("\n3. Initializing IBM watsonx AI SDK client...")
client = None
try:
    # Create credentials using the SDK
    sdk_credentials = Credentials(
        url=WATSONX_URL,
        api_key=WATSONX_API_KEY
    )
    
    # Create API client
    client = APIClient(sdk_credentials)
    
    # Set default project ID
    try:
        client.set.default_project(project_id=WATSONX_PROJECT_ID)
        print(f"✓ SDK client initialized")
        print(f"✓ URL: {WATSONX_URL}")
        print(f"✓ Default Project: {WATSONX_PROJECT_ID}")
    except Exception as project_error:
        # Project permission error - but client might still work
        print(f"⚠ Project assignment failed: {project_error}")
        print("   (Continuing without default project)")
    
except Exception as e:
    print(f"⚠ SDK initialization failed: {e}")
    print("   (This is normal if credentials are not fully configured)")
    client = None

# %% Cell 4: Test Connection (Optional)
print("\n4. Testing connection...")
if client is not None:
    try:
        # List available models
        print("\n  Available Foundation Models:")
        
        # Get enum members and their values
        try:
            models_dict = {name: model.value for name, model in client.foundation_models.TextModels.__members__.items()}
            
            # Show first 10 models as examples
            print("  Sample models (first 10):")
            for idx, (enum_name, model_id) in enumerate(list(models_dict.items())[:10]):
                print(f"    {enum_name}: {model_id}")
            print(f"  ... and {len(models_dict) - 10} more models available")
            print(f"  Total models: {len(models_dict)}")
        except Exception as model_error:
            print(f"  ⚠ Could not enumerate models: {model_error}")

        
        

        # Create ModelInference instance
        model_inference = ModelInference(
            model_id=MODEL_NAME,
            credentials=Credentials(
                url=WATSONX_URL,
                api_key=WATSONX_API_KEY
            ),
            project_id=WATSONX_PROJECT_ID
        )
        
        # Generate text
        response = model_inference.generate(
            prompt="Hello, watsonx!",
            params={
                "max_new_tokens": 50,
                "temperature": TEMPERATURE,
                "stop_sequences": ["```", "\n\n", "\n#"]
            }
        )
        
        # Extract result based on API response format
        if 'results' in response and len(response['results']) > 0:
            result = response['results'][0].get('generated_text', str(response['results'][0]))
        else:
            result = str(response)
            
        print("✓ Connection successful!")
        print(f"\nTest Response: {result}")
        
    except Exception as e:
        print(f"⚠ Connection test failed: {e}")
        print("   Check your API key and project permissions")
else:
    print("⚠ Cannot test connection - client not initialized")

# %% Cell 5: Summary
print("\n✓ Environment setup complete!")
print("="*60)
