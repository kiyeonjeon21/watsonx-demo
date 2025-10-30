"""
Example 05: Create Vector Index
Demonstrates creating a new Vector Index asset using IBM watsonx SDK.
Based on: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/fm_vector_index.html

Self-Contained Design:
- All code inline, no local package imports
- Use # %% cell markers for easy conversion to notebook
"""

import os
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models.utils import VectorIndexes
from dotenv import load_dotenv

# %% [markdown]
# # Example 05: Create Vector Index
# This script demonstrates creating a new Vector Index asset in IBM watsonx platform.

# %% Cell 1: Setup
load_dotenv()

# Credentials
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

# Vector Index Configuration
VECTORIZED_DOCUMENT_ASSET_NAME = os.getenv('VECTORIZED_DOCUMENT_ASSET_NAME')
VECTORIZED_DOCUMENT_ASSET_DESCRIPTION = os.getenv('VECTORIZED_DOCUMENT_ASSET_DESCRIPTION')

# Store Configuration (external vector store)
MILVUS_CONNECTION_ID = os.getenv('MILVUS_CONNECTION_ID')  # Connection to watsonx.data or external store
VECTOR_INDEX_NAME = os.getenv('VECTOR_INDEX_NAME')
MILVUS_DBNAME = os.getenv('MILVUS_DBNAME', 'default')

# Embedding Model Configuration
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID', 'ibm/granite-embedding-107m-multilingual')

# Settings
TOP_K = int(os.getenv('TOP_K', '5'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))

print("="*60)
print("EXAMPLE 05: Create Vector Index")
print("="*60)

# %% Cell 2: Initialize SDK Client
print("\n1. Initializing SDK client...")
try:
    wml_credentials = {"url": WATSONX_URL, "api_key": WATSONX_API_KEY}
    
    client = APIClient(wml_credentials=wml_credentials)
    client.set.default_project(WATSONX_PROJECT_ID)
    print(f"✓ SDK client initialized")
    print(f"  Project ID: {WATSONX_PROJECT_ID}")
except Exception as e:
    print(f"⚠ Client initialization failed: {e}")
    exit(1)

# %% Cell 3: Initialize Vector Indexes
print("\n2. Initializing Vector Indexes API...")
try:
    vector_indexes = VectorIndexes(api_client=client)
    print(f"✓ Vector Indexes API initialized")
except Exception as e:
    print(f"⚠ Vector Indexes initialization failed: {e}")
    exit(1)

# %% Cell 4: Check Existing Vector Indexes
print("\n3. Checking for existing vector indexes...")
try:
    existing_indexes = vector_indexes.list(limit=10)
    if not existing_indexes.empty:
        print(f"  Found {len(existing_indexes)} existing vector index(es):")
        for idx, row in existing_indexes.iterrows():
            print(f"    - {row.get('name', 'N/A')}: {row.get('asset_id', 'N/A')}")
    else:
        print("  No existing vector indexes found")
except Exception as e:
    print(f"⚠ Could not list existing indexes: {e}")

# %% Cell 5: Create Vector Index
# Check if connection ID is provided
if not MILVUS_CONNECTION_ID:
    print("\n⚠ ERROR: MILVUS_CONNECTION_ID not set in .env")
    print("  You must provide a MILVUS_CONNECTION_ID to create a vector index.")
    print("  Please update your .env file with a valid connection ID.")
    exit(1)

print("\n4. Creating new Vector Index...")
print(f"  Name: {VECTORIZED_DOCUMENT_ASSET_NAME}")
print(f"  Description: {VECTORIZED_DOCUMENT_ASSET_DESCRIPTION}")
print(f"  Embedding Model: {EMBEDDING_MODEL_ID}")
print(f"  Connection ID: {MILVUS_CONNECTION_ID}")

# Create vector index with external store
try:
    vector_index_params = {
        "name": VECTORIZED_DOCUMENT_ASSET_NAME,
        "description": VECTORIZED_DOCUMENT_ASSET_DESCRIPTION,
        "store": {
            "type": "watsonx.data",
            "connection_id": MILVUS_CONNECTION_ID,
            "index": VECTOR_INDEX_NAME,
            "new_index": False,  # Set to True if you want to create a new index
            "database": MILVUS_DBNAME
        },
        "settings": {
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "schema_fields": {"document_name": "document_name", "text": "text", "page_number": "page"}
        },
        "tags": ["watsonx-demo", "vector-index", "rag", "external-store"],
        "sample_questions": [
            "What is watsonx?",
            "How does vector indexing work?",
            "What are the features of watsonx platform?"
        ],
        "status": "ready"
    }
    
    print("\n  Creating vector index with external store...")
    vector_index_details = vector_indexes.create(**vector_index_params)
    print(f"✓ Vector Index created successfully!")
    
    # Extract the asset ID - the ID is at the top level
    vectorized_document_asset_id = vector_index_details.get('id')
    print(f"  Vector Index Asset ID: {vectorized_document_asset_id}")
    
    # Display store configuration
    if 'store' in vector_index_details:
        store_info = vector_index_details['store']
        print(f"\n  Store Configuration:")
        print(f"  Store Type: {store_info.get('type')}")
        print(f"  Index Name: {store_info.get('index')}")
        print(f"  Database: {store_info.get('database')}")
    
    # Save to .env file for later use
    print("\n  Saving Vector Index ID to .env file...")
    env_file = ".env"
    with open(env_file, 'a') as f:
        f.write(f"\nVECTORIZED_DOCUMENT_ASSET_ID={vectorized_document_asset_id}\n")
    print(f"✓ Added VECTORIZED_DOCUMENT_ASSET_ID to .env file")
    
except Exception as e:
    print(f"⚠ Vector Index creation failed: {e}")
    print(f"  Error details: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# %% Cell 6: Verify Vector Index
print("\n5. Verifying created Vector Index...")
try:
    if 'VECTORIZED_DOCUMENT_ASSET_ID' in locals() and vectorized_document_asset_id:
        # List all vector indexes to verify
        existing_indexes = vector_indexes.list(limit=10)
        # Check if the ID column exists and filter
        if 'id' in existing_indexes.columns:
            created_index = existing_indexes[existing_indexes['id'] == vectorized_document_asset_id]
            if not created_index.empty:
                print(f"✓ Vector Index verified")
                print(f"  Name: {created_index.iloc[0].get('name', 'N/A')}")
                print(f"  Status: {created_index.iloc[0].get('status', 'N/A')}")
            else:
                print("⚠ Vector Index not found in list")
        else:
            print(f"✓ Vector Index created with ID: {vectorized_document_asset_id}")
            print("  (Note: Verification skipped - structure may vary)")
    else:
        print("⚠ Cannot verify - Vector Index not created")
except Exception as e:
    print(f"⚠ Verification failed: {e}")

# %% Cell 7: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Next steps:")
print("  1. Update your .env file with the VECTORIZED_DOCUMENT_ASSET_ID")
print("  2. Use example 03 to populate the vector index with documents")
print("  3. Use example 04 to query the vector index")
print("  4. Check your project Assets in the IBM watsonx platform")
print("\n💡 Note:")
print("  - The Vector Index is now ready to be populated with documents")
print("  - You can see it in your watsonx.ai project under Assets")
print("  - If using external store, make sure the connection is properly configured")

