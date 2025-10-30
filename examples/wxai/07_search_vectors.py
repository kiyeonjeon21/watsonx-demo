"""
Example 07: RAG Query
Demonstrates querying the RAG system using IBM watsonx SDK.
Based on: samples/vectorized document milvus build.ipynb

Self-Contained Design:
- Uses IBM watsonx platform Vector Index
- Demonstrates RAG pipeline inline
- Use # %% cell markers for easy conversion to notebook

Note: This is a simplified RAG demo. For full RAG implementation,
see the services/rag/ directory for the complete implementation.
"""

import sys
try:
    from langchain.text_splitter import TextSplitter
except ImportError:
    # LangChain 1.0 호환성: langchain-classic 또는 langchain-text-splitters 사용
    from langchain_text_splitters import TextSplitter
    # 호환 레이어 생성
    class LangChainCompat:
        text_splitter = __import__('langchain_text_splitters')
    sys.modules['langchain.text_splitter'] = LangChainCompat.text_splitter

import os
from pathlib import Path
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import VectorStore
from ibm_watsonx_ai.foundation_models import Embeddings
from dotenv import load_dotenv

# %% [markdown]
# # Example 07: RAG Query
# This script demonstrates querying a RAG system with document retrieval.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
VECTORIZED_DOCUMENT_ASSET_ID = os.getenv('VECTORIZED_DOCUMENT_ASSET_ID')

MODEL_NAME = os.getenv('MODEL_NAME', 'ibm/granite-4-h-small')
TEMPERATURE = 0.3  # Lower temperature for more focused responses

print("="*60)
print("EXAMPLE 07: RAG Query")
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

# %% Cell 3: Get Vector Index Details and Initialize
print("\n2. Getting Vector Index details...")
if not VECTORIZED_DOCUMENT_ASSET_ID:
    print("⚠ VECTORIZED_DOCUMENT_ASSET_ID not set in .env")
    print("  This script requires a Vector Index asset ID")
    print("  Please run example 03 first to create a Vector Index")
    exit(1)

try:
    vector_index_details = client.data_assets.get_details(VECTORIZED_DOCUMENT_ASSET_ID)
    vector_index_properties = vector_index_details["entity"]["vector_index"]
    print(f"✓ Vector Index found")
    print(f"  Vector Index ID: {VECTORIZED_DOCUMENT_ASSET_ID}")
    
    # Get connection and store details
    connection_id = vector_index_properties["store"]["connection_id"]
    index_name = vector_index_properties["store"]["index"]
    database_name = vector_index_properties["store"].get("database")
    embedding_model_id = vector_index_properties["settings"]["embedding_model_id"]
    
    print(f"  Connection ID: {connection_id}")
    print(f"  Index name: {index_name}")
    
    # Get schema if exists
    text_field = None
    if "schema_fields" in vector_index_properties["settings"]:
        vector_store_schema = vector_index_properties["settings"]["schema_fields"]
        text_field = vector_store_schema.get("text")
        
except Exception as e:
    print(f"⚠ Could not load Vector Index: {e}")
    exit(1)

# %% Cell 4: Initialize Embeddings
print("\n3. Initializing embeddings...")
try:
    emb = Embeddings(
        model_id=embedding_model_id,
        credentials=wml_credentials,
        project_id=WATSONX_PROJECT_ID,
        params={
            "truncate_input_tokens": 512
        }
    )
    print(f"✓ Embeddings initialized")
except Exception as e:
    print(f"⚠ Embeddings initialization failed: {e}")
    exit(1)

# %% Cell 5: Initialize Vector Store
print("\n4. Initializing vector store...")
try:
    vector_store = VectorStore(
        client=client,
        connection_id=connection_id,
        embeddings=emb,
        index_name=index_name,
        drop_old=False,
        database=database_name if database_name else None,
        consistency_level='Strong',
        text_field=text_field
    )
    print(f"✓ Vector store initialized")
    
    # Check if vector store has documents
    count = vector_store.count()
    print(f"  Documents in store: {count}")
    
    if count == 0:
        print("⚠ No documents in vector store. Run example 03 first to populate the index.")
        print("  Exiting...")
        exit(0)
        
except Exception as e:
    print(f"⚠ Vector store initialization failed: {e}")
    exit(1)

# %% Cell 6: Query Vector Store
print("\n5. Querying vector store...")
QUERY = "What is Watsonx?"

try:
    # Search for relevant documents
    results = vector_store.search(query=QUERY, k=3)
    
    print(f"✓ Retrieved {len(results)} relevant documents")
    
    if results:
        print(f"\n6. Retrieved context:")
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"\n  Document {i}:")
            print(f"    Content: {content_preview}")
            if doc.metadata:
                print(f"    Metadata: {doc.metadata}")
    else:
        print("⚠ No results found")
        exit(0)
        
except Exception as e:
    print(f"⚠ Search failed: {e}")
    exit(1)

# %% Cell 7: Generate Response with Context
print("\n7. Generating response with context...")

# Build context from retrieved documents
context = "\n\n".join([doc.page_content for doc in results])

# Create prompt with context
prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {QUERY}

Answer:"""

try:
    # Create ModelInference instance
    model_inference = ModelInference(
        model_id=MODEL_NAME,
        credentials=Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY
        ),
        project_id=WATSONX_PROJECT_ID
    )
    
    # Generate response
    response = model_inference.generate(
        prompt=prompt,
        params={
            "max_new_tokens": 512,
            "temperature": TEMPERATURE
        }
    )
    
    # Extract result
    if 'results' in response and len(response['results']) > 0:
        result = response['results'][0].get('generated_text', str(response['results'][0]))
    else:
        result = str(response)
    
    print(f"\nQuery: {QUERY}")
    print(f"\nAnswer:")
    print(result)
    
    print(f"\nSources ({len(results)}):")
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get('filename', 'Unknown') if doc.metadata else 'Unknown'
        print(f"  [{i}] {filename}")
        
except Exception as e:
    print(f"⚠ Generation failed: {e}")

# %% Cell 8: Summary
print("\n✓ Complete!")
print("="*60)
