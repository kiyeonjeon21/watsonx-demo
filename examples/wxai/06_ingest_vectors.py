"""
Example 06: Create Vectors
Demonstrates creating and indexing document embeddings using IBM watsonx SDK.
Based on: samples/vectorized document milvus build.ipynb

Self-Contained Design:
- All code inline, no local package imports
- Use # %% cell markers for easy conversion to notebook
"""

# 06_ingest_vectors.py 상단에 추가
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
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores import VectorStore
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.foundation_models.extensions.rag.chunker import LangChainChunker
from langchain_core.documents import Document
from dotenv import load_dotenv

# %% [markdown]
# # Example 06: Create Vectors
# This script demonstrates creating document embeddings and indexing them to Vector Store.

# %% Cell 1: Setup
load_dotenv()

# Credentials
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

# Configuration
VECTORIZED_DOCUMENT_ASSET_ID = os.getenv('VECTORIZED_DOCUMENT_ASSET_ID')  # Vector Index asset ID

print("="*60)
print("EXAMPLE 06: Create Vectors")
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

# %% Cell 3: Get Vector Index Details
print("\n2. Getting Vector Index details...")
if not VECTORIZED_DOCUMENT_ASSET_ID:
    print("⚠ VECTORIZED_DOCUMENT_ASSET_ID not set in .env")
    print("  This script requires a Vector Index asset ID")
    print("  Please set VECTORIZED_DOCUMENT_ASSET_ID in your .env file")
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
    print(f"  Embedding model: {embedding_model_id}")
    
    # Get schema if exists
    text_field = None
    if "schema_fields" in vector_index_properties["settings"]:
        vector_store_schema = vector_index_properties["settings"]["schema_fields"]
        text_field = vector_store_schema.get("text")
        print(f"  Text field: {text_field}")
    
    # Get chunk_size and chunk_overlap from vector index settings
    # Default values if not specified in vector index
    CHUNK_SIZE = int(vector_index_properties["settings"].get("chunk_size", 1000))
    CHUNK_OVERLAP = int(vector_index_properties["settings"].get("chunk_overlap", 200))
    
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Chunk overlap: {CHUNK_OVERLAP}")
        
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
        drop_old=True, 
        database=database_name,
        consistency_level='Strong',
        connection_args={'secure': True},
        text_field=text_field
    )
    print(f"✓ Vector store initialized")
    print(f"  Index: {index_name}")
    print(f"  Connection: {connection_id}")
except Exception as e:
    print(f"⚠ Vector store initialization failed: {e}")
    exit(1)

# %% Cell 6: Create Sample Data
print("\n5. Preparing documents...")

# Create sample data
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

sample_file = data_dir / "sample.txt"
with open(sample_file, 'w') as f:
    f.write(
        "Watsonx is IBM's enterprise AI platform. "
        "It provides comprehensive AI capabilities including foundation models, "
        "vector databases, and orchestration. "
        "The platform enables organizations to build, deploy, and govern AI applications. "
        "Watsonx offers multi-model AI capabilities with strong governance features. "
        "It supports various use cases including text generation, document understanding, "
        "and conversational AI. The platform integrates seamlessly with IBM Cloud infrastructure."
    )

print(f"✓ Created sample document: {sample_file}")

# %% Cell 7: Process and Chunk Documents
print("\n6. Processing documents...")

# Create text splitter
text_splitter = LangChainChunker(
    method="recursive",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Load document as LangChain Document
documents = []
with open(sample_file, 'r', encoding='utf-8') as f:
    content = f.read()
    
    # Create LangChain Document with metadata
    # Add required fields based on Vector Index schema
    doc = Document(
        page_content=content,
        metadata={
            # "filename": sample_file.name,
            "source": str(sample_file),
            "page": 0,  # Required by Vector Index schema
            "document_name": sample_file.name
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents")

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"✓ Created {len(chunks)} chunks")

# %% Cell 8: Add Documents to Vector Store
print("\n7. Adding documents to vector store...")
try:
    vector_store.add_documents(content=chunks, batch_size=20)
    print(f"✓ Added {len(chunks)} documents to vector store")
    
    # Get count
    count = vector_store.count()
    print(f"\n8. Vector store statistics:")
    print(f"  Total documents: {count}")
    
    # Test search
    print(f"\n9. Testing search functionality...")
    test_query = "What is Watsonx?"
    results = vector_store.search(query=test_query, k=2)
    print(f"✓ Search test successful!")
    print(f"  Query: {test_query}")
    print(f"  Results found: {len(results)}")
    if results:
        print(f"\n  First result preview:")
        print(f"    Content: {results[0].page_content[:100]}...")
        print(f"    Metadata: {results[0].metadata}")
        print(f"    document_name field: {results[0].metadata.get('document_name', 'NOT FOUND')}")
        
except Exception as e:
    print(f"⚠ Document addition/search failed: {e}")
    print(f"  Error details: {type(e).__name__}")

# %% Cell 9: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 Next steps:")
print("  1. Vector Index has been populated in IBM watsonx platform")
print("  2. Use example 04 to query the vector index")
print("  3. Check your project Assets to see the Vector Index")
