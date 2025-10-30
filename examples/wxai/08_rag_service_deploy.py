"""
Example 08: RAG Service Deployment
Demonstrates deploying a RAG (Retrieval Augmented Generation) service using IBM watsonx SDK.
Based on: samples/rag service deploy sample.ipynb

Self-Contained Design:
- Vector index promotion
- RAG service function definition with grounding
- Local testing
- Deployment to watsonx
- Testing deployed service
- Use # %% cell markers for easy conversion to notebook
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
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.deployments import RuntimeContext
from dotenv import load_dotenv

# %% [markdown]
# # Example 08: RAG Service Deployment
# This demonstrates how to deploy a RAG service with vector index grounding.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
WATSONX_SPACE_ID = os.getenv('WATSONX_SPACE_ID_DEV')
VECTORIZED_DOCUMENT_ASSET_ID = os.getenv('VECTORIZED_DOCUMENT_ASSET_ID')  # Vector index ID to promote

print("="*60)
print("EXAMPLE 08: RAG Service Deployment")
print("="*60)

# %% Cell 2: Initialize Credentials
print("\n1. Initializing credentials...")

credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY
)

client = APIClient(credentials)
print(f"✓ Credentials initialized")
print(f"  Project ID: {WATSONX_PROJECT_ID}")

# %% Cell 3: Connect to Space
print("\n2. Connecting to space...")

if WATSONX_SPACE_ID:
    client.set.default_space(WATSONX_SPACE_ID)
    print(f"✓ Connected to space: {WATSONX_SPACE_ID}")
else:
    print("❌ ERROR: WATSONX_SPACE_ID not set")
    raise ValueError("WATSONX_SPACE_ID is required to deploy the AI service. Please set it in your .env file.")

# %% Cell 4: Promote Vector Index (Optional)
print("\n3. Promoting vector index...")

vector_index_id = None
if VECTORIZED_DOCUMENT_ASSET_ID and WATSONX_PROJECT_ID:
    try:
        vector_index_id = client.spaces.promote(
            VECTORIZED_DOCUMENT_ASSET_ID,
            WATSONX_PROJECT_ID,
            WATSONX_SPACE_ID
        )
        print(f"✓ Vector index promoted")
        print(f"  Vector Index ID: {vector_index_id}")
    except Exception as e:
        print(f"⚠ Failed to promote vector index: {e}")
        print("  Continuing without vector index...")
else:
    print("⚠ No vector index ID provided")
    print("  Using hardcoded vector index ID from params")

# %% Cell 5: Define RAG Service Function
print("\n4. Defining RAG service function...")

# Define service parameters
params = {
    "space_id": WATSONX_SPACE_ID,
    "vector_index_id": vector_index_id
}

def gen_ai_service(context, params=params, **custom):
    """RAG service function with vector index grounding."""
    
    # Merge custom kwargs into params
    if custom:
        params = {**params, **custom}
    
    # Import dependencies
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from ibm_watsonx_ai import APIClient, Credentials
    import json
    import requests
    import re
    
    space_id = params.get("space_id")
    vector_index_id = params.get("vector_index_id")
    
    def proximity_search(query, api_client):
        """Perform RAG query using proximity search."""
        document_search_tool = Toolkit(
            api_client=api_client
        ).get_tool("RAGQuery")
        
        config = {
            "vectorIndexId": vector_index_id,
            "spaceId": space_id
        }
        
        try:
            results = document_search_tool.run(input=query, config=config)
            return results.get("output", "")
        except Exception as e:
            print(f"Search error: {e}")
            return ""
    
    def get_api_client(context):
        """Get API client from context."""
        credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            token=context.get_token()
        )
        
        api_client = APIClient(
            credentials=credentials,
            space_id=space_id
        )
        
        return api_client
    
    def inference_model(messages, context, stream):
        """Run model inference with grounding."""
        query = messages[-1].get("content")
        api_client = get_api_client(context)
        
        grounding_context = proximity_search(query, api_client)
        
        grounding = grounding_context
        
        # Insert system message with grounding context
        messages.insert(0, {
            "role": "system",
            "content": """You always answer the questions with markdown formatting. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.

Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.

When returning code blocks, specify language.

Given the document and the current conversation between a user and an assistant, your task is as follows: answer any user query by using information from the document. Always answer as helpfully as possible, while being safe. When the question cannot be answered using the context or document, output the following response: "I cannot answer that question based on the provided document.".

Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

### Context:
{grounding}

""".format(grounding=grounding)
        })
        
        model_id = "meta-llama/llama-3-2-90b-vision-instruct"
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }
        
        model = ModelInference(
            model_id=model_id,
            api_client=api_client,
            params=parameters
        )
        
        # Generate grounded response
        if stream:
            generated_response = model.chat_stream(messages=messages)
        else:
            generated_response = model.chat(messages=messages)
        
        return generated_response
    
    def generate(context):
        """Generate non-streaming response."""
        payload = context.get_json()
        messages = payload.get("messages")
        
        # Grounded inferencing
        generated_response = inference_model(messages, context, False)
        
        execute_response = {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": generated_response
        }
        
        return execute_response
    
    def generate_stream(context):
        """Generate streaming response."""
        payload = context.get_json()
        messages = payload.get("messages")
        
        # Grounded inferencing
        response_stream = inference_model(messages, context, True)
        
        for chunk in response_stream:
            yield chunk
    
    return generate, generate_stream

print("✓ RAG service function defined")

# %% Cell 6: Test Locally (Optional)
print("\n5. Testing RAG service locally (optional)...")

if WATSONX_SPACE_ID:
    try:
        context = RuntimeContext(api_client=client)
        
        # Get the non-streaming function (index 0)
        streaming = False
        findex = 1 if streaming else 0
        local_function = gen_ai_service(
            context,
            vector_index_id=vector_index_id,
            space_id=WATSONX_SPACE_ID
        )[findex]
        
        # Test with a simple query
        messages = [{"role": "user", "content": "What is watsonx?"}]
        
        test_context = RuntimeContext(
            api_client=client,
            request_payload_json={"messages": messages}
        )
        
        # print(f"  Testing with query: Change this question to test your function")
        
        response = local_function(test_context)
        
        if streaming:
            print("\n  Streaming response:")
            for chunk in response:
                if len(chunk.get("choices", [])):
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
        else:
            print("\n  Response:")
            print(f"  {response}")
        
        print("✓ Local test completed")
        
    except Exception as e:
        print(f"⚠ Local test failed: {e}")
        print("  Continuing to deployment steps...")
else:
    print("  Skipping local test (no space ID)")

# %% Cell 7: Store AI Service
print("\n6. Storing AI service...")

try:
    # Look up software specification
    software_spec_id = None
    try:
        software_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")
        print(f"  Using software spec: runtime-24.1-py3.11")
    except:
        print("  ⚠ Could not find software specification")
    
    # Define request and response schemas
    request_schema = {
        "application/json": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "messages": {
                    "title": "The messages for this chat session.",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "title": "The role of the message author.",
                                "type": "string",
                                "enum": ["user", "assistant"]
                            },
                            "content": {
                                "title": "The contents of the message.",
                                "type": "string"
                            }
                        },
                        "required": ["role", "content"]
                    }
                }
            },
            "required": ["messages"]
        }
    }
    
    response_schema = {
        "application/json": {
            "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
        }
    }
    
    if software_spec_id:
        # Store the AI service
        ai_service_metadata = {
            client.repository.AIServiceMetaNames.NAME: "RAG Service Deploy Sample",
            client.repository.AIServiceMetaNames.DESCRIPTION: "RAG service with vector index grounding",
            client.repository.AIServiceMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
            client.repository.AIServiceMetaNames.CUSTOM: {},
            client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema,
            client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema,
            client.repository.AIServiceMetaNames.TAGS: ["wx-vector-index"]
        }
        
        ai_service_details = client.repository.store_ai_service(
            meta_props=ai_service_metadata,
            ai_service=gen_ai_service
        )
        
        ai_service_id = client.repository.get_ai_service_id(ai_service_details)
        print(f"✓ AI service stored")
        print(f"  Service ID: {ai_service_id}")
        
    else:
        print("⚠ Skipping storage (no software spec)")
        ai_service_id = None
        
except Exception as e:
    print(f"⚠ Failed to store AI service: {e}")
    ai_service_id = None

# %% Cell 8: Deploy AI Service
print("\n7. Deploying AI service...")

if ai_service_id:
    try:
        deployment_custom = {}
        
        deployment_metadata = {
            client.deployments.ConfigurationMetaNames.NAME: "RAG Service Deploy Sample",
            client.deployments.ConfigurationMetaNames.ONLINE: {},
            client.deployments.ConfigurationMetaNames.CUSTOM: deployment_custom,
            client.deployments.ConfigurationMetaNames.DESCRIPTION: "",
            client.repository.AIServiceMetaNames.TAGS: ["wx-vector-index"]
        }
        
        function_deployment_details = client.deployments.create(
            ai_service_id,
            meta_props=deployment_metadata,
            space_id=WATSONX_SPACE_ID
        )
        
        deployment_id = client.deployments.get_id(function_deployment_details)
        print(f"✓ AI service deployed")
        print(f"  Deployment ID: {deployment_id}")
        
    except Exception as e:
        print(f"⚠ Failed to deploy AI service: {e}")
        deployment_id = None
else:
    print("  Skipping deployment (no service ID)")
    deployment_id = None

# %% Cell 9: Test Deployed Service
print("\n8. Testing deployed service...")

if deployment_id:
    try:
        messages = [{"role": "user", "content": "What is watsonx?"}]
        payload = {"messages": messages}
        
        # print(f"  Query: Change this question to test your function")
        
        result = client.deployments.run_ai_service(deployment_id, payload)
        
        if "error" in result:
            print(f"  ⚠ Error: {result['error']}")
        else:
            response = result.get("body", {})
            choices = response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"  ✓ Response received:")
                print(f"  {content[:200]}...")
            else:
                print(f"  Response: {response}")
        
    except Exception as e:
        print(f"  ⚠ Test failed: {e}")
else:
    print("  Skipping test (no deployment)")

# %% Cell 10: Summary
print("\n✓ Complete!")
print("="*60)
print("\n📝 RAG Service capabilities:")
print("  ✓ Vector index grounding")
print("  ✓ Proximity search with RAGQuery")
print("  ✓ Context-aware responses")
print("  ✓ Streaming support")
print("\n💡 To use the deployed service:")
print("  result = client.deployments.run_ai_service(deployment_id, payload)")
print("\n📋 Deployment Information:")
if deployment_id:
    print(f"  Deployment ID: {deployment_id}")
    print(f"  Service ID: {ai_service_id}")
    print(f"  Space ID: {WATSONX_SPACE_ID}")
    if vector_index_id:
        print(f"  Vector Index ID: {vector_index_id}")

