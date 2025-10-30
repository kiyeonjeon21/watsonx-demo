"""
Example 02: Prompt Template Manager
Demonstrates prompt template management using IBM watsonx SDK.
Based on: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/prompt_template_manager.html

Self-Contained Design:
- Prompt template creation, storage, and management
- Template deployment and usage
- Use # %% cell markers for Jupyter notebook conversion
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.prompts import PromptTemplate, PromptTemplateManager
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.foundation_models.utils.enums import PromptTemplateFormats

# %% [markdown]
# # Example 08: Prompt Template Manager
# This demonstrates how to create, store, and manage prompt templates.

# %% Cell 1: Setup
load_dotenv()

WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_SPACE_ID = os.getenv('WATSONX_SPACE_ID_DEV')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-3-3-70b-instruct')

print("="*60)
print("EXAMPLE 08: Prompt Template Manager")
print("="*60)
print("\n1. Configuration:")
print(f"  API Key: {WATSONX_API_KEY[:10]}..." if WATSONX_API_KEY else "  API Key: Not set")
print(f"  Project ID: {WATSONX_PROJECT_ID}")
print(f"  Space ID: {WATSONX_SPACE_ID}")
print(f"  Model: {MODEL_NAME}")

# %% Cell 2: Initialize Prompt Template Manager
print("\n2. Initializing Prompt Template Manager...")

credentials = Credentials(
    api_key=WATSONX_API_KEY,
    url=WATSONX_URL
)

prompt_mgr = PromptTemplateManager(
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID
)

print("✓ Prompt Template Manager initialized")

# %% Cell 3: Create a Prompt Template
print("\n3. Creating a prompt template...")

# Create a prompt template with input variables
prompt_template = PromptTemplate(
    name="Educational Q&A Template",
    model_id=MODEL_NAME,
    input_prefix="Human:",
    output_prefix="Assistant:",
    input_text="What is {object} and how does it work?",
    input_variables=['object'],
    description="A template for educational questions about objects",
    examples=[
        ['What is the Stock Market?', 
         'A stock market is a place where investors buy and sell shares of publicly traded companies.']
    ]
)

print(f"✓ Template created: {prompt_template.name}")
print(f"  Input variables: {prompt_template.input_variables}")

# %% Cell 4: Store the Prompt Template
print("\n4. Storing the prompt template...")

try:
    stored_prompt_template = prompt_mgr.store_prompt(prompt_template)
    prompt_id = stored_prompt_template.prompt_id
    print(f"✓ Template stored successfully")
    print(f"  Prompt ID: {prompt_id}")
except Exception as e:
    print(f"⚠ Error storing template: {e}")
    # For demo purposes, use a placeholder if storage fails
    prompt_id = None

# %% Cell 5: List All Prompts
print("\n5. Listing all prompt templates...")

try:
    df_prompts = prompt_mgr.list(limit=10)
    print(f"✓ Found {len(df_prompts)} prompt templates")
    if len(df_prompts) > 0:
        print("\n  Recent prompts:")
        for idx, row in df_prompts.head(5).iterrows():
            print(f"    - {row.get('NAME', 'Unknown')}: {row.get('ID', 'Unknown')}")
except Exception as e:
    print(f"⚠ Error listing prompts: {e}")

# %% Cell 6: Load a Prompt Template
print("\n6. Loading prompt template...")

if prompt_id:
    try:
        loaded_template = prompt_mgr.load_prompt(
            prompt_id=prompt_id,
            astype=PromptTemplateFormats.PROMPTTEMPLATE
        )
        print(f"✓ Template loaded: {loaded_template.name}")
        print(f"  Model ID: {loaded_template.model_id}")
        print(f"  Input variables: {loaded_template.input_variables}")
    except Exception as e:
        print(f"⚠ Error loading template: {e}")
else:
    print("⚠ Cannot load - no prompt ID available")

# %% Cell 7: Update the Prompt Template
print("\n7. Updating prompt template...")

if prompt_id:
    try:
        # Update the template (e.g., modify the input text)
        updated_template = PromptTemplate(
            name="Educational Q&A Template",
            model_id=MODEL_NAME,
            input_prefix="Human:",
            output_prefix="Assistant:",
            input_text="Explain what {object} is and provide key details about how it functions.",
            input_variables=['object'],
            description="An updated template for educational questions about objects with more detail",
            examples=[
                ['What is the Stock Market?', 
                 'A stock market is a place where investors buy and sell shares of publicly traded companies.'],
                ['What is Machine Learning?',
                 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.']
            ]
        )
        
        updated_prompt = prompt_mgr.update_prompt(
            prompt_id=prompt_id,
            prompt_template=updated_template
        )
        print(f"✓ Template updated successfully")
    except Exception as e:
        print(f"⚠ Error updating template: {e}")
else:
    print("⚠ Cannot update - no prompt ID available")

# %% Cell 8: Deploy and Use the Template
print("\n8. Deploying and using the template...")

if prompt_id:
    try:
        # Initialize API client
        api_client = APIClient(credentials)
        api_client.set.default_project(project_id=WATSONX_PROJECT_ID)
        
        # Deploy the prompt template
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        
        meta_props = {
            api_client.deployments.ConfigurationMetaNames.NAME: "Educational Q&A Deployment",
            api_client.deployments.ConfigurationMetaNames.ONLINE: {},
            api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: MODEL_NAME
        }
        
        deployment_details = api_client.deployments.create(prompt_id, meta_props)
        deployment_id = deployment_details["metadata"]["id"]
        print(f"✓ Template deployed successfully")
        print(f"  Deployment ID: {deployment_id}")
        
        # Use the deployed template with variables
        print("\n  Testing the deployed template...")
        
        result = api_client.deployments.generate_text(
            deployment_id=deployment_id,
            params={
                GenParams.PROMPT_VARIABLES: {
                    "object": "artificial intelligence"
                }
            }
        )
        
        # Extract the generated text
        if 'results' in result and len(result['results']) > 0:
            generated_text = result['results'][0].get('generated_text', str(result['results'][0]))
            print(f"\n  ✓ Query: What is artificial intelligence and how does it work?")
            print(f"\n  Response: {generated_text[:200]}...")
        
    except Exception as e:
        print(f"⚠ Error deploying or using template: {e}")
        print("  (Deployment may require additional permissions)")
else:
    print("⚠ Cannot deploy - no prompt ID available")

# %% Cell 9: Get Lock Status
print("\n9. Checking lock status...")

if prompt_id:
    try:
        lock_info = prompt_mgr.get_lock(prompt_id)
        print(f"✓ Lock status retrieved")
        print(f"  Lock info: {lock_info}")
    except Exception as e:
        print(f"⚠ Error getting lock: {e}")
else:
    print("⚠ Cannot check lock - no prompt ID available")

# %% Cell 10: Space-Based Deployment
print("\n10. Deploying to Space (WATSONX_SPACE_ID)...")

if WATSONX_SPACE_ID and prompt_id:
    try:
        # Initialize Prompt Template Manager for Space
        print("  Initializing Space-based Prompt Template Manager...")
        space_prompt_mgr = PromptTemplateManager(
            credentials=credentials,
            space_id=WATSONX_SPACE_ID
        )
        print("✓ Space Prompt Template Manager initialized")
        
        # Store the template in the space
        print("\n  Storing template in space...")
        space_stored_template = space_prompt_mgr.store_prompt(prompt_template)
        space_prompt_id = space_stored_template.prompt_id
        print(f"✓ Template stored in space successfully")
        print(f"  Space Prompt ID: {space_prompt_id}")
        
        # Deploy the template in space
        print("\n  Deploying template in space...")
        space_api_client = APIClient(credentials)
        space_api_client.set.default_space(space_id=WATSONX_SPACE_ID)
        
        space_meta_props = {
            space_api_client.deployments.ConfigurationMetaNames.NAME: "Educational Q&A Space Deployment",
            space_api_client.deployments.ConfigurationMetaNames.ONLINE: {},
            space_api_client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: MODEL_NAME
        }
        
        space_deployment_details = space_api_client.deployments.create(space_prompt_id, space_meta_props)
        space_deployment_id = space_deployment_details["metadata"]["id"]
        print(f"✓ Template deployed to space successfully")
        print(f"  Space Deployment ID: {space_deployment_id}")
        
        # Test the space deployment
        print("\n  Testing space deployment...")
        space_result = space_api_client.deployments.generate_text(
            deployment_id=space_deployment_id,
            params={
                GenParams.PROMPT_VARIABLES: {
                    "object": "quantum computing"
                }
            }
        )
        
        if 'results' in space_result and len(space_result['results']) > 0:
            space_generated_text = space_result['results'][0].get('generated_text', str(space_result['results'][0]))
            print(f"\n  ✓ Query (Space): What is quantum computing and how does it work?")
            print(f"\n  Response: {space_generated_text[:200]}...")
        
    except Exception as e:
        print(f"⚠ Error in space deployment: {e}")
        print("  (Space deployment may require additional permissions)")
        space_prompt_id = None
        space_deployment_id = None
else:
    if not WATSONX_SPACE_ID:
        print("⚠ WATSONX_SPACE_ID not configured - skipping space deployment")
    if not prompt_id:
        print("⚠ Cannot deploy to space - no prompt ID available")
    space_prompt_id = None
    space_deployment_id = None

# %% Cell 11: Summary and Cleanup
print("\n11. Summary...")

print("\n✓ Prompt Template Manager demonstration complete!")
print("\n📝 Key features demonstrated:")
print("  ✓ Create prompt templates with input variables")
print("  ✓ Store templates in the project")
print("  ✓ List all stored templates")
print("  ✓ Load and use templates")
print("  ✓ Update existing templates")
print("  ✓ Deploy and use templates with variable substitution")
print("  ✓ Deploy to both project and space")
print("  ✓ Test deployments in different contexts")
print("  ✓ Check template lock status")

if prompt_id:
    print(f"\n⚠ Note: Template with ID '{prompt_id}' is still stored in project.")
    print("   To delete it manually, use:")
    print(f"   prompt_mgr.delete_prompt('{prompt_id}')")

if WATSONX_SPACE_ID:
    try:
        if space_prompt_id:
            print(f"\n⚠ Note: Template with ID '{space_prompt_id}' is still stored in space.")
            print("   To delete it manually, use:")
            print(f"   space_prompt_mgr.delete_prompt('{space_prompt_id}')")
    except NameError:
        pass  # space_prompt_id was not created

print("\n💡 To manually test cleanup:")
print("   # Uncomment to delete the templates")
if prompt_id:
    print(f"   # prompt_mgr.delete_prompt('{prompt_id}')")
if WATSONX_SPACE_ID:
    try:
        if space_prompt_id:
            print(f"   # space_prompt_mgr.delete_prompt('{space_prompt_id}')")
    except NameError:
        pass

print("="*60)

