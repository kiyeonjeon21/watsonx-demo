"""
Example 04: Upload to IBM COS
Demonstrates uploading files to IBM Cloud Object Storage using IBM watsonx SDK.
Based on: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.2/dataconnection_modules.html

Self-Contained Design:
- All code inline, no local package imports
- Use # %% cell markers for easy conversion to notebook
"""

import os
from pathlib import Path
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.helpers.connections import DataConnection, S3Location
from dotenv import load_dotenv

# %% [markdown]
# # Example 04: Upload to IBM COS
# This script demonstrates uploading files to IBM Cloud Object Storage using IBM watsonx SDK.

# %% Cell 1: Setup
# Load environment variables
load_dotenv()

# Get credentials
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

# Configuration
COS_BUCKET = os.getenv('COS_BUCKET')
COS_PREFIX = os.getenv('COS_PREFIX', 'datasets')
COS_CONNECTION_ASSET_ID = os.getenv('COS_CONNECTION_ASSET_ID')

print("="*60)
print("EXAMPLE 04: Upload to IBM COS")
print("="*60)

# %% Cell 2: Initialize Client
print("\n1. Initializing SDK client...")
try:
    sdk_credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    client = APIClient(sdk_credentials)
    client.set.default_project(project_id=WATSONX_PROJECT_ID)
    print(f"✓ Client initialized")
except Exception as e:
    print(f"⚠ Client initialization failed: {e}")
    exit(1)

# %% Cell 3: Create Sample Data
print("\n2. Creating sample data...")
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Create a sample file
sample_file = data_dir / "sample.txt"
with open(sample_file, 'w') as f:
    f.write("This is a sample document for testing.")

print(f"✓ Created sample file: {sample_file}")

# %% Cell 4: Upload to COS
print("\n3. Uploading to IBM COS...")
print("\nConfiguration:")
print(f"  Bucket: {COS_BUCKET}")
print(f"  Prefix: {COS_PREFIX}")
print(f"  File: {sample_file.name}")

if not all([COS_BUCKET, COS_CONNECTION_ASSET_ID]):
    print("\n⚠ Missing required configuration:")
    print("  Required environment variables:")
    print("    - COS_BUCKET")
    print("    - COS_CONNECTION_ASSET_ID")
    print("\n  To upload to COS:")
    print("  1. Create a Cloud Object Storage connection in watsonx.ai")
    print("  2. Get the connection asset ID")
    print("  3. Set COS_BUCKET and COS_CONNECTION_ASSET_ID in your .env file")
    print("\n  Skipping upload for now...")
else:
    try:
        # Get relative path to maintain local directory structure
        # Convert to absolute if needed, then get relative to cwd
        abs_sample_file = sample_file.resolve()
        relative_path = abs_sample_file.relative_to(Path.cwd().resolve())
        
        # Create DataConnection with S3Location - preserve local path structure
        data_connection = DataConnection(
            connection_asset_id=COS_CONNECTION_ASSET_ID,
            location=S3Location(
                bucket=COS_BUCKET,
                path=f"{COS_PREFIX}/{relative_path}"  # datasets/data/raw/sample.txt
            )
        )
        
        # Set API client
        data_connection.set_client(api_client=client)
        
        # Upload the file
        print(f"\n  Uploading {relative_path}...")
        data_connection.write(
            data=str(abs_sample_file)  # Local file absolute path
        )
        
        print(f"✓ Uploaded to: {COS_BUCKET}/{COS_PREFIX}/{relative_path}")
        print(f"✓ Upload successful!")
        
    except Exception as e:
        print(f"⚠ Upload failed: {e}")
        print("  This might be due to:")
        print("    - Incorrect connection asset ID")
        print("    - Insufficient permissions")
        print("    - Bucket does not exist")
        print("    - Network connectivity issues")

print("\n✓ Upload process complete!")
print("="*60)
