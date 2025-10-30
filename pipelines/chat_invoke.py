"""
Chat Invoke Sample for Pipeline
chat 모듈을 사용하여 prompt에 대해 invoke로 결과를 반환하는 간단한 스크립트
"""

import os
import datetime
from pathlib import Path
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.helpers.connections import DataConnection, S3Location

load_dotenv()

# 기본 설정
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_SPACE_ID = os.getenv('WATSONX_SPACE_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-3-3-70b-instruct')

# COS 설정 (선택적)
COS_BUCKET = os.getenv('COS_BUCKET')
COS_PREFIX = os.getenv('COS_PREFIX', 'chat-responses')
COS_CONNECTION_ASSET_ID = os.getenv('COS_CONNECTION_ASSET_ID')

# Prompt 설정
PROMPT = os.getenv('PROMPT', 'What is artificial intelligence?')

# Credentials 설정
credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)

try:
    # ModelInference 초기화
    model = ModelInference(
        model_id=MODEL_NAME,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID if not WATSONX_SPACE_ID else None,
        space_id=WATSONX_SPACE_ID if WATSONX_SPACE_ID else None
    )
    
    # Chat 메시지 구성
    messages = [
        {"role": "user", "content": PROMPT}
    ]
    
    # Chat invoke 실행
    response = model.chat(
        messages=messages,
        params={
            "max_new_tokens": 1000,
            "temperature": 0.0
        }
    )
    
    # Response에서 실제 응답 텍스트만 추출
    if isinstance(response, dict) and 'choices' in response:
        content = response['choices'][0]['message']['content']
        print(content)
    else:
        content = str(response)
        print(content)
    
    # Response를 txt 파일로 저장
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일명 생성 (타임스탬프 + 간단한 prompt 해시 또는 기본 이름)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Prompt의 첫 30자를 파일명에 사용 (특수문자 제거)
    prompt_safe = "".join(c for c in PROMPT[:30] if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    output_file = data_dir / f"response_{timestamp}_{prompt_safe}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {PROMPT}\n\n")
        f.write(f"Response:\n{content}\n")
    
    print(f"\n✓ Response saved to: {output_file}")
    
    # COS에 업로드 (설정된 경우)
    if not all([COS_BUCKET, COS_CONNECTION_ASSET_ID]):
        if any([COS_BUCKET, COS_CONNECTION_ASSET_ID]):
            print("⚠ COS configuration incomplete. Skipping upload.")
            print("  Required: COS_BUCKET and COS_CONNECTION_ASSET_ID")
    else:
        try:
            client = APIClient(credentials)
            client.set.default_project(project_id=WATSONX_PROJECT_ID)
            
            abs_output_file = output_file.resolve()
            relative_path = abs_output_file.relative_to(Path.cwd().resolve())
            
            data_connection = DataConnection(
                connection_asset_id=COS_CONNECTION_ASSET_ID,
                location=S3Location(
                    bucket=COS_BUCKET,
                    path=f"{COS_PREFIX}/{relative_path}"
                )
            )
            
            data_connection.set_client(api_client=client)
            
            print(f"Uploading to COS: {COS_BUCKET}/{COS_PREFIX}/{relative_path}...")
            data_connection.write(data=str(abs_output_file))
            print(f"✓ Uploaded to COS successfully!")
            
        except Exception as e:
            print(f"⚠ Upload failed: {e}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

