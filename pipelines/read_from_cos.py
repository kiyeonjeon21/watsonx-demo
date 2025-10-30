"""
Read from COS Sample for Pipeline
COS에 저장된 파일을 읽어오는 간단한 스크립트
와일드카드(*) 패턴 지원
참고: https://ibm.github.io/watsonx-ai-python-sdk/v1.4.4/dataconnection_modules.html
"""

import os
import fnmatch
from pathlib import Path
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.helpers.connections import DataConnection, S3Location

load_dotenv()

# 기본 설정
WATSONX_API_KEY = os.getenv('WATSONX_API_KEY')
WATSONX_PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
WATSONX_URL = os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')

# COS 설정
COS_BUCKET = os.getenv('COS_BUCKET')
COS_PREFIX = os.getenv('COS_PREFIX', 'chat-responses')
COS_CONNECTION_ASSET_ID = os.getenv('COS_CONNECTION_ASSET_ID')

# 파일 패턴 (와일드카드 * 사용 가능, 예: '*.txt', 'response_*.txt')
COS_FILE_PATTERN = os.getenv('COS_FILE_PATTERN', '*.txt')

try:
    if not all([COS_BUCKET, COS_CONNECTION_ASSET_ID]):
        print("⚠ Missing required configuration:")
        print("  Required: COS_BUCKET, COS_CONNECTION_ASSET_ID")
        exit(1)
    
    # Credentials 및 Client 초기화
    credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)
    client = APIClient(credentials)
    client.set.default_project(project_id=WATSONX_PROJECT_ID)
    
    print("="*60)
    print("Reading from COS")
    print("="*60)
    print(f"Bucket: {COS_BUCKET}")
    print(f"Prefix: {COS_PREFIX}")
    print(f"Pattern: {COS_FILE_PATTERN}")
    
    # 패턴에서 폴더 경로 추출
    if '/' in COS_FILE_PATTERN:
        # 패턴에 경로가 포함된 경우 (예: 'data/raw/*.txt')
        folder_path = '/'.join(COS_FILE_PATTERN.split('/')[:-1])
        file_pattern = COS_FILE_PATTERN.split('/')[-1]
        full_folder_path = f"{COS_PREFIX}/{folder_path}" if not folder_path.startswith(COS_PREFIX) else folder_path
    else:
        # 패턴만 있는 경우 (예: '*.txt')
        file_pattern = COS_FILE_PATTERN
        # chat_invoke.py에서 저장한 기본 경로 사용
        full_folder_path = f"{COS_PREFIX}/data/raw"
    
    print(f"\nDownloading folder: {full_folder_path}")
    print(f"Searching for files matching: {file_pattern}")
    
    # 로컬 다운로드 디렉토리 준비
    download_dir = Path("data/processed")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # DataConnection으로 폴더 다운로드
    folder_connection = DataConnection(
        connection_asset_id=COS_CONNECTION_ASSET_ID,
        location=S3Location(
            bucket=COS_BUCKET,
            path=full_folder_path
        )
    )
    folder_connection.set_client(api_client=client)
    
    # 폴더 다운로드
    try:
        folder_connection.download_folder(local_dir=str(download_dir))
        print(f"✓ Folder downloaded to: {download_dir}")
    except Exception as download_error:
        print(f"⚠ Folder download failed: {download_error}")
        print(f"   Trying to search in subdirectories...")
        # 하위 폴더까지 검색
        download_dir = download_dir / full_folder_path.replace('/', '_')
        download_dir.mkdir(parents=True, exist_ok=True)
        try:
            folder_connection.download_folder(local_dir=str(download_dir))
            print(f"✓ Folder downloaded to: {download_dir}")
        except Exception as e2:
            print(f"✗ Could not download folder: {e2}")
            exit(1)
    
    # 다운로드된 파일들 중 패턴 매칭
    matching_files = []
    for file_path in download_dir.rglob('*'):
        if file_path.is_file():
            file_name = file_path.name
            if fnmatch.fnmatch(file_name, file_pattern):
                matching_files.append(file_path)
    
    if not matching_files:
        print(f"⚠ No files matching pattern: {file_pattern}")
        print(f"   Searched in: {download_dir}")
        exit(0)
    
    print(f"\n✓ Found {len(matching_files)} matching file(s)\n")
    
    # 매칭된 파일들 읽어서 출력
    for file_path in matching_files:
        print("-" * 60)
        print(f"File: {file_path.relative_to(download_dir)}")
        print("-" * 60)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)
        print()
    
    print(f"✓ Processed {len(matching_files)} file(s)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
