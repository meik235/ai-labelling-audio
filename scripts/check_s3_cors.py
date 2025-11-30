#!/usr/bin/env python3
"""Check S3 bucket CORS configuration for Label Studio compatibility."""

import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

bucket_name = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET", "kgen-ai-labelling")
region = os.getenv("AWS_REGION", "ap-south-1")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
label_studio_url = os.getenv("LABEL_STUDIO_BASE_URL", "http://localhost:8080")

if not access_key or not secret_key:
    print("Error: AWS credentials not found in environment")
    exit(1)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
)

try:
    cors_config = s3_client.get_bucket_cors(Bucket=bucket_name)
    print(f"✓ CORS is configured for bucket: {bucket_name}")
    print(f"\nCurrent CORS rules:")
    for i, rule in enumerate(cors_config.get("CORSRules", []), 1):
        print(f"\n  Rule {i}:")
        print(f"    AllowedOrigins: {rule.get('AllowedOrigins', [])}")
        print(f"    AllowedMethods: {rule.get('AllowedMethods', [])}")
        print(f"    AllowedHeaders: {rule.get('AllowedHeaders', [])}")
        print(f"    ExposeHeaders: {rule.get('ExposeHeaders', [])}")
        print(f"    MaxAgeSeconds: {rule.get('MaxAgeSeconds', 'Not set')}")
    
    # Check if Label Studio URL is allowed
    allowed_origins = []
    for rule in cors_config.get("CORSRules", []):
        allowed_origins.extend(rule.get("AllowedOrigins", []))
    
    # Check if the Label Studio origin is allowed
    ls_origin = label_studio_url.rstrip("/")
    ls_origin_wildcard = ls_origin.replace("://", "://*")
    is_allowed = (
        "*" in allowed_origins 
        or ls_origin in allowed_origins 
        or ls_origin_wildcard in allowed_origins
        or any(ls_origin.startswith(o.rstrip("*")) for o in allowed_origins if "*" in o)
    )
    
    if is_allowed:
        print(f"\n✓ {ls_origin} should be allowed")
    else:
        print(f"\n⚠ {ls_origin} is NOT in allowed origins")
        print(f"  Current origins: {allowed_origins}")
        print(f"\n  To fix, add this CORS rule to your S3 bucket:")
        print("  {")
        print('  "CORSRules": [')
        print('    {')
        print(f'      "AllowedOrigins": ["{ls_origin}", "{ls_origin_wildcard}"],')
        print('      "AllowedMethods": ["GET", "HEAD"],')
        print('      "AllowedHeaders": ["*"],')
        print('      "ExposeHeaders": ["ETag"],')
        print('      "MaxAgeSeconds": 3000')
        print('    }')
        print('  ]')
        print("  }")
        
except ClientError as e:
    if e.response.get("Error", {}).get("Code") == "NoSuchCORSConfiguration":
        print(f"✗ CORS is NOT configured for bucket: {bucket_name}")
        print(f"\nThis is why Label Studio can't load audio files!")
        print(f"\nTo fix, configure CORS on your S3 bucket:")
        print(f"\n1. Go to AWS Console → S3 → {bucket_name} → Permissions → CORS")
        print(f"\n2. Add this CORS configuration:")
        cors_config_example = f"""
[
    {{
        "AllowedOrigins": [
            "{ls_origin}",
            "{ls_origin_wildcard}",
            "https://your-labelstudio-domain.com"
        ],
        "AllowedMethods": [
            "GET",
            "HEAD"
        ],
        "AllowedHeaders": [
            "*"
        ],
        "ExposeHeaders": [
            "ETag",
            "Content-Length"
        ],
        "MaxAgeSeconds": 3000
    }}
]
"""
        print(cors_config_example)
        print(f"\n3. Or use AWS CLI:")
        print(f"""
aws s3api put-bucket-cors --bucket {bucket_name} --cors-configuration file://cors.json
""")
        print(f"\n4. Or use boto3 in Python:")
        boto3_example = f"""
import boto3
s3 = boto3.client('s3')
s3.put_bucket_cors(
    Bucket='{bucket_name}',
    CORSConfiguration={{
        'CORSRules': [
            {{
                'AllowedOrigins': ['{ls_origin}', '{ls_origin_wildcard}'],
                'AllowedMethods': ['GET', 'HEAD'],
                'AllowedHeaders': ['*'],
                'ExposeHeaders': ['ETag', 'Content-Length'],
                'MaxAgeSeconds': 3000
            }}
        ]
    }}
)
"""
        print(boto3_example)
    else:
        raise
except Exception as e:
    print(f"Error checking CORS: {e}")

