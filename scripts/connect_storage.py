#!/usr/bin/env python3
"""Connect cloud storage to an existing Label Studio project."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
except ImportError:
    pass

from backend.integrations import LabelStudioClient


async def connect_storage(
    project_id: int,
    bucket: str,
    prefix: str = "",
    region: str = "",
    storage_type: str = "s3",
) -> None:
    """Connect cloud storage to a Label Studio project."""
    base_url = os.getenv("LABEL_STUDIO_BASE_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    
    if not base_url or not api_key:
        print("Error: LABEL_STUDIO_BASE_URL and LABEL_STUDIO_API_KEY must be set in .env file")
        sys.exit(1)
    
    # Auto-detect from env if not provided
    if not bucket:
        bucket = os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
        if not bucket:
            print("Error: Bucket name required. Set --bucket or S3_BUCKET env var")
            sys.exit(1)
    
    if not region:
        region = os.getenv("AWS_REGION", "ap-south-1")
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    client = LabelStudioClient(base_url, api_key)
    
    print(f"\nConnecting cloud storage to project {project_id}")
    print(f"  - Type: {storage_type}")
    print(f"  - Bucket: {bucket}")
    if prefix:
        print(f"  - Prefix: {prefix}")
    print(f"  - Region: {region}")
    
    try:
        result = await client.configure_cloud_storage(
            project_id=project_id,
            storage_type=storage_type,
            bucket=bucket,
            prefix=prefix,
            region=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            use_blob_urls=True,
        )
        print(f"\n✓ Cloud storage connected successfully!")
        print(f"  Storage ID: {result.get('id', 'N/A')}")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"\nView at: {base_url}/projects/{project_id}/settings/storage")
    except Exception as e:
        print(f"\n✗ Failed to connect cloud storage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect cloud storage to an existing Label Studio project."
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="Label Studio project ID",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="",
        help="S3 bucket name (defaults to S3_BUCKET env var)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 prefix/folder path (optional)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="",
        help="AWS region (defaults to AWS_REGION env var or ap-south-1)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="s3",
        choices=["s3", "gcs", "azure"],
        help="Storage type (default: s3)",
    )
    args = parser.parse_args()
    
    asyncio.run(connect_storage(
        project_id=args.project_id,
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        storage_type=args.type,
    ))


if __name__ == "__main__":
    main()

