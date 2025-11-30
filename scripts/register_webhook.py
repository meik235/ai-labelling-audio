#!/usr/bin/env python3
"""Register Label Studio webhook to trigger export on task completion."""

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


async def register_webhook(project_id: int, webhook_url: str) -> None:
    """Register webhook for a Label Studio project."""
    base_url = os.getenv("LABEL_STUDIO_BASE_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    
    if not base_url or not api_key:
        print("Error: LABEL_STUDIO_BASE_URL and LABEL_STUDIO_API_KEY must be set in .env file")
        sys.exit(1)
    
    client = LabelStudioClient(base_url, api_key)
    
    print(f"\nRegistering webhook for project {project_id}")
    print(f"  - URL: {webhook_url}")
    print(f"  - Events: TASK_COMPLETED, ANNOTATION_CREATED, ANNOTATION_UPDATED")
    
    try:
        result = await client.register_webhook(
            project_id=project_id,
            target_url=webhook_url,
            events=["TASK_COMPLETED", "ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
            is_active=True,
        )
        print(f"\n✓ Webhook registered successfully!")
        print(f"  Webhook ID: {result.get('id', 'N/A')}")
        print(f"  Status: {'Active' if result.get('is_active') else 'Inactive'}")
        print(f"\nView at: {base_url}/projects/{project_id}/settings/webhooks")
    except Exception as e:
        print(f"\n✗ Failed to register webhook: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check that Label Studio is running at {base_url}")
        print(f"  2. Verify your API token has admin/project permissions")
        print(f"  3. Try registering manually in UI: {base_url}/projects/{project_id}/settings/webhooks")
        print(f"  4. Check Label Studio logs for detailed error")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register webhook for automatic export on task completion."
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="Label Studio project ID",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default="http://localhost:8000/webhooks/label-studio/events",
        help="Webhook endpoint URL (default: http://localhost:8000/webhooks/label-studio/events)",
    )
    args = parser.parse_args()
    
    asyncio.run(register_webhook(args.project_id, args.webhook_url))


if __name__ == "__main__":
    main()

