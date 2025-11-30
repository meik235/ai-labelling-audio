#!/usr/bin/env python3
"""Export Label Studio annotations + deliverables for an existing project."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.pipeline import PipelineService  # noqa: E402


async def export_project(project_id: int) -> None:
    service = PipelineService.from_settings()
    result = await service.run_export_only(project_id=project_id)

    print("Export complete")
    print(f"- Project ID: {result['project_id']}")
    print(f"- Exported tasks: {result['export'].get('task_count')}")
    if result.get("deliverables"):
        print(f"- Final CSV: {result['deliverables'].get('final_csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export finalized annotations for an existing LS project."
    )
    parser.add_argument(
        "--project-id",
        type=int,
        required=True,
        help="Label Studio project ID to export",
    )
    args = parser.parse_args()
    asyncio.run(export_project(args.project_id))


if __name__ == "__main__":
    main()

