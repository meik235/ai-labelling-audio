#!/usr/bin/env python3
"""Label Studio debugging helper."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import get_config, get_value  # noqa: E402
from backend.annotation import LabelStudioClient  # noqa: E402


async def list_ml_backends(client: LabelStudioClient) -> None:
    backends = await client.list_ml_backends()
    if not backends:
        print("No ML backends registered.")
        return

    print("Registered ML backends:")
    for backend in backends:
        print(
            f"- ID={backend.get('id')} title={backend.get('title')} "
            f"url={backend.get('url')} is_interactive={backend.get('is_interactive')}"
        )


async def show_project_info(client: LabelStudioClient, project_id: int, *, raw: bool) -> None:
    project = await client.get_project(project_id)
    if raw:
        print(json.dumps(project, indent=2))
        return

    print(f"Project {project_id}: {project.get('title')}")
    ml_backends = project.get("ml_backends") or []
    if ml_backends:
        print(f"Attached ML backend IDs: {ml_backends}")
    else:
        print("No ML backends attached.")

    print(f"Task count: {project.get('task_number')}")
    owner = project.get("created_by") or {}
    owner_name = owner.get("username") or owner.get("email") or "unknown"
    print(f"Created by: {owner_name}")


async def attach_backend(client: LabelStudioClient, project_id: int, backend_id: int) -> None:
    await client.attach_ml_backend_to_project(project_id, backend_id)
    print(f"Attached ML backend {backend_id} to project {project_id}.")

async def async_main(args: argparse.Namespace) -> None:
    config = get_config()
    ls_cfg = get_value("integrations", "labelStudio", config=config)
    if not isinstance(ls_cfg, dict):
        raise RuntimeError("integrations.labelStudio section is missing in modules.yaml.")

    base_url = ls_cfg.get("baseUrl")
    api_key = ls_cfg.get("apiKey")
    if not base_url or not api_key:
        raise RuntimeError("labelStudio.baseUrl and apiKey must be set in modules.yaml.")

    client = LabelStudioClient(str(base_url), str(api_key))

    if args.command == "list-ml":
        await list_ml_backends(client)
    elif args.command == "project":
        await show_project_info(client, args.project_id, raw=args.raw)
    elif args.command == "attach-ml":
        await attach_backend(client, args.project_id, args.backend_id)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label Studio debugging utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-ml", help="List registered ML backends.")

    project_parser = sub.add_parser("project", help="Show project info and attached ML backends.")
    project_parser.add_argument("--project-id", type=int, required=True)
    project_parser.add_argument("--raw", action="store_true", help="Print raw JSON.")

    attach_parser = sub.add_parser("attach-ml", help="Attach an ML backend to a project.")
    attach_parser.add_argument("--project-id", type=int, required=True)
    attach_parser.add_argument("--backend-id", type=int, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()


