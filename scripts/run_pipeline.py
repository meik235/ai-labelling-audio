#!/usr/bin/env python3
"""Run the end-to-end pipeline from a YAML config."""

from __future__ import annotations

import argparse
import asyncio
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress pyannote.audio and dependency warnings early
# These are library deprecation warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
        print("Pipeline running...")
except ImportError:
    pass

from backend.pipeline import PipelineService
from backend.pipeline import PipelineConfig


async def run_pipeline(config_path: Path) -> None:
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print(f"Copy config/modules.example.yaml to {config_path} and customize it")
        sys.exit(1)

    service = None
    try:
        cfg = PipelineConfig.load(config_path)
        service = PipelineService.from_settings(config_path)
        result = await service.run_with_config(cfg)

        print("\nPipeline completed")
        print(f"- Preprocessing tasks processed: {result['preprocessing_tasks']}")
        if "annotation_tasks" in result:
            print(f"- Annotation tasks prepared: {result['annotation_tasks']}")
        print(f"- Label Studio project ID: {result['project']['id']}")
        
        # Show task push counts if available
        project_info = result.get('project_info', {})
        if project_info:
            tasks_created = project_info.get('tasks_created', 0)
            tasks_updated = project_info.get('tasks_updated', 0)
            tasks_skipped = project_info.get('tasks_skipped', 0)
            print(f"- Tasks created in Label Studio: {tasks_created}")
            print(f"- Tasks updated in Label Studio: {tasks_updated}")
            print(f"- Tasks skipped: {tasks_skipped}")
        
        print(f"- Exported tasks: {result['export']['task_count']}")
        if result.get("deliverables"):
            print(f"- Final CSV: {result['deliverables']['final_csv']}")
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure cleanup happens even on error
        if service:
            service._cleanup()


async def run_export_only(project_id: int, config_path: Optional[Path] = None) -> None:
    """Export annotations from an existing Label Studio project."""
    service = None
    try:
        service = PipelineService.from_settings(config_path)
        result = await service.run_export_only(project_id=project_id)

        print("\nExport completed")
        print(f"- Project ID: {project_id}")
        export_info = result.get("export", {})
        print(f"- Exported tasks: {export_info.get('task_count', 0)}")
        print(f"- Processed tasks: {export_info.get('processed_count', 0)}")
        print(f"- Skipped tasks: {export_info.get('skipped_count', 0)}")
        
        if result.get("deliverables"):
            deliverables = result["deliverables"]
            print(f"- Transcripts: {len(deliverables.get('transcripts', []))}")
            # Annotations are uploaded in workflow, not in deliverables
            # Count from export_info instead (processed annotations)
            annotation_count = export_info.get("processed_count", 0)
            print(f"- Annotations: {annotation_count}")
            if deliverables.get("final_csv"):
                print(f"- Final CSV: {deliverables['final_csv']}")
            elif deliverables.get("note"):
                print(f"- Note: {deliverables['note']}")
        else:
            print("- No deliverables (annotations may not be ready yet)")
            if export_info.get('skipped_count', 0) > 0:
                print(f"  → {export_info.get('skipped_count')} tasks were skipped (likely missing annotation data)")
                print("  → Check Label Studio to ensure annotations have proper 'result' structure")
    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure cleanup happens even on error
        if service:
            service._cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run audio annotation pipeline.")
    parser.add_argument(
        "--config",
        help="Path to modules YAML config (e.g., config/modules.yaml). Required unless using --export-only.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export annotations from existing project (requires --project-id)",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        help="Label Studio project ID (required for --export-only)",
    )
    args = parser.parse_args()

    if args.export_only:
        if not args.project_id:
            parser.error("--project-id is required when using --export-only")
        config_path = Path(args.config) if args.config else None
        if config_path and not config_path.exists():
            print(f"Warning: Config file not found: {config_path}")
            print("Continuing with environment variables only...")
            config_path = None
        asyncio.run(run_export_only(args.project_id, config_path))
    else:
        if not args.config:
            parser.error("--config is required (or use --export-only with --project-id)")
        asyncio.run(run_pipeline(Path(args.config)))


if __name__ == "__main__":
    main()

