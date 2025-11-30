"""Pre-annotation helpers (segmentation, diarization metadata)."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.pipeline.config import PreAnnotationConfig


class PreAnnotationRunner:
    """Applies configurable pre-annotation metadata to tasks."""

    def run(
        self,
        tasks: List[Dict[str, Any]],
        config: PreAnnotationConfig,
    ) -> List[Dict[str, Any]]:
        if not tasks:
            return tasks

        if not config.enabled:
            return tasks
        
        needs_schema = bool(config.label_schema.fields)
        if not needs_schema:
            return tasks
        
        updated: List[Dict[str, Any]] = []
        for task in tasks:
            enriched = dict(task)
            preann = dict(enriched.get("preannotation", {}))

            if needs_schema:
                preann["label_schema"] = {"fields": config.label_schema.fields}

            enriched["preannotation"] = preann
            updated.append(enriched)

        return updated

