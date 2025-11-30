"""Optional annotation enrichment helpers (keywords/tagging)."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.pipeline.config import AdditionalAnnotationConfig


class EnrichmentRunner:
    """Adds keyword extraction / tagging metadata when enabled."""

    def run(
        self,
        tasks: List[Dict[str, Any]],
        config: AdditionalAnnotationConfig,
    ) -> List[Dict[str, Any]]:
        if not tasks:
            return tasks

        keywords_enabled = config.keywords.enabled
        tagging_enabled = config.tagging.enabled

        if not keywords_enabled and not tagging_enabled:
            return tasks

        enriched_tasks: List[Dict[str, Any]] = []
        for task in tasks:
            enriched = dict(task)
            enrichment = dict(enriched.get("enrichment", {}))

            transcript_text = enriched.get("transcription", "")
            if keywords_enabled and transcript_text:
                enrichment["keywords"] = self._extract_keywords(
                    transcript_text, config.keywords.model
                )
                enriched["keywords"] = enrichment["keywords"]

            if tagging_enabled and transcript_text:
                enrichment["tagging"] = self._tag_transcription(
                    transcript_text, config.tagging.labels
                )

            if enrichment:
                enriched["enrichment"] = enrichment

            enriched_tasks.append(enriched)

        return enriched_tasks

    def _extract_keywords(self, text: str, model_name: str) -> List[str]:
        """Placeholder keyword extractor (frequency-based)."""
        tokens = [token.strip(".,!?").lower() for token in text.split()]
        unique_tokens = []
        for token in tokens:
            if not token:
                continue
            if token in unique_tokens:
                continue
            unique_tokens.append(token)
        return unique_tokens[:20]

    def _tag_transcription(self, text: str, labels: List[str]) -> Dict[str, float]:
        """Simple heuristic tagging: presence/absence of keywords."""
        text_lower = text.lower()
        tags: Dict[str, float] = {}
        for label in labels:
            tag = label.lower()
            tags[label] = 1.0 if tag in text_lower else 0.0
        return tags

