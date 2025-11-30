"""QC sheet update functionality."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class QCSheetUpdater:
    """Update QC tracking sheet with annotation status."""

    def __init__(self, sheet_path: str | Path) -> None:
        """
        Initialize QC sheet updater.

        Args:
            sheet_path: Path to CSV QC tracking sheet
        """
        self.sheet_path = Path(sheet_path)

    def update_task_status(
        self,
        task_id: str,
        status: str,
        transcript_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        qc_notes: Optional[str] = None,
    ) -> None:
        """
        Update task status in QC sheet.

        Args:
            task_id: Task identifier
            status: Status (e.g., "pending", "in_progress", "completed", "approved")
            transcript_path: Path to transcript file
            audio_path: Path to audio file
            qc_notes: QC reviewer notes
        """
        # Read existing data
        rows = []
        headers = []
        task_found = False

        if self.sheet_path.exists():
            with open(self.sheet_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

        # Ensure required columns exist
        required_columns = ["task_id", "status", "transcript_path", "audio_path", "qc_notes"]
        for col in required_columns:
            if col not in headers:
                headers.append(col)

        # Update or add row
        for row in rows:
            if row.get("task_id") == task_id:
                row["status"] = status
                if transcript_path:
                    row["transcript_path"] = transcript_path
                if audio_path:
                    row["audio_path"] = audio_path
                if qc_notes:
                    row["qc_notes"] = qc_notes
                task_found = True
                break

        if not task_found:
            # Add new row
            new_row = {
                "task_id": task_id,
                "status": status,
                "transcript_path": transcript_path or "",
                "audio_path": audio_path or "",
                "qc_notes": qc_notes or "",
            }
            # Fill missing columns
            for col in headers:
                if col not in new_row:
                    new_row[col] = ""
            rows.append(new_row)

        # Write back
        with open(self.sheet_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status from QC sheet.

        Args:
            task_id: Task identifier

        Returns:
            Task data dict or None if not found
        """
        if not self.sheet_path.exists():
            return None

        with open(self.sheet_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("task_id") == task_id:
                    return row

        return None

