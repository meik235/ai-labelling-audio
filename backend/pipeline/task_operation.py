"""Task operation types for CSV-based task management."""

from enum import Enum


class TaskOperation(str, Enum):
    """Enum for task operation types from CSV/DynamoDB.
    
    Used to control whether a task should be created, updated, or skipped.
    """
    CREATE = "CREATE"  # Create a new task
    UPDATE = "UPDATE"  # Update an existing task (requires task_id)
    SKIP = "SKIP"      # Skip this row (don't create or update)
    
    @classmethod
    def from_string(cls, value: str) -> "TaskOperation":
        """Parse operation string to enum, case-insensitive.
        
        Args:
            value: String value from CSV/DynamoDB
            
        Returns:
            TaskOperation enum value, defaults to CREATE if invalid
        """
        if not value:
            return cls.CREATE
        
        value_upper = value.strip().upper()
        try:
            return cls(value_upper)
        except ValueError:
            # Default to CREATE for unknown values
            return cls.CREATE

