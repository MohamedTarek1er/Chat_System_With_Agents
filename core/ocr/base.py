from abc import ABC, abstractmethod
from pathlib import Path


class BaseOCR(ABC):
    """Common interface for all OCR engines."""

    @abstractmethod
    def extract_text(self, image_source: str | Path | bytes) -> str:
        """Extract plain text from an image file path or raw bytes."""
        raise NotImplementedError
