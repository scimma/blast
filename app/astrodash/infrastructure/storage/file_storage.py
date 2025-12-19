from astrodash.config.settings import get_settings, Settings
import os
from typing import Optional, List

class FileStorage:
    """
    Generic file storage abstraction for saving, loading, deleting, and listing files in a directory.
    """
    def __init__(self, config: Settings = None):
        self.config = config or get_settings()
        self.storage_dir = self.config.storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, filename: str, data: bytes) -> str:
        """Save bytes to a file. Returns the file path."""
        path = os.path.join(self.storage_dir, filename)
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def load(self, filename: str) -> Optional[bytes]:
        """Load bytes from a file. Returns None if file does not exist."""
        path = os.path.join(self.storage_dir, filename)
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, filename: str) -> bool:
        """Delete a file. Returns True if deleted, False if not found."""
        path = os.path.join(self.storage_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        path = os.path.join(self.storage_dir, filename)
        return os.path.exists(path)

    def list_files(self) -> List[str]:
        """List all files in the storage directory."""
        return [f for f in os.listdir(self.storage_dir) if os.path.isfile(os.path.join(self.storage_dir, f))]
