from typing import Any

class BaseClassifier:
    """
    Abstract base classifier interface. All classifiers should implement an async classify method.
    """
    def __init__(self, config=None):
        self.config = config

    async def classify(self, spectrum: Any) -> dict:
        raise NotImplementedError("Subclasses must implement classify()")

    def classify_sync(self, spectrum: Any) -> dict:
        """
        Synchronous classification method for CPU-bound work. Subclasses should override.
        The default implementation raises to force subclasses to implement.
        """
        raise NotImplementedError("Subclasses must implement classify_sync()")
