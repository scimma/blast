from typing import List, Optional

class Spectrum:
    """
    Domain model representing a scientific spectrum.
    This model is independent of storage, API, or infrastructure concerns.
    """
    def __init__(
        self,
        x: List[float],
        y: List[float],
        redshift: Optional[float] = None,
        id: Optional[str] = None,
        osc_ref: Optional[str] = None,
        file_name: Optional[str] = None,
        meta: Optional[dict] = None
    ):
        self.id = id
        self.osc_ref = osc_ref
        self.file_name = file_name
        self.x = x  # Wavelengths
        self.y = y  # Flux values
        self.redshift = redshift
        self.meta = meta or {}

    def __repr__(self):
        return (
            f"Spectrum(id={self.id}, osc_ref={self.osc_ref}, file_name={self.file_name}, "
            f"x_len={len(self.x)}, y_len={len(self.y)}, redshift={self.redshift})"
        )
