import os
from typing import Dict, List, Optional
from astrodash.config.settings import get_settings
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import LineListNotFoundException, ElementNotFoundException

logger = get_logger(__name__)

class LineListService:
    """
    Service for handling element/ion line list operations.
    Reads and parses the sneLineList.txt file.
    """
    def __init__(self, line_list_path: Optional[str] = None):
        if line_list_path is None:
            settings = get_settings()
            line_list_path = settings.line_list_path
        self.line_list_path = line_list_path
        self._cache: Optional[Dict[str, List[float]]] = None
        logger.info(f"LineListService initialized with file: {self.line_list_path}")

    def load_line_list(self) -> Dict[str, List[float]]:
        if self._cache is not None:
            return self._cache
        if not os.path.exists(self.line_list_path):
            logger.error(f"Line list file not found: {self.line_list_path}")
            raise LineListNotFoundException(self.line_list_path)
        line_dict = {}
        with open(self.line_list_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    logger.warning(f"Skipping invalid line {line_num}: {line}")
                    continue
                key, values = line.split(':', 1)
                key = key.strip()
                wavelength_str = values.replace(',', ' ').replace(';', ' ')
                wavelengths = []
                for w_str in wavelength_str.split():
                    try:
                        wavelengths.append(float(w_str))
                    except ValueError:
                        logger.warning(f"Invalid wavelength '{w_str}' in line {line_num}")
                if wavelengths:
                    line_dict[key] = wavelengths
        self._cache = line_dict
        logger.info(f"Loaded line list with {len(line_dict)} elements/ions")
        return line_dict

    def get_line_list(self) -> Dict[str, List[float]]:
        return self.load_line_list()

    def get_available_elements(self) -> List[str]:
        return list(self.get_line_list().keys())

    def get_element_wavelengths(self, element: str) -> List[float]:
        line_list = self.get_line_list()
        if element not in line_list:
            raise ElementNotFoundException(element)
        return line_list[element]

    def filter_wavelengths_by_range(self, min_wavelength: float, max_wavelength: float) -> Dict[str, List[float]]:
        line_list = self.get_line_list()
        filtered = {}
        for element, wavelengths in line_list.items():
            filtered_wavelengths = [w for w in wavelengths if min_wavelength <= w <= max_wavelength]
            if filtered_wavelengths:
                filtered[element] = filtered_wavelengths
        return filtered
