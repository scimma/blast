import os
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from astrodash.infrastructure.ml.classifiers.architectures import AstroDashPyTorchNet
from astrodash.config.logging import get_logger
from astrodash.config.settings import get_settings


# Existing utility

def get_training_parameters(models_dir: str = None) -> Dict[str, Any]:
    """
    Load training parameters for the Dash model from a pickle file.
    This function matches the backend's interface for easy migration.

    Args:
        models_dir: Optional path to models directory. If None, will try to find it automatically.

    Returns:
        Dictionary containing training parameters (w0, w1, nw, nTypes, etc.)
    """
    if models_dir is None:
        settings = get_settings()
        models_dir = os.path.dirname(settings.dash_training_params_path)

    return load_training_parameters(models_dir)

def load_training_parameters(models_dir: str) -> Dict[str, Any]:
    """
    Load training parameters for the Dash model from a pickle file.
    """
    # If models_dir is a directory, construct the path; otherwise use it as the full path
    if os.path.isdir(models_dir):
        params_path = os.path.join(models_dir, "zeroZ/training_params.pickle")
    else:
        params_path = models_dir

    logger = get_logger(__name__)
    logger.info(f"Loading training parameters from: {params_path}")
    with open(params_path, "rb") as f:
        pars = pickle.load(f, encoding="latin1")
    logger.info("Training parameters loaded.")
    return pars

def classification_split(classification_string: str) -> tuple:
    """
    Split a Dash classification label string (e.g., 'Type: AgeBin') into ('', Type, AgeBin).
    Returns a tuple: (placeholder, type, age_bin)
    """
    parts = classification_string.split(': ')
    return "", parts[0], parts[1] if len(parts) > 1 else None

"""
Deprecated (unused in prod_backend as of 2025-08-10):
The following classes are retained for reference but commented out to avoid accidental use
and unnecessary heavy dependencies. Do not delete without review.

class AgeBinning:
    '''Handle age binning for supernova classification'''
    def __init__(self, min_age: float, max_age: float, age_bin_size: float):
        self.min_age = min_age
        self.max_age = max_age
        self.age_bin_size = age_bin_size
        get_logger(__name__).debug(f"Initialized AgeBinning: min_age={min_age}, max_age={max_age}, bin_size={age_bin_size}")

    def age_bin(self, age: float) -> int:
        return int(round(age / self.age_bin_size)) - int(round(self.min_age / self.age_bin_size))

    def age_labels(self) -> List[str]:
        age_labels = []
        age_bin_prev = 0
        age_label_min = self.min_age
        for age in np.arange(self.min_age, self.max_age, 0.5):
            age_bin = self.age_bin(age)
            if age_bin != age_bin_prev:
                age_label_max = int(round(age))
                age_labels.append(f"{int(age_label_min)} to {age_label_max}")
                age_label_min = age_label_max
            age_bin_prev = age_bin
        age_labels.append(f"{int(age_label_min)} to {int(self.max_age)}")
        return age_labels

class CreateLabels:
    '''Create classification labels for Dash model'''
    def __init__(self, n_types: int, min_age: float, max_age: float, age_bin_size: float, type_list: List[str]):
        self.n_types = n_types
        self.age_binning = AgeBinning(min_age, max_age, age_bin_size)
        self.type_list = type_list
        get_logger(__name__).debug(f"CreateLabels initialized with {n_types} types.")

    def type_names_list(self) -> np.ndarray:
        type_names_list = []
        for t_type in self.type_list:
            for age_label in self.age_binning.age_labels():
                type_names_list.append(f"{t_type}: {age_label}")
        get_logger(__name__).debug(f"Generated {len(type_names_list)} type names.")
        return np.array(type_names_list)

class LoadInputSpectra:
    '''Load and process input spectra for Dash classification'''
    def __init__(
        self,
        file_path_or_data: Union[str, Any],
        z: float,
        smooth: int,
        pars: Dict[str, Any],
        min_wave: float = None,
        max_wave: float = None
    ):
        self.nw = pars['nw']
        n_types, w0, w1, min_age, max_age, age_bin_size, type_list = (
            pars['nTypes'], pars['w0'], pars['w1'], pars['minAge'],
            pars['maxAge'], pars['ageBinSize'], pars['typeList']
        )
        self.type_names_list = CreateLabels(n_types, min_age, max_age, age_bin_size, type_list).type_names_list()
        self.n_bins = len(self.type_names_list)
        # Use DashSpectrumProcessor for preprocessing
        from astrodash.infrastructure.ml.data_processor import DashSpectrumProcessor
        processor = DashSpectrumProcessor(w0, w1, self.nw)
        logger = get_logger(__name__)
        logger.info(f"Loading and processing input spectra. z={z}, smooth={smooth}, min_wave={min_wave}, max_wave={max_wave}")
        if isinstance(file_path_or_data, str):
            data = np.loadtxt(file_path_or_data)
            wave, flux = data[:, 0], data[:, 1]
        else:
            wave, flux = file_path_or_data.x, file_path_or_data.y
        self.flux, self.min_index, self.max_index, self.z = processor.process(
            np.array(wave), np.array(flux), z, smooth, min_wave, max_wave
        )

    def input_spectra(self):
        import torch
        get_logger(__name__).debug("Converting processed flux to torch tensor for model input.")
        input_images = torch.from_numpy(self.flux).float().reshape(1, -1)
        return input_images, [self.z], self.type_names_list, int(self.nw), self.n_bins, [(self.min_index, self.max_index)]

class BestTypesListSingleRedshift:
    '''Get best classification types using the Dash PyTorch model'''
    def __init__(
        self,
        model_path: str,
        input_images: torch.Tensor,
        type_names_list: list,
        n_bins: int
    ):
        self.type_names_list = np.array(type_names_list)
        logger = get_logger(__name__)
        logger.info(f"Loading PyTorch model from {model_path}")

        # Load the saved model state to determine the actual number of output classes
        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        actual_n_types = saved_state_dict['classifier.3.weight'].shape[0]
        logger.info(f"Saved model has {actual_n_types} output classes, but we need {n_bins} classes")

        # Create model with the actual number of output classes from the saved model
        model = AstroDashPyTorchNet(actual_n_types)
        model.load_state_dict(saved_state_dict)
        model.eval()
        logger.info("Model loaded and set to eval mode.")

        with torch.no_grad():
            outputs = model(input_images)

        self.best_types = []
        self.softmax_ordered = []
        for i in range(outputs.shape[0]):
            softmax = outputs[i].numpy()[:n_bins]
            best_types, _, softmax_ordered = self.create_list(softmax)
            self.best_types.append(best_types)
            self.softmax_ordered.append(softmax_ordered)
        logger.info("Classification inference complete.")

    def create_list(self, softmax: np.ndarray):
        idx = np.argsort(softmax)[::-1]
        best_types = self.type_names_list[idx]
        return best_types, idx, softmax[idx]
"""

def combined_prob(best_match_list: list) -> tuple:
    """
    Combine probabilities for best-matching Dash types/ages.
    Returns (best_name, best_age, prob_total, reliable_flag).
    """
    logger = get_logger(__name__)
    prev_name, age, _ = best_match_list[0]
    prob_initial = float(best_match_list[0][2])
    best_name, prob_total = prev_name, 0.0
    prev_broad_type = prev_name[:2]
    ages_list = [int(v) for v in age.split(' to ')]
    prob_possible, ages_list_possible = 0., []
    prob_possible2, ages_list_possible2 = 0., []
    for i, (name, age, prob) in enumerate(best_match_list[:10]):
        min_age, max_age = map(int, age.split(' to '))
        broad_type = "Ib" if "IIb" in name else name[:2]
        if name == prev_name:
            if prob_possible == 0:
                if min_age in ages_list or max_age in ages_list:
                    prob_total += float(prob)
                    ages_list.extend([min_age, max_age])
                else:
                    prob_possible = float(prob)
                    ages_list_possible = [min_age, max_age]
        elif broad_type == prev_broad_type:
            if prob_possible == 0:
                if i <= 1: best_name = broad_type
                prob_total += float(prob)
                ages_list.extend([min_age, max_age])
    if prob_total < prob_initial: prob_total = prob_initial
    best_age = f'{min(ages_list)} to {max(ages_list)}'
    reliable_flag = prob_total > prob_initial
    logger.debug(f"Combined probability: best_type={best_name}, best_age={best_age}, prob_total={prob_total}, reliable={reliable_flag}")
    return best_name, best_age, round(prob_total, 4), reliable_flag
