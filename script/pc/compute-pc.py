import numpy as np                          # type: ignore
import sys
import os
import logging
from netCDF4 import Dataset                 # type: ignore
from scipy.ndimage import maximum_filter    # type: ignore

# Local module
sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics                              # type: ignore

# --- Configuration ---
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_PATH = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Global/pc/"
ALLOWED_MONTHS = {'06', '07', '08', '09'}
MIN_YEAR = 2004
MAX_YEAR = 2019

logging.basicConfig(level=logging.INFO)


def prepare_core(file):
    """
    Prepares core data from a NetCDF file by applying binary thresholding
    and a 5x5 maximum filter to expand active pixels.

    Args:
        file (str): Path to the NetCDF file containing core data.

    Returns:
        np.ndarray: Binary array of processed core data after filtering.

    Raises:
        FileNotFoundError, OSError, RuntimeError, IndexError
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"The file '{file}' does not exist.")

    try:
        with Dataset(file, "r") as data:
            cores = data.variables["cores"][0, :, :]

            if cores.ndim != 2:
                raise RuntimeError("Input data must be a 2D array.")

            cores = np.nan_to_num(cores, nan=0.0)
            binary = (cores > 0).astype(np.uint8)

            # Apply a 5x5 neighbourhood max filter
            filtered = maximum_filter(binary, size=(5, 5))

            return filtered

    except (FileNotFoundError, OSError, RuntimeError, IndexError) as e:
        raise e


def compute_pc(dataset):
    """
    Computes the probabilistic core (PC) from a list of pre-filtered file paths.

    Args:
        dataset (list): List of file paths.

    Returns:
        np.ndarray: 2D array of probabilistic core.

    Raises:
        ValueError: If no valid files were processed.
    """
    sum_cores = None
    valid_files = 0

    for file in dataset:
        try:
            cores = prepare_core(file)
            if sum_cores is None:
                sum_cores = np.zeros_like(cores, dtype=np.float32)

            sum_cores += cores
            valid_files += 1

        except (FileNotFoundError, OSError, RuntimeError, IndexError) as e:
            logging.warning(f"Skipping {file} due to error: {e}")
            continue

    if valid_files == 0:
        raise ValueError("No valid files were processed from the dataset.")

    return sum_cores / valid_files


def search(hour, minute, data_path, allowed_months, min_year, max_year):
    """
    Searches for files matching the given time, filtered by month and year.

    Args:
        hour (str): Hour as two-digit string.
        minute (str): Minute as two-digit string.
        data_path (str): Path to search.
        allowed_months (set): Allowed months as {'06', '07', ...}
        min_year (int): Minimum allowed year.
        max_year (int): Maximum allowed year.

    Returns:
        list: List of file paths.
    """
    if not (isinstance(hour, str) and hour.isdigit() and len(hour) == 2):
        raise ValueError(f"Invalid hour: {hour}")
    if not (isinstance(minute, str) and minute.isdigit() and len(minute) == 2):
        raise ValueError(f"Invalid minute: {minute}")

    all_files = snflics.all_files_in(data_path)
    matching_files = []

    for file in all_files:
        try:
            file_time = snflics.get_time(file)
            year = int(file_time["year"])
            if (file_time["minute"] == minute and
                file_time["hour"] == hour and
                file_time["month"] in allowed_months and
                min_year <= year <= max_year):
                matching_files.append(file)

        except Exception as e:
            logging.warning(f"Error parsing time from {file}: {e}")
            continue

    return matching_files


if __name__ == "__main__":
    hour = sys.argv[1]
    minutes = ["00", "15", "30", "45"]

    for minute in minutes:
        dataset = search(hour, minute, data_path=DATA_PATH,
                         allowed_months=ALLOWED_MONTHS,
                         min_year=MIN_YEAR, max_year=MAX_YEAR)

        pc = compute_pc(dataset)
        output_filename = f"pc-global-{hour}-{minute}.npy"
        np.save(os.path.join(OUTPUT_PATH, output_filename), pc)
        logging.info(f"Saved PC to {output_filename}")
