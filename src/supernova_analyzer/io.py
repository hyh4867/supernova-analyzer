"""Input utilities for supernova light curve data."""

from os import PathLike

import pandas as pd

REQUIRED_COLUMNS = ("time", "magnitude", "magnitude_error")


def load_light_curve(path: str | PathLike[str]) -> pd.DataFrame:
    """Load a supernova light curve CSV file into a DataFrame.

    Args:
        path: Path to a CSV file containing light curve observations.

    Returns:
        A DataFrame containing the light curve data.

    Raises:
        ValueError: If the CSV is empty or does not contain all required columns.
    """
    try:
        light_curve = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("Light curve CSV is empty.") from error

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in light_curve.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Light curve CSV is missing required columns: {missing}.")

    if light_curve.empty:
        raise ValueError("Light curve CSV contains no data rows.")

    return light_curve
