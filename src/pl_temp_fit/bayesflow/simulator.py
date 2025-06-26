import numpy as np


def simulator(params):
    """Simulate data based on model parameters.

    Args:
    ----
        params (dict): Dictionary of model parameters.

    Returns:
    -------
        np.ndarray: Simulated data.

    """
    # Example: Simulate data based on a Gaussian distribution
    mean = params["mean"]
    std = params["std"]
    data = np.random.normal(mean, std, size=100)
    return data
