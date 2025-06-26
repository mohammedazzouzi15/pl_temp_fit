"""script to calculate the absorption properties of the system"""

import numpy as np


def calculate_alpha(EX, CT, D):
    """Calculate the absorption of the system.

    Args:
    ----
        EX (State): The exciton state.
        CT (State): The charge transfer state.
        D (Data): The data.

    Returns:
    -------
        EX (State): The exciton state.
        CT (State): The charge transfer state.

    """
    alpha = np.zeros(len(D.T))
    if CT.off == 1:
        alpha = EX.ka_hw * D.nie * D.Excitondesnity
    else:
        alpha = (
            EX.ka_hw * D.nie * D.Excitondesnity
            + CT.ka_hw * D.nie * D.Excitondesnity * D.RCTE
        )
    return alpha
