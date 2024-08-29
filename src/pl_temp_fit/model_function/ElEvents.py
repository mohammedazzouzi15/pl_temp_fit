"""EL generation function for the model."""

import numpy as np


def el_gen(EX, CT, D):
    """Calculate the EL generation of excitons.

    Here if the CT is present we consider that the generation goes through the CT.
    the value 1e25 is used to represent a certain amoung of generation. it should not affect the results as it stands

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
    if CT.off == 1:
        EX.Gen = 1e25
        CT.Gen = "state is off"
    else:
        EX.Gen = 0
        CT.Gen = 1e25 
    return EX, CT
