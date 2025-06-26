"""This module contains utility classes and functions for working with the emcee library."""

import emcee
import numpy as np


class ensemble_sampler(emcee.EnsembleSampler):
    """Workaround the issues with blobs from the sampler."""

    def compute_log_prob(self, coords):
        """Calculate the vector of log-probability for the walkers

        Args:
        ----
            coords: (ndarray[..., ndim]) The position vector in parameter
                space where the probability should be calculated.

        This method returns:

        * log_prob: A vector of log-probabilities with one entry for each
          walker in this sub-ensemble.
        * blob: The list of meta data returned by the ``log_post_fn`` at
          this position or ``None`` if nothing was returned.

        """
        p = coords

        # Check that the parameters are in physical ranges.
        if np.any(np.isinf(p)):
            raise ValueError("At least one parameter value was infinite")
        if np.any(np.isnan(p)):
            raise ValueError("At least one parameter value was NaN")

        # If the parmaeters are named, then switch to dictionaries
        # if self.params_are_named:
        #   p = ndarray_to_list_of_dicts(p, self.parameter_names)

        # Run the log-probability calculations (optionally in parallel).
        if self.vectorize:
            results = self.log_prob_fn(p)
        else:
            # If the `pool` property of the sampler has been set (i.e. we want
            # to use `multiprocessing`), use the `pool`'s map method.
            # Otherwise, just use the built-in `map` function.
            if self.pool is not None:
                map_func = self.pool.map
            else:
                map_func = map
            results = list(map_func(self.log_prob_fn, p))

            log_prob = np.array([float(l) for l in results])

        # Check for log_prob returning NaN.
        if np.any(np.isnan(log_prob)):
            raise ValueError("Probability function returned NaN")

        return log_prob, log_prob


class hDFBackend_2(emcee.backends.HDFBackend):
    """Custom HDFBackend with additional functionalities."""

    def grow(self, ngrow, blobs):
        """Grow the dataset.

        Args:
        ----
            ngrow (int): Number of steps to grow.
            blobs (list): List of blobs.

        """
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            g.attrs["has_blobs"] = False

    def _check(self, state, accepted):
        """Check the state and accepted steps.

        Args:
        ----
            state (State): The state of the sampler.
            accepted (list): List of accepted steps.

        """
        nwalkers, ndim = self.shape
        if state.coords.shape != (nwalkers, ndim):
            raise ValueError(
                f"invalid coordinate dimensions; expected {(nwalkers, ndim)}"
            )
        if state.log_prob.shape != (nwalkers,):
            raise ValueError(
                f"invalid log probability size; expected {nwalkers}"
            )
        if accepted.shape != (nwalkers,):
            raise ValueError(f"invalid acceptance size; expected {nwalkers}")

    def save_step(self, state, accepted):
        """Save a step in the HDF5 file.

        Args:
        ----
            state (State): The state of the sampler.
            accepted (list): List of accepted steps.

        """
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)
        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            g["chain"][iteration, :, :] = state.coords
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None and state.blobs.size > 0:
                g["blobs"][iteration, :] = state.blobs

            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs[f"random_state_{i}"] = v

            g.attrs["iteration"] = iteration + 1

    def has_blobs(self):
        """Check if the backend has blobs.

        Returns
        -------
            bool: True if the backend has blobs, False otherwise.

        """
        return False
