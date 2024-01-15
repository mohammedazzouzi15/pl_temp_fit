#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.random as nr
import pytensor
import scipy.linalg
import scipy.special

from pytensor import tensor as pt

import pymc as pm

from pymc.blocking import RaveledVars,DictToArrayBijection
from pymc.pytensorf import (
    CallableTensor,
    compile_pymc,
    floatX,
    join_nonshared_inputs,
    replace_rng_nodes,
)
from pymc.step_methods.arraystep import (

    StatsType,
    metrop_select,
)

__all__ = [
    "Metropolis",
    "DEMetropolis",
    "DEMetropolisZ",
    "BinaryMetropolis",
    "BinaryGibbsMetropolis",
    "CategoricalGibbsMetropolis",
    "NormalProposal",
    "CauchyProposal",
    "LaplaceProposal",
    "PoissonProposal",
    "MultivariateNormalProposal",
]

from pymc.util import get_value_vars_from_user_vars

# Available proposal distributions for Metropolis


class Proposal:
    def __init__(self, s):
        self.s = s




class NormalProposal(Proposal):
    def __call__(self, rng: Optional[np.random.Generator] = None):
        return (rng or nr).normal(scale=self.s)





class UniformProposal(Proposal):
    def __call__(self, rng: Optional[np.random.Generator] = None):
        return (rng or nr).uniform(low=-self.s, high=self.s, size=len(self.s))





class CauchyProposal(Proposal):
    def __call__(self, rng: Optional[np.random.Generator] = None):
        return (rng or nr).standard_cauchy(size=np.size(self.s)) * self.s





class LaplaceProposal(Proposal):
    def __call__(self, rng: Optional[np.random.Generator] = None):
        size = np.size(self.s)
        r = rng or nr
        return (r.standard_exponential(size=size) - r.standard_exponential(size=size)) * self.s





class PoissonProposal(Proposal):
    def __call__(self, rng: Optional[np.random.Generator] = None):
        return (rng or nr).poisson(lam=self.s, size=np.size(self.s)) - self.s





class MultivariateNormalProposal(Proposal):


    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = scipy.linalg.cholesky(s, lower=True)


    def __call__(self, num_draws=None, rng: Optional[np.random.Generator] = None):
        rng_ = rng or nr
        if num_draws is not None:
            b = rng_.normal(size=(self.n, num_draws))
            return np.dot(self.chol, b).T
        else:
            b = rng_.normal(size=self.n)
            return np.dot(self.chol, b)


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """
    tuned_scale =  scale * np.where(
        acc_rate < 0.001,
        # reduce by 90 percent
        0.1,
        np.where(
            acc_rate < 0.05,
            # reduce by 50 percent
            0.5,
            np.where(
                acc_rate < 0.2,
                # reduce by ten percent
                0.9,
                np.where(
                    acc_rate > 0.95,
                    # increase by factor of ten
                    10.0,
                    np.where(
                        acc_rate > 0.75,
                        # increase by double
                        2.0,
                        np.where(
                            acc_rate > 0.5,
                            # increase by ten percent
                            1.1,
                            # Do not change
                            1.0,
                        ),
                    ),
                ),
            ),
        ),
    )
    print("Acceptance rate:", acc_rate, "-> Tuned scale:", tuned_scale)
    return tuned_scale


class Metropolis(pm.Metropolis):

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, StatsType]:
        point_map_info = q0.point_map_info
        q0d = q0.data

        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(self.scaling, self.accepted_sum / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted_sum[:] = 0

        delta = self.proposal_dist() * self.scaling

        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype("int64")
                q0d = q0d.astype("int64")
                q = (q0d + delta).astype("int64")
            else:
                delta[self.discrete] = np.round(delta[self.discrete], 0)
                q = q0d + delta
        else:
            q = floatX(q0d + delta)

        if self.elemwise_update:
            q0d = q0d.copy()
            q_temp = q0d.copy()
            # Shuffle order of updates (probably we don't need to do this in every step)
            np.random.shuffle(self.enum_dims)
            for i in self.enum_dims:
                q_temp[i] = q[i]
                accept_rate_i = self.delta_logp(q_temp, q0d)
                q_temp_, accepted_i = metrop_select(accept_rate_i, q_temp, q0d)
                q_temp[i] = q0d[i] = q_temp_[i]
                self.accept_rate_iter[i] = accept_rate_i
                self.accepted_iter[i] = accepted_i
                self.accepted_sum[i] += accepted_i
            q = q_temp
        else:
            accept_rate = self.delta_logp(q, q0d)
            q, accepted = metrop_select(accept_rate, q, q0d)
            self.accept_rate_iter = accept_rate
            self.accepted_iter = accepted
            self.accepted_sum += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": np.mean(self.scaling),
            "accept": np.mean(np.exp(self.accept_rate_iter)),
            "accepted": np.mean(self.accepted_iter),
        }

        return RaveledVars(q, point_map_info), [stats]

class DEMetropolis(pm.DEMetropolis):

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, StatsType]:
        point_map_info = q0.point_map_info
        q0d = q0.data

        if not self.steps_until_tune and self.tune:
            if self.tune == "scaling":
                self.scaling = tune(self.scaling, self.accepted / float(self.tune_interval))
            elif self.tune == "lambda":
                self.lamb = tune(self.lamb, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        epsilon = self.proposal_dist() * self.scaling

        # differential evolution proposal
        # select two other chains
        ir1, ir2 = np.random.choice(self.other_chains, 2, replace=False)
        r1 = DictToArrayBijection.map(self.population[ir1])
        r2 = DictToArrayBijection.map(self.population[ir2])
        # propose a jump
        q = floatX(q0d + epsilon + self.lamb * (r1.data - r2.data))

        accept = self.delta_logp(q, q0d)
        q_new, accepted = metrop_select(accept, q, q0d)
        self.accepted += accepted

        self.steps_until_tune -= 1

        stats = {
            "tune": self.tune,
            "scaling": np.mean(self.scaling),
            "lambda": self.lamb,
            "accept": np.exp(accept),
            "accepted": accepted,
        }

        return RaveledVars(q_new, point_map_info), [stats]


def sample_except(limit, excluded):
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate


def delta_logp(
    point: Dict[str, np.ndarray],
    logp: pt.TensorVariable,
    vars: List[pt.TensorVariable],
    shared: Dict[pt.TensorVariable, pt.sharedvar.TensorSharedVariable],
) -> pytensor.compile.Function:
    [logp0], inarray0 = join_nonshared_inputs(
        point=point, outputs=[logp], inputs=vars, shared_inputs=shared
    )

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = CallableTensor(logp0)(inarray1)
    # Replace any potential duplicated RNG nodes
    (logp1,) = replace_rng_nodes((logp1,))

    f = compile_pymc([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f