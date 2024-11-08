import unittest
from pathlib import Path

from pl_temp_fit import (
    Exp_data_utils,
    config_utils,
    fit_pl_utils,
)
from pl_temp_fit.data_generators import PLAbsandAlllifetime
import emcee
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)


class TestPLSampler(unittest.TestCase):
    def setUp(self):
        self.model_config_id = "6ecb03ab-0512-4de7-9e42-11440cfd11be"
        self.database_folder = "data_for_test/data_pl"
        self.model_config, self.model_config_save = (
            config_utils.load_model_config(
                self.model_config_id,
                database_folder=self.database_folder,
            )
        )
        self.csv_name_pl = self.model_config_save["csv_name_pl"]
        self.save_folder = self.model_config_save["save_folder"]
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        self.Exp_data_pl, self.temperature_list_pl, self.hws_pl = (
            Exp_data_utils.read_data(self.csv_name_pl)
        )
        self.pl_data_gen = PLAbsandAlllifetime.PLAbsandAlllifetime(
            self.temperature_list_pl, self.hws_pl
        )
        self.pl_data_gen.relative_error_lifetime = self.model_config_save[
            "relative_error_lifetime"
        ]
        self.pl_data_gen.error_in_max_abs_pos = self.model_config_save[
            "error_in_max_abs_pos"
        ]
        self.pl_data_gen.max_abs_pos_exp = self.model_config_save[
            "max_abs_pos_exp"
        ]
        self.pl_data_gen.temperature_lifetimes_exp = self.model_config_save[
            "temperature_lifetimes_exp"
        ]
        self.pl_data_gen.update_with_model_config(self.model_config_save)


    def test_load_model_config(self):
        logging.debug("model_config: ", self.model_config)
        logging.debug("model_config_save: ", self.model_config_save)
        self.assertIsNotNone(self.model_config)
        self.assertIsNotNone(self.model_config_save)

    def test_read_data(self):
        self.assertIsNotNone(self.Exp_data_pl)
        self.assertIsNotNone(self.temperature_list_pl)
        self.assertIsNotNone(self.hws_pl)

    def test_update_with_model_config(self):
        logging.debug(
            "temperature_lifetimes_exp: ",
            self.pl_data_gen.temperature_lifetimes_exp,
        )
        # self.assertEqual(
        #    self.pl_data_gen.temperature_lifetimes_exp.values(),
        #    self.model_config_save["temperature_lifetimes_exp"].values(),
        # )

    def test_plot_limits(self):
        fit_pl_utils.plot_fit_limits(self.model_config, self.model_config_save)

    def test_get_covariance_matrix(self):
        co_var_mat_pl, variance_pl = self.pl_data_gen.get_covariance_matrix()
        self.assertIsNotNone(co_var_mat_pl)
        self.assertIsNotNone(variance_pl)

    def test_get(self):
        co_var_mat_pl, variance_pl = self.pl_data_gen.get_covariance_matrix()
        soln_min = self.pl_data_gen.get_maximum_likelihood_estimate(
            self.Exp_data_pl,
            co_var_mat_pl,
            self.save_folder,
            coeff_spread=0.1,
            num_coords=1,
        )
        assert soln_min is not None
        assert hasattr(soln_min, "x")
        assert hasattr(soln_min, "fun")
        assert isinstance(soln_min.x, np.ndarray)
        assert isinstance(soln_min.fun, float)

    def test_run_sampler_parallel(self):
        co_var_mat_pl, variance_pl = self.pl_data_gen.get_covariance_matrix()
        fit_pl_utils.run_sampler_parallel(
            self.save_folder,
            self.Exp_data_pl,
            co_var_mat_pl,
            self.pl_data_gen,
            nsteps=5,
            coeff_spread=self.model_config_save["coeff_spread"],
            num_coords=self.model_config_save["num_coords"],
            restart_sampling=True,
        )

    def read_sampler_results(self):
        filename = self.model_config_save["save_folder"] + "/sampler.h5"
        reader = emcee.backends.HDFBackend(filename, name="multi_core")
        distribution = reader.get_chain(discard=0)
        self.assertIsNotNone(distribution)
        self.assertEqual(distribution.shape[0], 5)
        self.assertEqual(
            distribution.shape[1], self.model_config_save["num_coords"]
        )
        self.assertEqual(distribution.shape[2], 5)

    def read_blobs(self):
        filename = self.model_config_save["save_folder"] + "/sampler.h5"
        reader = emcee.backends.HDFBackend(filename, name="multi_core")
        blobs = reader.get_blobs(discard=0)
        self.assertIsNotNone(blobs)
        self.assertEqual(blobs.shape[0], 5)
        self.assertEqual(blobs.shape[1], self.model_config_save["num_coords"])
        self.assertEqual(
            blobs.shape[2],
            len(self.pl_data_gen.temperature_lifetimes_exp.keys()) * 2,
        )

        # Add assertions to verify the results of the sampler if possible


if __name__ == "__main__":
    unittest.main()
