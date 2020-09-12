import unittest

from examples.covid.covid import launch_covid_experiment


class TestCovid(unittest.TestCase):

    def test_covid_experiment_is_working(self):
        launch_covid_experiment()
        self.assertTrue(True)
