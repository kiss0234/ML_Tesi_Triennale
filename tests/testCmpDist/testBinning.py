import unittest
import numpy as np
import pandas as pd
from src.data.study.comparedist.cmpdist import binning
import pandas.testing as pdt


class TestBinning(unittest.TestCase):

    def setUp(self):
        df1 = pd.DataFrame({
            "numbers": [0.01, 0.0, 0.05, 0.15, 0.51, 0.76, 1.00],
            "numbers2" : [0.02, 0.04, 0.07, 0.98, 0.95, 0.10, 0.15]})
        
        self.data1 = {"host1" : df1,
                      "host2" : df1 }

    def test_binning(self):
        datasets = binning(self.data1, 20)
        sol = pd.DataFrame({
            "numbers" : [0.0, 0.0, 0.0, 2.0, 10.0, 15.0, 19.0],
            "numbers2" : [0.0, 0.0, 1.0, 19.0, 18.0, 1.0, 2.0]
        })

        pdt.assert_frame_equal(datasets["host1"].astype(float), sol.astype(float))
        pdt.assert_frame_equal(datasets["host2"].astype(float), sol.astype(float))

if __name__ == "__main__":
    unittest.main()