import unittest
import numpy as np
import pandas as pd
from src.data.prepdata.binning import binning
import pandas.testing as pdt

class TestBinning(unittest.TestCase):

    def setUp(self):
        df1 = pd.DataFrame({
            "numbers": [0.01, 0.0, 0.05, 0.15, 0.51, 0.76, 1.00],
            "numbers2" : [0.02, 0.04, 0.07, 0.98, 0.95, 0.10, 0.15]})
        
        df2 = pd.DataFrame({
            "cat1" : ["a","b","c","a","b","c","a"],
            "cat2" : ["a","b","c","a","b","c","a"]
        })
        self.data1 = {"host1" : df1,
                      "host2" : df2 }

    def test_binning(self):
        datasets = binning(self.data1, 20, ["numbers", "numbers2"])
        sol1 = pd.DataFrame({
            "numbers" : [0.0, 0.0, 0.0, 2.0, 10.0, 15.0, 19.0],
            "numbers2" : [0.0, 0.0, 1.0, 19.0, 18.0, 1.0, 2.0],
        })
        
        sol2 = pd.DataFrame({
            "cat1" : ["a","b","c","a","b","c","a"],
            "cat2" : ["a","b","c","a","b","c","a"]
        })

        pdt.assert_frame_equal(datasets["host1"].astype(float), sol1.astype(float))
        pdt.assert_frame_equal(datasets["host2"], sol2)

if __name__ == "__main__":
    unittest.main()