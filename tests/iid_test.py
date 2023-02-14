import unittest
import numpy as np


class TestIID(unittest.TestCase):
    def test_cov(self, dim=5000, batch=100000):
        samples = np.random.normal(size=(dim, batch))
        cov = np.cov(samples, ddof=0)
        cov -= np.eye(cov.shape[0])
        print(np.max(np.abs(cov.flatten())))
     

if __name__ == "__main__":
    unittest.main()
