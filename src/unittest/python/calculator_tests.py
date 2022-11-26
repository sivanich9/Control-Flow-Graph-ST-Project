from mockito import mock, verify

import numpy as np
import unittest


from matrixCalculator import calculator

class MatrixCalculatorTest(unittest.TestCase):
    def test_scalar_addition(self):
        actual_output = calculator(1,1,[[1,2,3],[4,5,6],[7,8,9]],5)
        expected_output = [[6,7,8],[9,10,11],[12,13,14]]
        np.testing.assert_array_equal(actual_output, expected_output)
    
    
    
    def test_vector_addition(self):
        actual_output = calculator(2,1,[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]])
        expected_output = [[2,4,6],[8,10,12],[14,16,18]]
        np.testing.assert_array_equal(actual_output, expected_output)


    def test_vector_multiplication(self):
        actual_output = calculator(2,1,[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]])
        expected_output = [[30,36,42],[66,81,96],[102,126,150]]
        np.testing.assert_array_equal(actual_output, expected_output)


        


if __name__ == "__main__":
    unittest.main()