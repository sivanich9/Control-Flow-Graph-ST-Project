from mockito import mock, verify

import numpy as np
import unittest


from matrixCalculator import calculator

class MatrixCalculatorTest(unittest.TestCase):
   
    #Scalar operations test cases
      
    def test_scalar_addition(self):
        actual_output = calculator(1,1,[[1,2,3],[4,5,6],[7,8,9]],5)
        expected_output = [[6,7,8],[9,10,11],[12,13,14]]
        np.testing.assert_array_equal(actual_output, expected_output)
    
    def test_scalar_subtraction(self):
        actual_output = calculator(1,2,[[5,6,7],[8,9,10],[11,12,13]],3)
        expected_output = [[2,3,4],[5,6,7],[8,9,10]]
        np.testing.assert_array_equal(actual_output, expected_output)
        
    def test_scalar_multiplication(self):
        actual_output = calculator(1,3,[[1,3,5],[7,9,11],[13,15,17]],2)
        expected_output = [[2,6,10],[14,18,22],[26,30,34]]
        np.testing.assert_array_equal(actual_output, expected_output)
        
    def test_scalar_division(self):
        actual_output = calculator(1,4,[[2,4,6],[8,10,12],[14,16,18]],2)
        expected_output = [[1,2,3],[4,5,6],[7,8,9]]
        np.testing.assert_array_equal(actual_output, expected_output)
    
        
    #Vector operation test cases    
   
    def test_vector_addition(self):
        actual_output = calculator(2,1,[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]])
        expected_output = [[2,4,6],[8,10,12],[14,16,18]]
        np.testing.assert_array_equal(actual_output, expected_output)

    def test_vector_subtraction(self):
        actual_output = calculator(2,2,[[10,11,12],[13,14,15],[16,17,18]],[[1,2,3],[4,5,6],[7,8,9]])
        expected_output = [[9,9,9],[9,9,9],[9,9,9]]
        np.testing.assert_array_equal(actual_output, expected_output)
    
    def test_vector_multiplication1(self):
        actual_output = calculator(2,1,[],[])
        expected_output = []
        np.testing.assert_array_equal(actual_output, expected_output)
         
    def test_vector_multiplication2(self):
        actual_output = calculator(2,1,[[2]],[[3]])
        expected_output = [[6]]
        np.testing.assert_array_equal(actual_output, expected_output)    
   
    def test_vector_multiplication3(self):
        actual_output = calculator(2,1,[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]])
        expected_output = [[30,36,42],[66,81,96],[102,126,150]]
        np.testing.assert_array_equal(actual_output, expected_output)
      
    def test_vector_multiplication4(self):
        actual_output = calculator(2,1,[[1,2,3],[4,5,6]],[[7,8],[9,10]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)   
        
    #Singular matrix operation test cases
   
    def test_singular_transpose1(self):
        actual_output = calculator(3,1,[[]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)
      
    def test_singular_transpose2(self):
        actual_output = calculator(3,1,[[5]])
        expected_output = [[5]]
        np.testing.assert_array_equal(actual_output, expected_output)
         
    def test_singular_transpose3(self):
        actual_output = calculator(3,1,[[1,2,3],[4,5,6]])
        expected_output = [[1,4],[2,5],[3,6]]
        np.testing.assert_array_equal(actual_output, expected_output)     
         
      
    def test_singular_determinant1(self):
        actual_output = calculator(3,2,[[]])
        expected_output = 1
        np.testing.assert_array_equal(actual_output, expected_output)
      
    def test_singular_determinant2(self):
        actual_output = calculator(3,2,[[10]])
        expected_output = 10
        np.testing.assert_array_equal(actual_output, expected_output)  
         
    def test_singular_determinant3(self):
        actual_output = calculator(3,2,[[1,0,2,-1],[3,0,0,5],[2,1,4,-3],[1,0,5,0]])
        expected_output = 30
        np.testing.assert_array_equal(actual_output, expected_output)    
         
    def test_singular_determinant4(self):
        actual_output = calculator(3,2,[[1,0,2,-1],[3,0,0,5],[2,1,4,-3]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)   
      
      
    def test_singular_minor1(self):
        actual_output = calculator(3,3,[[]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)
      
    def test_singular_minor2(self):
        actual_output = calculator(3,3,[[5]])
        expected_output = [[1]]
        np.testing.assert_array_equal(actual_output, expected_output)  
         
    def test_singular_minor3(self):
        actual_output = calculator(3,3,[[2,-2,3],[1,4,5],[2,1,-3]])
        expected_output = [[-17,-13,-7],[3,-12,6],[-22,7,10]]
        np.testing.assert_array_equal(actual_output, expected_output)       
     
    def test_singular_minor4(self):
        actual_output = calculator(3,3,[[1,2,3],[4,5,6]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)     
      
      
    def test_singular_cofactor1(self):
        actual_output = calculator(3,4,[[]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)
      
    def test_singular_cofactor2(self):
        actual_output = calculator(3,4,[[5]])
        expected_output = [[1]]
        np.testing.assert_array_equal(actual_output, expected_output)  
         
    def test_singular_cofactor3(self):
        actual_output = calculator(3,4,[[2,-2,3],[1,4,5],[2,1,-3]])
        expected_output = [[-17,13,-7],[-3,-12,-6],[-22,-7,10]]
        np.testing.assert_array_equal(actual_output, expected_output)       
     
    def test_singular_cofactor4(self):
        actual_output = calculator(3,4,[[1,2,3],[4,5,6]])
        expected_output = pass
        np.testing.assert_array_equal(actual_output, expected_output)
      
    #Stream operation test cases
   
   
    #Exit  
   
    
        
        


        


if __name__ == "__main__":
    unittest.main()
