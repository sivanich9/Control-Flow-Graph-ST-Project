import numpy as np

class ScalarOperations:
    def __init__(self, matrix, scalar):
        self.matrix = matrix
        self.scalar = scalar 
        self.rows = len(self.matrix)
        self.cols = len(self.matrix[0])

    
    def addition(self):
        result = np.array(self.matrix)
        result[:] += self.scalar
        return result
    

    def subtraction(self):
        result = np.array(self.matrix)
        result[:] -= self.scalar
        return result
    

    def multiplication(self):
        result = np.array(self.matrix)
        result[:] *= self.scalar
        return result 
    

    def division(self):
        result = np.array(self.matrix)
        result[:] /= self.scalar
        return result
    



class VectorOperations:
    def __init__(self, matrix1, matrix2):
        self.matrix1 = matrix1
        self.matrix2 = matrix2


    def addition(self):
        result = np.add(self.matrix1,self.matrix2)
        return result 
    
    def subtraction(self):
        result = np.subtract(self.matrix1, self.matrix2)
        return result
    
    def multiplication(self):
        result = np.matmul(self.matrix1, self.matrix2)
        return result