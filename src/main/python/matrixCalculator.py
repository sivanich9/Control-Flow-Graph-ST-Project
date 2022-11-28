import numpy as np
import math 
import os
import regex
import glob as glob
import shutil
import random



class ScalarOperations:

    """
    Class that consists of operations performed to scalar + matrix
    @method scalar addition: adds a scalar to all entities of a matrix
    @method scalar subtraction: subtracts a scalar from all entities of a matrix
    @method scalar multiplication: multiplies a scalar to all entities of a matrix
    @method scalar division: divides a scalar from all entities of a matrix
    """

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
        result = result / self.scalar
        return result


    

    
class VectorOperations:

    """
    Class that consists of operations performed to matrix + matrix
    @method vector addition: adds a matrix to another matrix
    @method vector subtraction: subtracts a matrix from another matrix
    @method vector multiplication: multiplies a matrix to another matrix
    """

    def __init__(self, matrix1, matrix2):
        self.matrix1 = np.array(matrix1)
        self.matrix2 = np.array(matrix2)

        
    def addition(self):
        if self.matrix1.shape == self.matrix2.shape:
            result = np.add(self.matrix1, self.matrix2)
            return result
        else:
            raise Exception("Dimensions of two matrices are not equal")

    
    def subtraction(self):
        if self.matrix1.shape == self.matrix2.shape:
            result = np.subtract(self.matrix1, self.matrix2)
            return result
        else:
            raise Exception("Dimensions of two matrices are not equal")

    
    def multiplication(self):
        result = np.zeros((len(self.matrix1), len(self.matrix2[0])))
        if len(self.matrix1[0]) != len(self.matrix2):
            raise Exception("Matrix sizes not compatible")
        if self.matrix1 == [[]] or self.matrix2 == [[]]:
            return [[]]
        for i in range(len(self.matrix1)):
            for j in range(len(self.matrix2[0])):
                for k in range(len(self.matrix2)):
                    result[i][j] += self.matrix1[i][k] * self.matrix2[k][j]
        return result

    
    def dotProduct(self):
        result = np.dot(self.matrix1, self.matrix2)
        return result

    
    def outerProduct(self):
        result = np.outer(self.matrix1, self.matrix2)
        return result


    



class SingleMatrixOperation:
    """
    Class to perform operations on a single matrix
    @method transpose -> gives the transpose of the matrix
    @method determinant -> gives the determinant of the matrix
    @method minor -> gives the minor matrix of the matrix
    @method inverse -> gives the inverse matrix of the matrix
    @method cofactor -> gives the cofactors of the givn matrix
    @method adjoint -> gives the adjoint of the given matrix
    @method checkSymmetric -> checks if the matrix is symmetric or not
    """

    def __init__(self, matrix):
        self.matrix = matrix

    
    def transpose(self):
        result = np.zeros((len(self.matrix[0]), len(self.matrix)))
        if self.matrix == [[]]:
            result = [[]]
            return result
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                result[j][i] = self.matrix[i][j]
        return result

    
    def determinant(self):
        if self.matrix == [[]]:
            return 1

        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        n = len(self.matrix)
        mat = self.matrix
        temp = [0] * n
        total = 1
        det = 1

        for i in range(0, n):
            index = i
            while index < n and mat[index][i] == 0:
                index += 1

            if index == n:
                continue

            if index != i:
                for j in range(0, n):
                    mat[index][j], mat[i][j] = mat[i][j], mat[index][j]
                det = det * int(pow(-1, index - i))

            for j in range(0, n):
                temp[j] = mat[i][j]

            for j in range(i + 1, n):
                num1 = temp[i]
                num2 = mat[j][i]

                for k in range(0, n):
                    mat[j][k] = (num1 * mat[j][k]) - (num2 * temp[k])

                total = total * num1
        for i in range(0, n):
            det = det * mat[i][i]

        return int(det / total)

    

    def minor(self, i, j):
        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        if len(self.matrix[0]) == 1 and len(self.matrix) == 1:
            return [[1]]

        return [
            row[:j] + row[j + 1 :] for row in (self.matrix[:i] + self.matrix[i + 1 :])
        ]
    

    def minorMatrix(self):
        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        if len(self.matrix[0]) == 1 and len(self.matrix) == 1:
            return [[1]]
        
        if self.matrix == [[]]:
            raise Exception("Null matrix")
        
        mm = np.zeros(np.array(self.matrix).shape)
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[0])):
                mm[r][c] = round(np.linalg.det(self.minor(r, c)))

        return mm

    

    def inverse(self):
        if self.matrix == [[]]:
            raise Exception("Empty matrix")
        
        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        determinant = self.determinant()
        if len(self.matrix) == 2:
            return [
                [self.matrix[1][1] / determinant, -1 * self.matrix[0][1] / determinant],
                [-1 * self.matrix[1][0] / determinant, self.matrix[0][0] / determinant],
            ]

        cofactors = []
        for r in range(len(self.matrix)):
            cofactorRow = []
            for c in range(len(self.matrix)):
                minor = self.minor(r, c)
                cofactorRow.append(((-1) ** (r + c)) * np.linalg.det(minor))
            cofactors.append(cofactorRow)
        cofactors = np.array(cofactors).T
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c] / determinant
        return cofactors

    

    def cofactor(self):
        if self.matrix == [[]]:
            raise Exception("Empty matrix")

        if len(self.matrix[0]) == 1 and len(self.matrix) == 1:
            return [[1]]

        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        cofactors = np.zeros(np.array(self.matrix).shape)
        minor = self.minorMatrix()
        for r in range(len(minor)):
            for c in range(len(minor[0])):
                cofactors[r][c] = (-1)**(r + c) * minor[r][c]
        return cofactors

    

    def adjoint(self):
        if self.matrix == [[]]:
            raise Exception("Empty matrix")

        if len(self.matrix[0]) == 1 and len(self.matrix) == 1:
            return [[1]]

        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        cofactors = []
        cofactors = np.zeros(np.array(self.matrix).shape)
        minor = self.minorMatrix()
        for r in range(len(minor)):
            for c in range(len(minor[0])):
                cofactors[r][c] = (-1)**(r + c) * minor[r][c]
        return cofactors.T 

    

    def checkSymmetric(self):
        if self.matrix == [[]]:
            return True

        if len(self.matrix[0]) != len(self.matrix):
            raise Exception("Not a square matrix")

        self.matrix = np.array(self.matrix)
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if self.matrix[i][j] != self.matrix.T[i][j]:
                    return False
        return True
    

    def checkOrthogonal(self) :
        if self.matrix == [[]]:
            return False
        a = self.matrix
        m = len(self.matrix)
        n = len(self.matrix[0])
        if (m != n) :
            return False
        
        trans = [[0 for x in range(n)]
                    for y in range(n)]
                    
        for i in range(0, n) :
            for j in range(0, n) :
                trans[i][j] = a[j][i]
                
        prod = [[0 for x in range(n)]
                for y in range(n)]
                    
        for i in range(0, n) :
            for j in range(0, n) :
        
                sum = 0
                for k in range(0, n) :
                    sum = sum + (a[i][k] * a[j][k])
        
                prod[i][j] = sum

        for i in range(0, n) :
            for j in range(0, n) :

                if (i != j and prod[i][j] != 0) :
                    return False
                if (i == j and prod[i][j] != 1) :
                    return False

        return True

    
    def rank(self):
        def swap(Matrix, row1, row2, col):
            for i in range(col):
                temp = Matrix[row1][i]
                Matrix[row1][i] = Matrix[row2][i]
                Matrix[row2][i] = temp
                
    
        rank = len(self.matrix[0])
        for row in range(0, rank, 1):
            if self.matrix[row][row] != 0:
                for col in range(0, len(self.matrix), 1):
                    if col != row:
                        multiplier = (self.matrix[col][row] /
                                    self.matrix[row][row])
                        for i in range(rank):
                            self.matrix[col][i] -= (multiplier * self.matrix[row][i])
                                                
          
            else:
                reduce = True
                for i in range(row + 1, len(self.matrix), 1):
                    if self.matrix[i][row] != 0:
                        self.swap(self.matrix, row, i, rank)
                        reduce = False
                        break
                        
                if reduce:
                    rank -= 1
        
                    for i in range(0, len(self.matrix), 1):
                        self.matrix[i][row] = self.matrix[i][rank]
                        
                row -= 1
        return rank
    
    

    def checkInvertible(self):
        if self.matrix == [[]]:
            raise Exception("Empty matrix")
        if np.linalg.det(self.matrix) != 0:
            return True
        else:
            return False 



# main function that acts as a matrix calculator
def calculator(firstChoice, secondChoice, firstInput, secondInput=None):
    if firstChoice == 1:
        scalaroperations = ScalarOperations(firstInput, secondInput)

        if secondChoice == 1:
            result = scalaroperations.addition()
            return result

        elif secondChoice == 2:
            result = scalaroperations.subtraction()
            return result

        elif secondChoice == 3:
            result = scalaroperations.multiplication()
            return result

        elif secondChoice == 4:
            result = scalaroperations.division()
            return result

    elif firstChoice == 2:
        vectoroperations = VectorOperations(firstInput, secondInput)

        if secondChoice == 1:
            result = vectoroperations.addition()
            return result

        elif secondChoice == 2:
            result = vectoroperations.subtraction()
            return result

        elif secondChoice == 3:
            result = vectoroperations.multiplication()
            return result

        elif secondChoice == 4:
            result = vectoroperations.dotProduct()
            return result

        elif secondChoice == 5:
            result = vectoroperations.outerProduct()
            return result

    elif firstChoice == 3:
        singlematrixops = SingleMatrixOperation(firstInput)

        if secondChoice == 1:
            result = singlematrixops.transpose()
            return result

        elif secondChoice == 2:
            result = singlematrixops.determinant()
            return result

        elif secondChoice == 3:
            result = singlematrixops.minorMatrix()
            return result

        elif secondChoice == 4:
            result = singlematrixops.cofactor()
            return result

        elif secondChoice == 5:
            result = singlematrixops.adjoint()
            return result

        elif secondChoice == 6:
            result = singlematrixops.inverse()
            return result

        elif secondChoice == 7:
            result = singlematrixops.checkSymmetric()
            return result
        
        elif secondChoice == 8:
            result = singlematrixops.checkOrthogonal()
            return result 
        
        elif secondChoice == 9:
            result = singlematrixops.checkInvertible()
            return result 
        

        elif secondChoice == 10:
            result = singlematrixops.rank()
            return result 