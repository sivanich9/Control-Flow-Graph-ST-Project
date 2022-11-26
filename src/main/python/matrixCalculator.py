import numpy as np

from matrix_ops import ScalarOperations, VectorOperations


def getFirstTierChoice():
    while True:
        try:
            print("""
            Enter the first tier operation you want made:
            1. Scalar Operations 
            2. Vector Operations
            3. Singular Matrix Operations
            4. Stream Operations
            5. Exit
            """)
            firstChoice = int(input("Enter the first tier choice you want: ")) 
        except ValueError:
            print("Please enter a valid option.")
        if firstChoice >= 1 and firstChoice <= 5:
            return firstChoice
        else:
            print("Please enter a valid option.")


def getScalarOperation():
    while True:
        try:
            print("""
            Enter the scalar operation you want to perform:
            1. Scalar addition
            2. Scalar subtraction
            3. Scalar multiplication
            4. Scalar division
            5. Scalar remainder
            """)
            choice = int(input("Enter the scalar operation you want: "))
        except ValueError:
            print("Please enter a valid option")
        if choice >= 1 and choice <= 5:
            return choice
        else:
            print("Please enter a valid option.")



def getMatrixInput():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))         
    matrix = []
    for i in range(1, rows + 1):
        arr = []
        for j in range(1, cols + 1):
            arr.append(int(input("A{}{}: ".format(i, j))))
        matrix.append(arr)
    return rows, cols, matrix 


def getScalarInput():
    scalar = int(input("Enter the scalar you want operated: "))
    return scalar



def calculator(firstChoice, secondChoice, firstInput, secondInput):
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


        


# if __name__ == "__main__":
#     print("-------------- MATRIX CALCULATOR v1.0 Dev --------------")
#     firstChoice = getFirstTierChoice()
#     if firstChoice == 1:
#         secondChoice = getScalarOperation()
#         r,c,m = getMatrixInput()
#         s = getScalarInput()
#         result = calculator(firstChoice, secondChoice, m, s)
#         print(result)