import numpy as np
import time


class InputError(Exception):
    """Exception raised by input error

    Attributes:
        size1 -- Number of columns in the first matrix
        size2 -- Number of rows in the second matrix
        msg -- Explanation of why the input raised an error
    """

    def __init__(self, size1, size2, msg):
        self.size1 = size1
        self.size2 = size2
        self.msg = msg


# Takes two numpy matrices as input and returns the dot product
def matmult(M1, M2):

    # Enforce matrix multiplication size requirements
    if np.size(M1, axis=1) != np.size(M2, axis=0):
        raise InputError(np.size(M1, axis=0), np.size(M2, axis=1),
        "Number of columns in first matrix must be equal to number "
        "of rows in second matrix")

    # Number of columns in the new array
    new_cols = np.size(M1, axis=0)
    # Number of rows in the new array
    new_rows = np.size(M2, axis=1)
    # Size shared by columns in M1 and rows in M2
    common_size = np.size(M1, axis=1)

    # Initialize new array with zeros in each position
    M3 = np.zeros((new_cols, new_rows))

    # Columns
    for i in range(new_cols):
        # Rows
        for j in range(new_rows):
            for k in range(common_size):
                M3[i, j] += np.multiply(M1[i, k], M2[k, j])

    return M3


arr1 = np.random.randint(10, size=(100, 75))

arr2 = np.random.randint(10, size=(75, 125))

start = time.time()
print(np.matmul(arr1, arr2))
end = time.time()
print(end - start)

start = time.time()
print(matmult(arr1, arr2))
end = time.time()
print(end - start)