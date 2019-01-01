import numpy as np
import threading
import time
from scipy.linalg import blas as FB

arr1 = np.random.randint(10, size=(1000, 750))

arr2 = np.random.randint(10, size=(750, 1250))

start = time.time()
print(np.matmul(arr1, arr2))
end = time.time()
print(end - start)


# Counter that keeps track of the next position to calculate for
class pos_count():
    def __init__(self):
        self.curr_col = 0
        self.curr_row = 0

    def set_size(self, cols, rows):
        self.cols = cols
        self.rows = rows

    # Increment the currently pointed-to position
    def increment(self):
        # Check if at end of column
        if self.curr_row + 1 == self.rows:
            # # If also at end of row
            # if self.curr_col + 1 == self.cols:
            #     return None
            self.curr_col += 1
            self.curr_row = 0
        else:
            self.curr_row += 1

        # Success
        return 1

    # Return the current position in the new matrix to be calculated
    def get_pos(self):
        return self.curr_col, self.curr_row

    # Get pos and increment
    def get_inc(self):
        # Save pos
        temp = self.get_pos()

        self.increment()

        # Return pos
        return temp


def multi_mat_mul(A, B):

    # Convert to float

    # Number of columns in the new array
    new_cols = np.size(A, axis=0)
    # Number of rows in the new array
    new_rows = np.size(B, axis=1)

    count = pos_count()

    # Counter object for distributing positions to calculate
    count.set_size(new_cols, new_rows)

    # Number of threads to create
    num_threads = 16

    # List containing threads
    threads = []

    # Result matrix
    C = np.zeros((new_cols, new_rows))

    for i in range(num_threads):
        new_thread = MyThread("thread-" + str(i), i, count, A, B, C)
        threads.append(new_thread)
        threads[i].start()

    for i in range(num_threads):
        threads[i].join()

    return C


# Thread definition
class MyThread(threading.Thread):
    # Prevent race conditions
    pos_lock = threading.Lock()

    def __init__(self, threadID, i, count, A, B, C):
        super(MyThread, self).__init__()
        self.threadID = threadID
        self.i = i
        self.count = count
        self.A = A
        self.B = B
        self.C = C

    def run(self):

        while True:
            # Get the position of the next element to calculate
            # Lock the shared counter object
            self.pos_lock.acquire()

            # Get pos from counter
            pos = self.count.get_inc()

            # Release counter object
            self.pos_lock.release()

            # Not finished
            # Split position into two variables
            col = pos[0]
            row = pos[1]

            # End of matrix, thread is done
            if col >= np.size(self.C, axis=0):
                break

            # Calculate value for pos in result matrix
            self.C[col, row] = np.dot(self.A[col, :], self.B[:, row])


start = time.time()
print(multi_mat_mul(arr1, arr2))
end = time.time()
print(end - start)
