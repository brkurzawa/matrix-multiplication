import numpy as np
import threading
import time

arr1 = np.random.randint(10, size=(100, 75))

arr2 = np.random.randint(10, size=(75, 125 ))

start = time.time()
print(np.matmul(arr1, arr2))
end = time.time()
print(end - start)


def multi_mat_mul(A, B):
    # Number of columns in the new array
    new_cols = np.size(A, axis=0)
    # Number of rows in the new array
    new_rows = np.size(B, axis=1)

    # Number of threads to create
    num_threads = new_cols * new_rows

    # List containing threads
    threads = []

    # Result matrix
    C = np.zeros((new_cols, new_rows))

    start = time.time()
    cnt = 0
    for i in range(new_cols):
        for j in range(new_rows):
            new_thread = MyThread("thread-" + str(cnt), cnt, i, j, A[i, :], B[:, j], C)
            threads.append(new_thread)
            cnt += 1
    end = time.time()
    print("Created all threads in " + str(end - start) + " seconds")

    start = time.time()
    # Start threads
    for i in range(cnt):
        threads[i].start()
    end = time.time()
    print("Started all threads in " + str(end - start) + " seconds")

    for i in range(num_threads):
        threads[i].join()

    return C


# Thread definition
class MyThread(threading.Thread):

    def __init__(self, threadID, i, col, row, a, b, C):
        super(MyThread, self).__init__()
        self.threadID = threadID
        self.i = i
        self.col = col
        self.row = row
        self.a = a
        self.b = b
        self.C = C

    def run(self):

        # Calculate value for pos in result matrix
        self.C[self.col, self.row] = np.dot(self.a, self.b)


start = time.time()
print(multi_mat_mul(arr1, arr2))
end = time.time()
print(end - start)
