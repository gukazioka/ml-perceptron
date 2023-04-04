from glob import iglob
import numpy as np

INPUT_DIR = 'data/*'

def read_file():
    for path in iglob(INPUT_DIR):
        with open(path, 'r') as file:
            sample = np.array(list(filter(lambda x: x, file.read().split(' '))))
            print(sample)


read_file()
