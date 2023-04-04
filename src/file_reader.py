from glob import iglob
import numpy as np

def get_samples(sample_number: int, number: int) -> list:
    samples_qtd = 0
    samples = list()
    for path in iglob(f'data/{number}_*.txt'):
        samples_qtd += 1
        if samples_qtd > sample_number:
            break
        with open(path, 'r') as file:
            samples.append(np.array(list(filter(lambda x: x, file.read().split(' ')))))
    return samples