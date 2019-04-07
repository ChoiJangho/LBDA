import numpy as np
from numpy import genfromtxt

def import_breast_cancer_wisconsin_data(filepath):
    label_convert_func = lambda x: 1. if x == b'M' else 0.
    label_name_dict = {'M': 1, 'B': 0}
    numpy_array = genfromtxt(filepath, delimiter=',', dtype=np.float64,
        skip_header=1, converters={1: label_convert_func})
    numpy_array = numpy_array.view('<f8').reshape(len(numpy_array), -1)
    data = numpy_array[:, 2:]
    label = numpy_array[:, 1]
    print(label)
    return data, label, label_name_dict

