import numpy as np
from rps import RelevantPatternSelection

rps = RelevantPatternSelection(k=5)

size_class_0 = 20
random_data_class_0 = np.random.multivariate_normal(
    mean=[1,0,0],
    cov=[[0.5, 0, 0],[0, 0.2, 0], [0, 0, 0.3]],
    size=size_class_0
)

size_class_1 = 25
random_data_class_1 = np.random.multivariate_normal(
    mean=[0,0,2],
    cov=[[0.2, 0, 0],[0, 1, 0.4], [0, 0.4, 0.5]],
    size=size_class_1
)
data = np.vstack((random_data_class_0, random_data_class_1))
label = np.concatenate((np.zeros(size_class_0), np.ones(size_class_1)))
random_order = np.random.shuffle(np.arange(np.shape(data)[0]))
data = np.squeeze(data[random_order], axis=0)
label = np.squeeze(label[random_order], axis=0)
print("data: ", data)
print("label: ", label)
data_non_boundary, data_boundary = rps.run(data, label, 2)
print("Data from non-boundary: ", data_non_boundary)
print("Data from boundary: ", data_boundary)

