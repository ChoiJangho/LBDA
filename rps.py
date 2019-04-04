import numpy as np
from scipy.spatial.distance import cdist
class RelevantPatternSelection:
    def __init__(self, k):
        self._k = k

    def threshold(self, numclass):
        return 1 - 1 / numclass

    def run(self, data, label, numclass):
        """
        :param data: 2D numpy array of input data.
            dimension: (numdata x input dimension)
        :param label: 1D numpy array of label. Length is numdata
        """
        if np.shape(data)[0] != np.shape(label)[0]:
            raise Exception("data and label do not match the size.")
        if np.max(label) >= numclass:
            raise Exception("labels should be in range (0 - (numclass - 1))")
        numdata = np.shape(data)[0]
        threshold = self.threshold(numclass)
        print("proximity threshold: ", threshold)

        distances = cdist(data, data, 'euclidean')
        k_neighbors = np.argsort(distances, axis=1)[:, 0:self._k+1]
        p = np.zeros((numclass, numdata))
        for j in range(numclass):
            for i in range(numdata):
                class_list, class_count = np.unique(
                    label[k_neighbors[i][:]],
                    return_counts=True)
                # When class_count = 0, p should be 0,
                # but here we use 1 as a trick to avoid
                # zero-division in log in the next line
                p[j][i] = class_count[np.argwhere(class_list==j)] \
                    / (self._k + 1) \
                    if any(class_list==j) else 1
        proximity_k = np.sum(- p * np.log(p) / np.log(numclass), axis = 0)
        print(proximity_k)
        data_non_boundary = data[proximity_k < threshold]
        data_boundary = data[proximity_k > threshold]
        return data_non_boundary, data_boundary
