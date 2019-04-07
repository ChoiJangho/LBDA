import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

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
        data_non_boundary = data[proximity_k < threshold]
        label_non_boundary = label[proximity_k < threshold]
        data_boundary = data[proximity_k >= threshold]
        label_boundary = label[proximity_k >= threshold]

        return data_non_boundary, label_non_boundary,\
            data_boundary, label_boundary


class LinearBoundaryDiscriminantAnalysis:
    def __init__(self, k):
        self._RPS = RelevantPatternSelection(k)

    def run(self, data, label, numclass, numfeature):
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
        dim = np.shape(data)[1]
        # Get non-boundary and boundary data from Relevant Pattern Selection.
        X_non_boundary, _, X_boundary, _ = self._RPS.run(data, label, numclass)

        n_NB = np.shape(X_non_boundary)[0]
        n_B = np.shape(X_boundary)[0]
        print("Number of non-boundary data is %d and boundary data is %d."
            % (n_NB, n_B))

        if numfeature > min(n_B + numclass - 1, dim):
            raise Exception("Number of target feature exceeds " +
                            "possible maximum value of LBDA.")

        # Initialization of between and within Scatter Matrices.
        S_between = np.zeros((dim, dim))
        S_within = np.zeros((dim, dim))

        for i in range(numclass):
            X_i = data[label==i, :]
            n_i = np.shape(X_i)[0]
            if n_i == 0:
                raise Exception("Data for label %d not provided." % i)
            mean_i = np.sum(X_i, axis=0) / n_i
            differences_i_boundary = X_boundary - np.tile(mean_i, (n_B, 1))
            S_between = S_between + np.matmul(
                differences_i_boundary.T, differences_i_boundary)
            differences_i_non_boundary = X_non_boundary \
                                         - np.tile(mean_i, (n_NB, 1))
            S_within = S_within + np.matmul(
                differences_i_non_boundary.T, differences_i_non_boundary
            )

        if np.linalg.det(S_within) == 0:
            print("Within Scatter Matrix is regularized to avoid singularity.")
            # Regularization
            alpha = 0.001
            S_within = S_within + alpha * np.identity(dim)

        eigvals, eigvecs = np.linalg.eig(np.matmul(np.linalg.inv(S_within), S_between))

        transformation_matrix = eigvecs[:, np.argsort(eigvals)[-numfeature:]]

        return transformation_matrix

def train_and_evaluate_nn_classifier(
        data, label, transformation_matrix=None, cross_validation='LOO'
    ):
    if transformation_matrix is not None:
        projected_data = np.matmul(data, transformation_matrix)
    else:
        projected_data = data
    correct_count = 0
    if cross_validation == 'LOO':
        datanum = projected_data.shape[0]
        for i in range(datanum):
            data_in_use = np.delete(projected_data, i, 0)
            label_in_use = np.delete(label, i, 0)
            data_evaluate = projected_data[i]
            label_evaluate = label[i]
            classifier = KNeighborsClassifier(n_neighbors=1)
            classifier.fit(data_in_use, label_in_use)
            correct_count += classifier.score(
                np.array([data_evaluate]), np.array([label_evaluate])
            )
        return correct_count / datanum


