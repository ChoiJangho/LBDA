import numpy as np
from dataio import import_breast_cancer_wisconsin_data
from algorithms import RelevantPatternSelection, \
    LinearBoundaryDiscriminantAnalysis, train_and_evaluate_nn_classifier
from sklearn.decomposition import PCA

LBDA_nn_non_accuracies = np.zeros((10, 30))
LBDA_nn_all_accuracies = np.zeros((10, 30))
PCA_accuracies = np.zeros(30)

filepath = 'breast-cancer-wisconsin-data/data.csv'
data, label, label_name_dict = import_breast_cancer_wisconsin_data(filepath)

for k_rps in range(1, 11):
    RPS = RelevantPatternSelection(k=k_rps)
    LBDA = LinearBoundaryDiscriminantAnalysis(k=k_rps)

    data_non_boundary, label_non_boundary,\
                data_boundary, label_boundary = RPS.run(data, label, 2)

    for feature_num in range(1, 31):
        transformation_LBDA = LBDA.run(data, label, 2, feature_num)

        LBDA_nn_non_accuracy = \
            train_and_evaluate_nn_classifier(
                data_non_boundary, label_non_boundary, transformation_LBDA
            )
        LBDA_nn_all_accuracy = \
            train_and_evaluate_nn_classifier(
                data, label, transformation_LBDA
            )

        LBDA_nn_non_accuracies[k_rps-1][feature_num-1] = LBDA_nn_non_accuracy
        LBDA_nn_all_accuracies[k_rps-1][feature_num-1] = LBDA_nn_all_accuracy

    pca = PCA(n_components=feature_num)
    projected_data_pca = pca.fit_transform(data)
    PCA_accuracy = \
        train_and_evaluate_nn_classifier(
            data, label
            )
    PCA_accuracies[feature_num-1] = PCA_accuracy

print(LBDA_nn_non_accuracies)
print(LBDA_nn_all_accuracies)
print(PCA_accuracies)
print(np.argmax(LBDA_nn_non_accuracies))
print(np.argmax(LBDA_nn_all_accuracies))
print(np.argmax(PCA_accuracies))
print(np.max(LBDA_nn_non_accuracies))
print(np.max(LBDA_nn_all_accuracies))
print(np.max(PCA_accuracies))
