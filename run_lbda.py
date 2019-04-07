from dataio import import_breast_cancer_wisconsin_data
from algorithms import RelevantPatternSelection, \
    LinearBoundaryDiscriminantAnalysis

LBDA = LinearBoundaryDiscriminantAnalysis(k=5)

filepath = 'breast-cancer-wisconsin-data/data.csv'
data, label, label_name_dict = import_breast_cancer_wisconsin_data(filepath)

LBDA.run(data, label, 2, 15)


