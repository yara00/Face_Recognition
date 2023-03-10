import numpy as np
import pandas as pd

from Classifier import Classifier
from Dataset import Dataset
from PCA import PCA
from PGM import PGM


def calculate_pca_scores(training_set, training_labels, test_set, test_labels, name):
    satisfiability = [0.8, 0.85, 0.9, 0.95]
    k = [1, 3, 5, 7]
    pca = PCA(training_set)
    file = open(name, "w")
    for alpha in satisfiability:
        projected_training_set, projected_test_set = pca.project_data(training_set, test_set, alpha)
        scores = []
        print(f"alpha={alpha}:")
        file.write(f"alpha={alpha}:\n")
        for n_neighbor in k:
            scores.append(Classifier(n_neighbor).classify(projected_training_set, training_labels, projected_test_set, test_labels))
        df = pd.DataFrame({
            'K': k,
            'scores': scores
        })
        print(df)
        file.write(df.to_string())
        file.write("\n")
    file.close()


if __name__ == '__main__':

    # Faces Classification
    X_train_faces, y_train_faces, X_test_faces, y_test_faces = Dataset().generate_matrix()
    calculate_pca_scores(X_train_faces, y_train_faces, X_test_faces, y_test_faces, "PCA Faces Classification")

    # Faces vs. Non-faces Classification
    # generate faces dataset
    y_train_faces = [1] * len(y_train_faces)
    y_test_faces = [1] * len(y_test_faces)

    # generate non-faces dataset
    non_faces_dataset, non_faces_labels = PGM().generate_nonface_imgs()
    X_train_non_faces, y_train_non_faces, X_test_non_faces, y_test_non_faces = Dataset().split_matrix(np.array(non_faces_dataset), non_faces_labels)

    # concatenate the 2 datasets together
    X_train = np.concatenate((X_train_faces, X_train_non_faces), axis=0)
    y_train = y_train_faces + y_train_non_faces
    X_test = np.concatenate((X_test_faces, X_test_non_faces), axis=0)
    y_test = y_test_faces + y_test_non_faces

    calculate_pca_scores(X_train, y_train, X_test, y_test, "PCA Faces vs. Non-faces Classification")


