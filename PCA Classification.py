import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as RandomizedPCA
import time

from Classifier import Classifier
from Dataset import Dataset
from PCA import PCA
from PGM import PGM


def calculate_pca_scores(training_set, training_labels, test_set, test_labels, name):
    satisfiability = [0.8, 0.85, 0.9, 0.95]
    k = [1, 3, 5, 7]
    pca = PCA(training_set)
    file = open(name, "w")
    for a in satisfiability:
        projected_training_set, projected_test_set = pca.project_data(training_set, test_set, a)
        scores = []
        print(f"alpha={a}:")
        file.write(f"alpha={a}:\n")
        for n_neighbor in k:
            scores.append(Classifier(n_neighbor).classify(projected_training_set, training_labels,
                                                          projected_test_set, test_labels))
        df = pd.DataFrame({
            'K': k,
            'scores': scores
        })
        print(df)
        file.write(df.to_string())
        file.write("\n")
    file.close()


def map_faces(x):
    if x == 1:
        return "Face"
    else:
        return "Non_Face"


if __name__ == '__main__':
    X_train_faces, y_train_faces, X_test_faces, y_test_faces = Dataset().generate_matrix()
    K = 1
    alpha = 0.95
    classifier = Classifier(K)
    start = time.time()
    pca_reducer = PCA(X_train_faces)
    proj_X_train_faces, proj_X_test_faces = pca_reducer.project_data(X_train_faces, X_test_faces, alpha)
    end = time.time()
    score = classifier.classify(proj_X_train_faces, y_train_faces, proj_X_test_faces, y_test_faces)
    print(f"Accuracy of PCA = {score}")
    print(f"Time Elapsed = {end - start} seconds")

    # Faces Classification
    X_train_faces, y_train_faces, X_test_faces, y_test_faces = Dataset().generate_matrix()
    calculate_pca_scores(X_train_faces, y_train_faces, X_test_faces, y_test_faces, "PCA Faces Classification")

    # Faces vs. Non-faces Classification
    # generate faces dataset
    y_train_faces = [1] * len(y_train_faces)
    y_test_faces = [1] * len(y_test_faces)

    sample_sizes = [20, 40, 80, 120, 160]
    alpha = 0.8
    K = 1
    accuracy = []
    classifier = Classifier(K)

    for sample_size in sample_sizes:
        # generate non-faces dataset
        non_faces_dataset, non_faces_labels = PGM().generate_nonface_imgs(sample_size)
        X_train_non_faces, y_train_non_faces, X_test_non_faces, y_test_non_faces = \
            Dataset().split_matrix(np.array(non_faces_dataset), non_faces_labels)

        # concatenate the 2 datasets together
        X_train = np.concatenate((X_train_faces, X_train_non_faces), axis=0)
        y_train = y_train_faces + y_train_non_faces
        X_test = np.concatenate((X_test_faces, X_test_non_faces), axis=0)
        y_test = y_test_faces + y_test_non_faces

        pca_reducer = PCA(X_train)
        proj_X_train, proj_X_test = pca_reducer.project_data(X_train, X_test, alpha)
        accuracy.append(classifier.classify(proj_X_train, y_train, proj_X_test, y_test))
        predicted, status = classifier.success_failed_cases(proj_X_train, y_train, proj_X_test, y_test)
        df = pd.DataFrame({
            'Labels': [map_faces(label) for label in y_test],
            'Predicted': [map_faces(pred) for pred in predicted],
            'Status': status
        })
        df.to_csv("Success_Fail_PCA" + str(sample_size) + ".csv")

    summary = pd.DataFrame({
        "Number of Non-Faces": sample_sizes,
        "accuracy": accuracy
    })
    f = open("faces vs non-faces classification", "w")
    f.write(summary.to_string())
    f.close()

    # BONUS: Data split into 7:3 ratio
    training_data_7_3, training_labels_7_3, test_data_7_3, test_labels_7_3 = Dataset().generate_split_data_70_30()
    pca_reducer = PCA(training_data_7_3)
    alphas = [0.8, 0.85, 0.9, 0.95]
    scores = []
    K = 1
    classifier = Classifier(K)
    for alpha in alphas:
        projected_training_data, projected_test_data = pca_reducer.project_data(training_data_7_3, test_data_7_3, alpha)
        scores.append(classifier.classify(projected_training_data, training_labels_7_3,
                                          projected_test_data, test_labels_7_3))
    summary = pd.DataFrame({
        'alpha': alphas,
        'Accuracy': scores
    })
    f = open("BONUS_70 30 split", "a")
    f.write("PCA:\n")
    f.write(summary.to_string())

    # BONUS 2: PCA Variation
    X_train, y_train, X_test, y_test = Dataset().generate_matrix()
    randomized_pca = RandomizedPCA(svd_solver='randomized')
    start = time.time()
    randomized_pca.fit(X_train)
    proj_X_train = randomized_pca.transform(X_train)
    proj_X_test = randomized_pca.transform(X_test)
    end = time.time()
    K = 1
    classifier = Classifier(K)
    score = classifier.classify(proj_X_train, y_train, proj_X_test, y_test)
    print(f"Accuracy of Randomized PCA = {score}")
    print(f"Time Elapsed = {end - start} seconds")
