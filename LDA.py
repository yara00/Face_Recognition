import numpy as np
from Classifier import Classifier
from Dataset import Dataset
from PGM import PGM
import pandas as pd


class LDA:
    def __init__(self, training_set, training_labels, test_set, test_labels, sample_size, classes):
        self.training_set = training_set
        self.training_labels = training_labels
        self.test_set = test_set
        self.test_labels = test_labels
        self.mean_vector = np.zeros(shape=(classes, 10304))
        self.sample_size = sample_size
        self.classes = classes

    def compute_mean_vector(self):
        sample_size = self.sample_size
        if self.classes > 2:
            sample_sum = self.training_set.reshape(-1, sample_size, self.training_set.shape[1]).sum(axis=1)
            self.mean_vector = sample_sum / sample_size
        else:
            first_class = self.training_set[:sample_size, :]
            second_class = self.training_set[sample_size:, :]
            u1 = np.mean(first_class, axis=0).reshape(10304, 1)
            u2 = np.mean(second_class, axis=0).reshape(10304, 1)
            self.mean_vector = np.concatenate((np.transpose(u1), np.transpose(u2)), axis=0)

        return self.mean_vector

    def compute_bscatter_matrix(self):
        sample_size = self.sample_size
        overall_mean = np.mean(self.training_set, axis=0)
        Sb = np.zeros(shape=(10304, 10304))  # Sb 10304x10304 --> between classes scatter matrix
        if self.classes > 2:
            for k in range(0, self.classes):
                delta_u = np.subtract(self.mean_vector[k], overall_mean)
                Sb += sample_size * np.matmul(delta_u, np.transpose(delta_u))
        else:
            delta_u = np.subtract(self.mean_vector[0], self.mean_vector[1])
            delta_u = delta_u.reshape(10304, 1)
            Sb = np.matmul(delta_u, np.transpose(delta_u))

        return Sb

    def compute_class_scatter_matrix(self):
        scatter_matrix = np.zeros(shape=(10304, 10304))
        for i in range(0, self.classes):
            Zi = np.subtract(self.training_set[i * self.sample_size:(i * self.sample_size) + self.sample_size, :],
                             self.mean_vector[i])
            Si = np.matmul(np.transpose(Zi), Zi)
            scatter_matrix = np.add(scatter_matrix, Si)  # S 10304x10304 --> within class scatter matrix

        return scatter_matrix

    def compute_eigens(self, scatter_matrix, Sb):
        eigen_values, eigen_vectors = np.linalg.eigh(np.matmul(np.linalg.inv(scatter_matrix), Sb))
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        eigen_vectors = eigen_vectors.T
        if self.classes > 2:
            eigen_vectors = eigen_vectors[:39, :]  # U 39x10304
        else:
            eigen_vectors = eigen_vectors[:1, :]  # U 1x10304

        return eigen_values, eigen_vectors

    def compute_projected_data(self, matrix, eigen_vectors):
        return np.matmul(matrix, np.transpose(eigen_vectors))

    def algorithm(self):
        self.compute_mean_vector()
        Sb = self.compute_bscatter_matrix()
        scatter_matrix = self.compute_class_scatter_matrix()
        eigen_values, eigen_vectors = self.compute_eigens(scatter_matrix, Sb)
        projected_training = self.compute_projected_data(self.training_set, eigen_vectors)
        projected_test = self.compute_projected_data(self.test_set, eigen_vectors)

        return projected_training, projected_test

    def map_faces(self, x):
        if x == 1:
            return "Face"
        else:
            return "Non_Face"

    def faces_vs_nonfaces(self, sample_size):
        training_set, training_labels, test_set, test_labels = Dataset().generate_matrix()
        matrix, labels = PGM().generate_nonface_imgs(sample_size)
        training_set2, training_labels2, test_set2, test_labels2 = Dataset().split_matrix(np.asarray(matrix), labels)

        training_set = np.concatenate((training_set, training_set2), axis=0)
        test_set = np.concatenate((test_set, test_set2), axis=0)
        training_labels = [1] * 200
        training_labels = training_labels + training_labels2
        test_labels = [1] * 200
        test_labels = test_labels + test_labels2

        lda = LDA(training_set, training_labels, test_set, test_labels, 200, 2)
        projected_training, projected_test = lda.algorithm()
        classifier = Classifier(1)
        score = classifier.classify(projected_training, training_labels, projected_test, test_labels)
        print("Faces vs Non-Faces LDA Score")
        print("Training set: 200 faces & ", sample_size * 5 / 2, " non-faces")
        print(score)
        predicted, status = classifier.success_failed_cases(projected_training, training_labels, projected_test,
                                                          test_labels)

        df = pd.DataFrame({
            'Labels': list(map(self.map_faces, test_labels)),
            'Predicted': list(map(self.map_faces, predicted)),
            'Status': status
        })
        df.to_csv("Succes_Fail" + str(sample_size) +".csv")

if __name__ == '__main__':
    training_set, training_labels, test_set, test_labels = Dataset().generate_matrix()
    lda = LDA(training_set, training_labels, test_set, test_labels, 5, 40)
    '''
    projected_training, projected_test = lda.algorithm()
    
    k = [1, 3, 5, 7]
    scores = []
    for n_neighbor in k:
        scores.append(Classifier(n_neighbor).classify(projected_training, training_labels, projected_test, test_labels))
    df = pd.DataFrame({
        'K': k,
        'scores': scores
    })
    print("Faces LDA Accuracy")
    print(df)
    '''

    lda.faces_vs_nonfaces(40)
    lda.faces_vs_nonfaces(80)
