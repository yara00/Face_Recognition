import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Classifier import Classifier
from Dataset import Dataset


class LDA:
    def __init__(self, training_set, training_labels, test_set, test_labels):
        self.training_set = training_set;
        self.training_labels = training_labels
        self.test_set = test_set
        self.test_labels = test_labels
        self.mean_vector = np.zeros(shape=(40, 10304))

    def compute_mean_vector(self):
        sample_size = 5
        sample_sum = self.training_set.reshape(-1, sample_size, self.training_set.shape[1]).sum(axis=1)
        self.mean_vector = sample_sum / sample_size
        print("Mean vector")
        print(self.mean_vector.shape)
        return self.mean_vector

    def compute_bscatter_matrix(self):
        sample_size = 5
        overall_mean = np.mean(self.training_set, axis=0)
        print("Overall mean")
        print(overall_mean)
        Sb = np.zeros(shape=(10304, 10304))
        for k in range(0, 40):
            delta_u = np.subtract(self.mean_vector[k], overall_mean)
            Sb += sample_size * np.matmul(delta_u, np.transpose(delta_u))
        print("Sb")
        print(Sb)
        return Sb

    def compute_class_scatter_matrix(self):
        scatter_matrix = np.zeros(shape=(10304, 10304))
        for i in range(0, 40):
            Zi = np.subtract(self.training_set[i * 5:(i * 5) + 5, :], self.mean_vector[i])
            Si = np.matmul(np.transpose(Zi), Zi)
            scatter_matrix = np.add(scatter_matrix, Si)
        return scatter_matrix

    def compute_eigens(self, scatter_matrix, Sb):
        eigen_values, eigen_vectors = np.linalg.eig(np.matmul(np.linalg.inv(scatter_matrix), Sb))
        idx = np.abs(eigen_values).argsort()[::-1]
        eigen_vectors = eigen_vectors[idx]
        eigen_vectors = eigen_vectors[:-1, :]
        return eigen_values, eigen_vectors

    def compute_projected_data(self, matrix, eigen_vectors):
        return np.dot(matrix, np.transpose(eigen_vectors).real)

    def algorithm(self):
        self.compute_mean_vector()
        Sb = self.compute_bscatter_matrix()
        scatter_matrix = self.compute_class_scatter_matrix()
        eigen_values, eigen_vectors = self.compute_eigens(scatter_matrix, Sb)
        projected_training = self.compute_projected_data(self.training_set, eigen_vectors)
        projected_test = self.compute_projected_data(self.test_set, eigen_vectors)
        classifier = Classifier(1)
        return classifier.classify(projected_training, self.training_labels, projected_test, self.test_labels)


if __name__ == '__main__':
    training_set, training_labels, test_set, test_labels = Dataset().split_matrix()
    lda = LDA(training_set, training_labels, test_set, test_labels)
    score = lda.algorithm()
    print(score)




