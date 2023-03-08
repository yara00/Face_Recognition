from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, neighbours):
        self.neighbours = neighbours

    def classify(self, training_set, training_labels, test_set, test_labels):
        knn = KNeighborsClassifier(n_neighbors=self.neighbours)
        knn.fit(training_set, training_labels)
        # Calculate the accuracy of the model
        return knn.score(test_set, test_labels)