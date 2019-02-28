import numpy as np
from sklearn.neighbors import KDTree

class SOM:
    def __init__(self, data, x_axis, y_axis, n_neighbors, epochs, learning_rate):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.n_features = data.shape[1]
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _weight_initialization(self):
        weights = np.random.rand(self.x_axis*self.y_axis, self.n_features)

        norm = np.linalg.norm(weights, axis=1)
        for i in range(weights.shape[0]):
            weights[i] = weights[i]/norm[i]

        return weights

    def _update_weights(self, record, weights, indexes):
        weights[indexes] = weights[indexes] + self.learning_rate*(record - weights[indexes])

        norm = np.linalg.norm(weights[indexes])
        weights[indexes] = weights[indexes]/norm

        return weights

    def _create_kdtree(self, weights):
        return KDTree(weights)

    def _neighbors(self, query_record, kdtree):
        return kdtree.query(query_record, k=self.n_neighbors)

    def train(self, data):
        # initialize weights
        weights = self._weight_initialization()

        # store winning nodes in a dictonary
        # {record index, winning node index}
        winners = []
        distances = []

        # iterate over each epoch
        for ep in range(self.epochs):
            # iterate over each record
            for i, record in enumerate(data):
                # create our kdtree to query from
                kdtree = self._create_kdtree(weights)

                # find nearest neighbors
                dist, ind = self._neighbors(record.reshape(1,-1), kdtree)

                if ep == self.epochs - 1:
                    winners.append(ind[0][0])
                    distances.append(dist[0][0])

                # update weights
                weights = self._update_weights(record, weights, ind)

        return kdtree, weights, winners, distances

    def predict(self, data, tree):
        dist, ind = tree.query(data, k=1)

        return dist, ind
