import numpy as np


class MLPGeneral:

    def __init__(self, inputLayer, hiddenLayers, outputLayer, seed=0, epislom=0.12):
        np.random.seed(seed)

        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer
        self.eps = epislom

        # Lista con tamaños de todas las capas
        # Ejemplo: [28, 64, 32, 16, 4]
        self.layers = [inputLayer] + hiddenLayers + [outputLayer]

        # Lista de matrices theta
        self.thetas = []

        for i in range(len(self.layers) - 1):
            rows = self.layers[i + 1]
            cols = self.layers[i] + 1  # +1 por el bias

            theta = np.random.uniform(
                -self.eps,
                self.eps,
                (rows, cols)
            )

            self.thetas.append(theta)

    def _sigmoid(self, z):
        z = np.asarray(z)
        return 1 / (1 + np.exp(-z))

    def _sigmoidPrime(self, a):
        a = np.asarray(a)
        return a * (1.0 - a)

    def feedforward(self, x):
        x = np.asarray(x)

        activations = []
        zs = []

        # Activación inicial
        a = x
        activations.append(a)

        # Recorremos todas las capas
        for theta in self.thetas:
            m = a.shape[0]

            # Añadir bias
            a_bias = np.hstack([np.ones((m, 1)), a])

            z = a_bias @ theta.T
            a = self._sigmoid(z)

            zs.append(z)
            activations.append(a)

        return activations, zs

    def compute_cost(self, yPrime, y, lambda_):
        yPrime = np.asarray(yPrime)
        y = np.asarray(y)

        m = y.shape[0]
        eps = 1e-8

        J = -(1.0 / m) * np.sum(
            y * np.log(yPrime + eps) + (1 - y) * np.log(1 - yPrime + eps)
        )

        J += self._regularizationL2Cost(m, lambda_)

        return J

    def _regularizationL2Cost(self, m, lambda_):
        if lambda_ == 0:
            return 0.0

        total = 0.0

        for theta in self.thetas:
            total += np.sum(theta[:, 1:] ** 2)

        return (lambda_ / (2.0 * m)) * total

    def _regularizationL2Gradient(self, theta, lambda_, m):
        reg = np.zeros_like(theta)

        if lambda_ != 0:
            reg[:, 1:] = (lambda_ / m) * theta[:, 1:]

        return reg

    def compute_gradients(self, x, y, lambda_):
        x = np.asarray(x)
        y = np.asarray(y)

        m = x.shape[0]

        activations, zs = self.feedforward(x)

        a_final = activations[-1]
        J = self.compute_cost(a_final, y, lambda_)

        # Lista de deltas
        deltas = [None] * len(self.thetas)

        # Delta de salida
        deltas[-1] = a_final - y

        # Deltas de capas ocultas
        for i in range(len(self.thetas) - 2, -1, -1):
            theta_next = self.thetas[i + 1]

            delta_next = deltas[i + 1]

            # Quitamos la columna de bias de theta_next
            theta_next_no_bias = theta_next[:, 1:]

            a_hidden = activations[i + 1]

            deltas[i] = (delta_next @ theta_next_no_bias) * self._sigmoidPrime(a_hidden)

        # Gradientes
        grads = []

        for i in range(len(self.thetas)):
            a_prev = activations[i]
            a_prev_bias = np.hstack([np.ones((m, 1)), a_prev])

            grad = (deltas[i].T @ a_prev_bias) / m

            grad += self._regularizationL2Gradient(self.thetas[i], lambda_, m)

            grads.append(grad)

        return J, grads

    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        Jhistory = []

        for i in range(numIte):
            J, grads = self.compute_gradients(x, y, lambda_)

            for j in range(len(self.thetas)):
                self.thetas[j] -= alpha * grads[j]

            Jhistory.append(J)

            if verbose > 0:
                if i % verbose == 0 or i == numIte - 1:
                    print(f"Iteration {(i+1):6}: Cost {float(J):8.4f}")

        return Jhistory

    def predict(self, a):
        a = np.asarray(a)
        return np.argmax(a, axis=1)