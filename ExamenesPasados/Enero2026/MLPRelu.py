import numpy as np
from scipy.special import xlogy


class MLPRelu:

    """
    Constructor: Computes MLPRelu.

    Args:
        inputLayer (int): size of input
        hiddenLayer (int): size of hidden layer.
        outputLayer (int): size of output layer
        seed (scalar): seed of the random numeric.
        epislom (scalar): random initialization range. e.j: 1 = [-1..1], 2 = [-2,2]...
        function = "sigmoid" | "relu"
        out_function = "sigmoid" | "softmax"
    """

    def __init__(
        self,
        inputLayer,
        hiddenLayer,
        outputLayer,
        seed=0,
        epislom=0.12,
        function="sigmoid",
        out_function="sigmoid"
    ):
        np.random.seed(seed)

        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.eps = epislom

        self.function = function
        self.out_function = out_function

        self.theta1 = np.random.uniform(
            -self.eps,
            self.eps,
            (hiddenLayer, inputLayer + 1)
        )

        self.theta2 = np.random.uniform(
            -self.eps,
            self.eps,
            (outputLayer, hiddenLayer + 1)
        )

    def new_trained(self, theta1, theta2):
        self.theta1 = theta1
        self.theta2 = theta2

    def _size(self, x):
        return x.shape[0]

    def _sigmoid(self, z):
        z = np.asarray(z)

        # Evita overflow en exp
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def _sigmoidPrime(self, a):
        a = np.asarray(a)
        return a * (1.0 - a)

    def _relu(self, z):
        z = np.asarray(z)
        return np.maximum(0, z)

    def _reluPrime(self, a):
        a = np.asarray(a)

        # Derivada indicada en el enunciado: (a > 0) * 1
        return (a > 0) * 1.0

    def _softmax(self, z):
        z = np.asarray(z)

        # Versión estable numéricamente
        z = z - np.max(z, axis=1, keepdims=True)

        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _hidden_activation(self, z):
        if self.function == "sigmoid":
            return self._sigmoid(z)

        if self.function == "relu":
            return self._relu(z)

        raise ValueError("function debe ser 'sigmoid' o 'relu'")

    def _hidden_activation_prime(self, a):
        if self.function == "sigmoid":
            return self._sigmoidPrime(a)

        if self.function == "relu":
            return self._reluPrime(a)

        raise ValueError("function debe ser 'sigmoid' o 'relu'")

    def _output_activation(self, z):
        if self.out_function == "sigmoid":
            return self._sigmoid(z)

        if self.out_function == "softmax":
            return self._softmax(z)

        raise ValueError("out_function debe ser 'sigmoid' o 'softmax'")

    def feedforward(self, x):
        x = np.asarray(x)

        m = self._size(x)

        # Capa de entrada con bias
        a1 = np.hstack([np.ones((m, 1)), x])

        # Capa oculta
        z2 = a1 @ self.theta1.T
        a2_no_bias = self._hidden_activation(z2)
        a2 = np.hstack([np.ones((m, 1)), a2_no_bias])

        # Capa de salida
        z3 = a2 @ self.theta2.T
        a3 = self._output_activation(z3)

        return a1, a2, a3, z2, z3

    def compute_cost(self, yPrime, y, lambda_):
        yPrime = np.asarray(yPrime)
        y = np.asarray(y)

        m = y.shape[0]

        # Evitamos log(0)
        eps = 1e-12
        yPrime = np.clip(yPrime, eps, 1 - eps)

        if self.out_function == "softmax":
            # Cross entropy multiclase para softmax
            J = -(1.0 / m) * np.sum(xlogy(y, yPrime))
        else:
            # Cross entropy binaria aplicada por salida, como en el MLP original
            J = -(1.0 / m) * np.sum(
                xlogy(y, yPrime) + xlogy(1 - y, 1 - yPrime)
            )

        J += self._regularizationL2Cost(m, lambda_)

        return J

    def predict(self, a3):
        a3 = np.asarray(a3)
        return np.argmax(a3, axis=1)

    def compute_gradients(self, x, y, lambda_):
        x = np.asarray(x)
        y = np.asarray(y)

        m = x.shape[0]

        # Forward
        a1, a2, a3, z2, z3 = self.feedforward(x)

        # Coste
        J = self.compute_cost(a3, y, lambda_)

        # Para sigmoid + cross entropy y softmax + cross entropy:
        # delta de salida = prediccion - valor_real
        delta3 = a3 - y

        delta2_full = delta3 @ self.theta2

        # Quitamos columna de bias
        delta2 = delta2_full[:, 1:] * self._hidden_activation_prime(a2[:, 1:])

        # Gradientes
        Delta1 = delta2.T @ a1
        Delta2 = delta3.T @ a2

        grad1 = Delta1 / m
        grad2 = Delta2 / m

        grad1 += self._regularizationL2Gradient(self.theta1, lambda_, m)
        grad2 += self._regularizationL2Gradient(self.theta2, lambda_, m)

        return J, grad1, grad2

    def _regularizationL2Gradient(self, theta, lambda_, m):
        reg = np.zeros_like(theta)

        if lambda_ != 0:
            reg[:, 1:] = (lambda_ / m) * theta[:, 1:]

        return reg

    def _regularizationL2Cost(self, m, lambda_):
        if lambda_ == 0:
            return 0.0

        reg1 = np.sum(self.theta1[:, 1:] ** 2)
        reg2 = np.sum(self.theta2[:, 1:] ** 2)

        return (lambda_ / (2.0 * m)) * (reg1 + reg2)

    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        Jhistory = []

        for i in range(numIte):
            J, grad1, grad2 = self.compute_gradients(x, y, lambda_)

            self.theta1 -= alpha * grad1
            self.theta2 -= alpha * grad2

            Jhistory.append(J)

            if verbose > 0:
                if i % verbose == 0 or i == numIte - 1:
                    print(f"Iteration {(i+1):6}: Cost {float(J):8.4f}")

        return Jhistory