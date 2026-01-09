import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def one_hot(y_int, k):
    oh = np.zeros((y_int.size, k), dtype=np.float32)
    oh[np.arange(y_int.size), y_int] = 1.0
    return oh

class MLPGeneralizado:
    """
    Perceptrón Multicapa (NumPy) generalizado:
    - Múltiples capas y neuronas por capa
    - Activación en ocultas (ReLU / tanh)
    - Salida seleccionable: output="logistic" o output="softmax"

    Pensado para entregar como librería en el examen (basado en la práctica 5/6).
    """
    def __init__(self, layer_sizes, seed=13, l2=1e-4, hidden_activation="relu", output="softmax"):
        self.sizes = list(layer_sizes)  # [n_in, h1, h2, ..., n_out]
        self.rng = np.random.default_rng(seed)
        self.l2 = float(l2)
        self.hidden_activation = hidden_activation
        self.output = output

        self.W, self.b = [], []
        for i in range(len(self.sizes)-1):
            fan_in, fan_out = self.sizes[i], self.sizes[i+1]
            # He init para ReLU en ocultas
            if i < len(self.sizes)-2 and hidden_activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.W.append(self.rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros((1, fan_out), dtype=np.float32))

    def _act(self, z):
        if self.hidden_activation == "relu":
            return np.maximum(0, z)
        if self.hidden_activation == "tanh":
            return np.tanh(z)
        raise ValueError("hidden_activation no soportada")

    def _act_deriv(self, z, a):
        if self.hidden_activation == "relu":
            return (z > 0).astype(np.float32)
        if self.hidden_activation == "tanh":
            return 1.0 - a*a
        raise ValueError("hidden_activation no soportada")

    def forward(self, X):
        a = X.astype(np.float32)
        activations, zs = [a], []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            if i == len(self.W)-1:
                if self.output == "softmax":
                    a = softmax(z)
                else:
                    a = sigmoid(z)
            else:
                a = self._act(z)
            activations.append(a)
        return activations, zs

    def predict(self, X):
        out = self.forward(X)[0][-1]
        if self.output == "softmax":
            return np.argmax(out, axis=1)
        # logistic
        if out.shape[1] == 1:
            return (out.ravel() >= 0.5).astype(int)
        return np.argmax(out, axis=1)

    def fit(self, X, y, epochs=800, lr=0.05, batch_size=32, verbose_every=0):
        """
        Entrenamiento por mini-batch GD.
        - output="softmax": y debe ser vector de enteros (0..C-1) o one-hot.
        - output="logistic" con n_out=1: y debe ser {0,1}.
        """
        X = X.astype(np.float32)
        n = X.shape[0]

        if self.output == "softmax":
            if y.ndim == 1:
                y_mat = one_hot(y.astype(int), self.sizes[-1])
            else:
                y_mat = y.astype(np.float32)
        else:
            # logistic
            if self.sizes[-1] == 1:
                y_mat = y.astype(np.float32).reshape(-1, 1)
            else:
                y_mat = y.astype(np.float32)  # one-vs-all

        for ep in range(epochs):
            idx = self.rng.permutation(n)
            Xs, Ys = X[idx], y_mat[idx]

            for start in range(0, n, batch_size):
                xb = Xs[start:start+batch_size]
                yb = Ys[start:start+batch_size]

                activations, zs = self.forward(xb)
                out = activations[-1]

                # softmax+CE o sigmoid+BCE => delta = out - y
                delta = (out - yb) / xb.shape[0]

                dW = [None]*len(self.W)
                db = [None]*len(self.b)

                for i in reversed(range(len(self.W))):
                    a_prev = activations[i]
                    dW[i] = a_prev.T @ delta + self.l2 * self.W[i]
                    db[i] = np.sum(delta, axis=0, keepdims=True)

                    if i > 0:
                        delta = (delta @ self.W[i].T) * self._act_deriv(zs[i-1], activations[i])

                for i in range(len(self.W)):
                    self.W[i] -= lr * dW[i]
                    self.b[i] -= lr * db[i]

            if verbose_every and (ep % verbose_every == 0 or ep == epochs-1):
                pred = self.predict(X)
                if self.output == "softmax":
                    acc = (pred == np.argmax(y_mat, axis=1)).mean()
                else:
                    if self.sizes[-1] == 1:
                        acc = (pred == y_mat.ravel()).mean()
                    else:
                        acc = (pred == np.argmax(y_mat, axis=1)).mean()
                print(f"Epoch {ep:4d} | train acc = {acc:.3f}")

        return self
