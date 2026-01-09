import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

class MLPGeneral:
    """
    MLP generalizado (múltiples capas) con salida seleccionable:
    - output="logistic"  -> sigmoid en la salida (binario o one-vs-all)
    - output="softmax"   -> softmax en la salida (multi-clase)

    Pensado para que puedas partir del estilo de tu MLP.py (prácticas 5/6),
    pero ya con:
    - lista de capas (layer_sizes)
    - opción de softmax vs logistic en la SALIDA
    """
    def __init__(self, layer_sizes, seed=0, l2=1e-5, hidden_activation="relu", output="logistic"):
        self.sizes = list(layer_sizes)  # [n_in, h1, h2, ..., n_out]
        self.rng = np.random.default_rng(seed)
        self.l2 = float(l2)
        self.hidden_activation = hidden_activation
        self.output = output

        self.W = []
        self.b = []
        for i in range(len(self.sizes) - 1):
            fan_in = self.sizes[i]
            fan_out = self.sizes[i+1]
            # He init para ReLU en ocultas
            if i < len(self.sizes) - 2 and hidden_activation == "relu":
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
        if self.hidden_activation == "sigmoid":
            return sigmoid(z)
        raise ValueError("hidden_activation no soportada")

    def _act_deriv(self, z, a):
        if self.hidden_activation == "relu":
            return (z > 0).astype(np.float32)
        if self.hidden_activation == "tanh":
            return 1.0 - a*a
        if self.hidden_activation == "sigmoid":
            return a*(1.0-a)
        raise ValueError("hidden_activation no soportada")

    def forward(self, X):
        a = X.astype(np.float32)
        activations = [a]
        zs = []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            if i == len(self.W) - 1:
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

    def fit(self, X, y, epochs=1000, lr=0.01, batch_size=32, verbose_every=0):
        """
        Entrena por mini-batch GD.
        - Si output="softmax": espera y en one-hot (m, n_out) y usa cross-entropy.
        - Si output="logistic":
            - si n_out==1: espera y en {0,1} (m,) y usa BCE.
            - si n_out>1: espera y one-hot y usa entropía cruzada "one-vs-all" (sigmoid independiente).
        """
        X = X.astype(np.float32)
        n = X.shape[0]

        if self.output == "softmax":
            y_mat = y.astype(np.float32)  # one-hot
        else:
            # logistic
            if self.sizes[-1] == 1:
                y_mat = y.astype(np.float32).reshape(-1, 1)
            else:
                y_mat = y.astype(np.float32)  # one-hot

        for ep in range(epochs):
            idx = self.rng.permutation(n)
            Xs = X[idx]
            Ys = y_mat[idx]

            for start in range(0, n, batch_size):
                xb = Xs[start:start+batch_size]
                yb = Ys[start:start+batch_size]

                activations, zs = self.forward(xb)
                out = activations[-1]

                # delta en salida
                if self.output == "softmax":
                    # softmax + cross-entropy
                    delta = (out - yb) / xb.shape[0]
                else:
                    # sigmoid + BCE / one-vs-all
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
                if self.sizes[-1] == 1 and self.output != "softmax":
                    acc = (pred == y.reshape(-1)).mean()
                else:
                    acc = (pred == np.argmax(y_mat, axis=1)).mean()
                print(f"Epoch {ep:4d} | train acc = {acc:.3f}")

        return self
