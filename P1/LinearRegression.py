import numpy as np
import copy
import math

class LinearReg:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
    """
    def __init__(self, x, y,w,b):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64)

        if self.x.ndim == 2 and self.x.shape[1] == 1:
            self.x = self.x.ravel()

        self.b = float(b)

    # --- Regularización (base: sin regularización) ---
    def _regularizationL2Cost(self):
        return 0.0
    
    def _regularizationL2Gradient(self):
        # Debe devolver 0 con misma dimensionalidad que w (escalar o vector)
        if np.isscalar(self.w) or (np.ndim(self.w)) == 0:
            return 0.0
        return np.zeros_like(self.w, dtype=np.float64)

    """
    Computes the linear regression function.

    Args:
        x (ndarray): Shape (m,) Input to the model
    
    Returns:
        the linear regression value
    """
    def f_w_b(self, x):
        return self.w * x + self.b


    """
    Computes the cost function for linear regression.

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    def compute_cost(self):
        m = self.y.shape[0]
        f = self.f_w_b(self.x)
        err = f - self.y
        data_cost = (1.0 / (2 * m)) * np.sum(err ** 2)
        return data_cost + self._regularizationL2Cost()
    

    """
    Computes the gradient for linear regression 
    Args:

    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    def compute_gradient(self):
        m = self.y.shape[0]
        f = self.f_w_b(self.x)
        err = f- self.y
        # Soporta x (m,) o X (m,n)
        if self.x.ndim == 1:
            dj_dw = (1.0 / m) * np.dot(err, self.x)
        else:
            dj_dw = (1.0 / m) * (self.x.T @ err)
        dj_db = (1.0 / m) * np.sum(err)
        # Añadir gradiente de regularización sobre w
        reg = self._regularizationL2Gradient()
        dj_dw = dj_dw + reg
        return dj_dw, dj_db
    
    
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
      w_initial : (ndarray): Shape (1,) initial w value before running gradient descent
      b_initial : (scalar) initial b value before running gradient descent
    """
    def gradient_descent(self, alpha, num_iters):
        # An array to store cost J and w's at each iteration — primarily for graphing later
        J_history = []
        w_initial = copy.deepcopy(self.w)  # avoid modifying global w within function
        b_initial = copy.deepcopy(self.b)  # avoid modifying global w within function
        for _ in range(num_iters):
            dj_dw, dj_db = self.compute_gradient()
            self.w = self.w - alpha * dj_dw
            self.b = self.b - alpha * dj_db
            J_history.append(self.compute_cost())
        return self.w, self.b, J_history, w_initial, b_initial


def cost_test_obj(x,y,w_init,b_init):
    lr = LinearReg(x,y,w_init,b_init)
    return lr.compute_cost()

def compute_gradient_obj(x,y,w_init,b_init):
    lr = LinearReg(x,y,w_init,b_init)
    return lr.compute_gradient()
