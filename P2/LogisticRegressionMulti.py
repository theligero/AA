import numpy as np
import copy
import math

from LinearRegressionMulti import LinearRegMulti

class LogisticRegMulti(LinearRegMulti):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y,w,b, lambda_):
        super().__init__(x, y,w,b,lambda_)
        # Si la base dejó x en 1D (por tener 1 feature), lo reformamos a (m, n= len(w))
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, self.w.shape[0])

    #  --- Función sigmoidal ---
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # --- Coste de cómputo ---
    def compute_cost(self):
        m, n = self.x.shape

        z = np.dot(self.x, self.w) + self.b     # (m,)
        f_wb = self.sigmoid(z)

        # Para evitar errores de log(0)
        epsilon = 1e-8
        cost = (-1 / m) * np.sum(
            self.y * np.log(f_wb + epsilon) + (1 - self.y) * np.log(1 - f_wb + epsilon)
        )

        # Regularización (evitar sesgo)
        reg_cost = (self.lambda_ / (2 * m)) * np.sum(np.square(self.w))

        total_cost = cost + reg_cost
        return total_cost
    
    # --- Gradiente de cómputo ---
    def compute_gradient(self):
        m, n = self.x.shape
        z = np.dot(self.x, self.w) + self.b
        f_wb = self.sigmoid(z)

        # Errores
        diff = f_wb - self.y        # (m,)

        # Gradientes
        dj_dw = (1 / m) * np.dot(self.x.T, diff) + (self.lambda_ / m) * self.w
        dj_db = (1 / m) * np.sum(diff)

        return dj_dw, dj_db

    
    

    
def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LogisticRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
