import os
import numpy as np

from hw1.h1_util import numerical_grad_check

### Static functions ###
def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """

    logi = 1 / (1 + np.exp(-z))

    #assert logi.shape == z.shape
    return logi


def cost_grad(x, y, w, lambda_=0.01):
    """Compute the binary cross-entropy loss and its gradient."""

    y_binary = (y + 1) / 2

    n = x.shape[0]

    z = x @ w
    h = logistic(z)
    h = np.clip(h, 1e-9, 1 - 1e-9)  # avoid log(0)

    cost = -np.mean(y_binary * np.log(h) + (1 - y_binary) * np.log(1 - h)) + (lambda_ / 2) * np.sum(w ** 2)

    gradient = (x.T @ (h - y_binary)) / n #+ lambda_ * w  # regularization of gradient

    #assert gradient.shape == w.shape
    return cost, gradient

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def tune_hyperparameters(X_train, y_train, X_test, y_test):

    param_grid = {
        'C': [0.01, 0.1, 1, 3, 5, 10],   # learning rate
        'max_iter': [25, 50, 100, 200],  # epochs
        'solver': ['lbfgs', 'saga']
    }

    log_reg = LogisticRegression()

    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")


class LogisticRegressionClassifier:

    def __init__(self):
        self.history = []
        self.w = None

    def predict(self, x):
        """ Classify each data element in x.

        Args:
            x: np.array shape (n,d) dtype float - Features
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on x (-1, 1). NOTE: We want a class here,
           not a probability between 0 and 1. You should thus return the most likely class!

        """

        z = np.dot(x, self.w)

        probabilities = logistic(z)

        # Convert to class labels (-1, 1)
        out = np.where(probabilities >= 0.5, 1, -1)

        return out
    
    def score(self, x, y):
        """ Compute model accuracy  on data x with labels y

        Args:
            x: np.array shape (n,d) dtype float - Features
            y: np.array shape (n,) dtype int - Labels 

        Returns: 
           s: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!

        """

        predictions = self.predict(x)
        correct_predictions = np.sum(predictions == y)
        s = correct_predictions / len(y)

        return s


    def fit(self, x, y, w=None, lr=0.1, batch_size=16, epochs=10, patience=25):

        n, d = x.shape
        if w is None:
            w = np.random.normal(0, 0.01, d)
        self.w = w
        self.history = []

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            permutation = np.random.permutation(n)
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            for i in range(0, n, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                loss, gradient = cost_grad(x_batch, y_batch, self.w)

                self.w -= lr * gradient

            self.history.append(loss)

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}')

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Stopping early with patience {patience_counter} over threshold {patience}!")
                    break

        return self.w, self.history

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    x = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([-1, -1, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', x.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = cost_grad(x, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    x = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    w = np.array([0.0, 0.0])
    y = np.array([-1, -1, 1]).astype('int64')
    print('shapes', x.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: cost_grad(x, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')


    
if __name__ == '__main__':

    print(os.environ['PATH'])

    test_logistic()
    test_cost()
    test_grad()
    
    
