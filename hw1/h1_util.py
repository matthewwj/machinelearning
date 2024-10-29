import os
import numpy as np

from sklearn.datasets import fetch_openml
# Load data from https://www.openml.org/d/554

def load_digits_train_data():
    mnist = fetch_openml('mnist_784')
    xm = mnist.data
    ym = mnist.target

    x_train = xm.to_numpy()[0:60000, :]/256.0
    y_train = ym.to_numpy()[0:60000].astype(int)

    return x_train, y_train

def load_digits_test_data():
    xm, ym = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    x_test = xm[60001:, :]/256.0
    y_test = ym[60001:].squeeze().astype(int)

    return x_test, y_test


def print_score(classifier, x_train, x_test, y_train, y_test):
    """ Simple print score function that prints train and test score of classifier - almost not worth it"""
    print('In Sample Score (accuracy): ',
          classifier.score(x_train, y_train))
    print('Test Score (accuracy): ',
          classifier.score(x_test, y_test))
    
def export_fig(fig, name):
    result_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)


def numerical_grad_check(f, x):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-4
    # d = x.shape[0]
    cost, grad = f(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        dim = it.multi_index
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        print('grad, num_grad, grad-num_grad', grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

    
            
            
    
