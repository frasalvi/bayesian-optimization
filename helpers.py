import numpy as np


class Function2D():
    '''
    Abstract class, blueprint for 2d functions.
    '''
    def __call__(self, x, y):
        '''
        Executes the function, returning its value in (x, y).
        '''
        raise NotImplementedError
    def grad(self, x, y):
        '''
        Computes the gradient of the function in (x, y).
        '''
        raise NotImplementedError

class Rosenbrock(Function2D):
    '''
    Implements the Rosenbrock function.
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def __call__(self, x, y):
        return (self.a - x)**2 + self.b * (y - x**2)**2

    def grad(self, x, y):
        return (-2*(self.a - x) - 2 * self.b * (y - x**2) * 2 * x, self.b * 2 * (y - x**2))

class Rastrigin(Function2D):
    '''
    Implements the Rastrigin function in 2D.
    '''
    def __init__(self, A):
        self.A = A
    
    def __call__(self, x, y):
        return self.A * 2 + x**2 + y**2 - self.A * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    def grad(self, x, y):
        return (2*x + self.A * np.sin(2*np.pi*x) * 2*np.pi, 
                2*y + self.A * np.sin(2*np.pi*y) * 2*np.pi)


class Rastrigin1d:
    '''
    Implements the Rastrigin function in 1D.
    '''
    def __init__(self, A):
        self.A = A
    
    def __call__(self, x):
        return self.A * 1 + x**2 - self.A * (np.cos(2*np.pi*x))

    def grad(self, x):
        return (2*x + self.A * np.sin(2*np.pi*x) * 2*np.pi)


def gradient_descent(loss, initial_x, initial_y, learning_rate, max_iters, verbose=1):
    '''
    Performs gradient descent of an arbitrary function.

    Args:
        loss (Function2D): function to optimize 
        initial_x (float): x-coordinate of the initial point
        initial_y (float): y-coordinate of the initial point
        learning_rate (float): learning rate of the algorithm.
        max_iters (int): maximum number of iterations
        verbose (bool): if true, prints the loss every 100 iterations.
    
    Returns:
        (list): loss per iteration
        (list): point per iteration
    '''
    xs = [[initial_x, initial_y]]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    y = initial_y
    
    for iteration in range(max_iters):
        gradx, grady = loss.grad(x, y)

        # update x through the stochastic gradient update
        x -= learning_rate * gradx
        y -= learning_rate * grady

        # store x and objective
        xs.append([x, y])
        objective = loss(x, y)
        objectives.append(objective)
        
        if verbose and iteration % 100 == 0:
            print("GD({bi:04d}/{ti:04d}): objective = {l:}".format(
                  bi=iteration, ti=max_iters - 1, l=objective))
    return objectives, xs

