import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

def gradient_descent(function, initial_x, initial_y, learning_rate, max_iters, print_p=True):
    """Perform gradient descent on input function
       
       parameters:
           * function: function class with __call__ and grad methods, the objective to run GD on
           * initial_x: float, Starting point x
           * initial_y: float, Starting point y
           * learning_rate: float, learning rate of GD
           * max_iters: int, What number of iterations to run GD for
           * print_p: bool, to print status or not
    """
    
    xs = [[initial_x, initial_y]]  # parameters after each update 
    objectives = []  # loss values after each update
    x = initial_x
    y = initial_y
    
    for iteration in range(max_iters):
        gradx, grady = function.grad(x, y)

        # update x through the stochastic gradient update
        x -= learning_rate * gradx
        y -= learning_rate * grady

        # store x and objective
        xs.append([x, y])
        objective = function(x, y)
        objectives.append(objective)
        
        if iteration % 100 == 0 and print_p:
            print("GD({bi:04d}/{ti:04d}): objective = {l:}".format(
                  bi=iteration, ti=max_iters - 1, l=objective))
    return objectives, xs


class Rosenbrock():
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def __call__(self, x, y):
        return (self.a - x)**2 + self.b * (y - x**2)**2

    def grad(self, x, y):
        return (-2*(self.a - x) - 2 * self.b * (y - x**2) * 2 * x, self.b * 2 * (y - x**2))

def plot_w_CI(BO_losses, GD_losses, max_num_iterations, title, xlabel, ylabel, figname):
    """
       Helper for plotting losses from bayesian optimization compared to random initialization
       
       parameters:
           * BO_losses: numpy array with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * GD_losses: numpy array with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * title: str, title of plot
           * xlabel: str, x-label of plot
           * ylabel: str, y-label of plot
           * figname: str, path to save image
    """
    
    # Means
    mean_BO_losses = np.mean(BO_losses, axis=1)
    mean_GD_losses = np.mean(GD_losses, axis=1)

    # 0.95 confidence intervals
    CI_BO = np.zeros((max_num_iterations, 2))
    CI_GD = np.zeros((max_num_iterations, 2))
    for i in range(max_num_iterations):
      BO = BO_losses[i,:].copy()
      GD = GD_losses[i,:].copy()
      CI_BO[i,:] = st.t.interval(0.95, len(BO)-1, loc=np.mean(BO), scale=st.sem(BO))
      CI_GD[i,:] = st.t.interval(0.95, len(GD)-1, loc=np.mean(GD), scale=st.sem(GD))

    # Plot
    x = np.arange(1, max_num_iterations+1)
    plt.plot(x, mean_BO_losses, label='UCB GP, $\kappa = 0.1$')
    plt.plot(x, mean_GD_losses, linestyle="-.", label="Random")
    plt.fill_between(x, y1=CI_BO[:,1], y2=CI_BO[:,0], color='blue', alpha=0.2, label="0.95 CI for UCB GP")
    plt.fill_between(x, y1=CI_GD[:,1], y2=CI_GD[:,0], color='orange', alpha=0.2, label="0.95 CI for Random")

    # Plot description
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figname, bbox_inches='tight')

def plot_w_CI_subplot(BO_losses, GD_losses, max_num_iterations, main_title, titles, xlabels, ylabels, figname):
    """
       Helper for plotting several settings of bayesian optimization compared to random initialization
       
       parameters:
           * BO_losses: list of numpy arrays with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * GD_losses: list of numpy arrays with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * main_title: Title for entire figure
           * titles: list of str, title of each subplot
           * xlabel: list of str, x-label of each subplot (note only bottom is displayed)
           * ylabel: list of str, y-label of each subplot
           * figname: str, path to save image
    """
    
    # Means
    fig, ax = plt.subplots(len(BO_losses), figsize=(8,8))
    for j in range(len(BO_losses)):
      mean_BO_losses = np.mean(BO_losses[j], axis=1)
      mean_GD_losses = np.mean(GD_losses[j], axis=1)

      # 0.95 confidence intervals
      CI_BO = np.zeros((max_num_iterations[j], 2))
      CI_GD = np.zeros((max_num_iterations[j], 2))
      for i in range(max_num_iterations[j]):
        BO = BO_losses[j][i,:].copy()
        GD = GD_losses[j][i,:].copy()
        CI_BO[i,:] = st.t.interval(0.95, len(BO)-1, loc=np.mean(BO), scale=st.sem(BO))
        CI_GD[i,:] = st.t.interval(0.95, len(GD)-1, loc=np.mean(GD), scale=st.sem(GD))

      # Plot
      x = np.arange(1, max_num_iterations[j]+1)
      ax[j].plot(x, mean_BO_losses, label='UCB GP, $\kappa = 0.1$')
      ax[j].plot(x, mean_GD_losses, linestyle="-.", label="Random")
      ax[j].fill_between(x, y1=CI_BO[:,1], y2=CI_BO[:,0], color='blue', alpha=0.2, label="0.95 CI for UCB GP")
      ax[j].fill_between(x, y1=CI_GD[:,1], y2=CI_GD[:,0], color='orange', alpha=0.2, label="0.95 CI for Random")
      ax[j].set_title(titles[j])
      ax[j].set_ylabel(ylabels[j])

    # Plot description
    ax[0].legend()
    ax[len(BO_losses)-1].set_xlabel(xlabels[0])
    fig.suptitle(main_title, fontsize=16)
    plt.savefig(figname, bbox_inches='tight')

    plt.subplots_adjust(hspace=0.35)