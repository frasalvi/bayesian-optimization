import numpy as np
from matplotlib import gridspec, ticker
import matplotlib.pyplot as plt
import scipy.stats as st
from bayes_opt import BayesianOptimization, UtilityFunction
from helpers import *


def visualize_rosenbrock(xs=None):
    '''
    Visualize the Rosenbrock function, with a=1 and b=100.

    Args:
        xs (list): list points crossed by gradient descent.
    '''
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-1, 3, 1000)
    xx, yy = np.meshgrid(x, y)

    rosenbrock = Rosenbrock(1, 100)
    zz = rosenbrock(xx, yy)

    fig = plt.figure(1, (6, 5))
    plt.contour(xx, yy, zz, levels=np.logspace(-4, 3, 10), alpha=0.4, extend='both', locator=ticker.LogLocator(), colors='white')
    plt.contourf(xx, yy, zz, levels=np.logspace(-4, 3, 30), alpha=0.9, extend='both', locator=ticker.LogLocator())
    if(xs):
        plt.plot(*tuple(zip(*xs[::5])), '*-', color='red')
    plt.colorbar(ticks=np.logspace(-6, 3, 10))
    plt.axis('scaled')

    plt.xlabel('x')
    plt.ylabel('y')


def visualize_rastrigin(xs=None, final_points=None):
    '''
    Visualize the Rastrigin function, with A=10.

    Args:
        xs (list): points crossed by gradient descent, over iterations of a single run.
        final_points (list): points to which gradient descent converged, over multiple runs.
    '''
    x = np.linspace(-5.12, 5.12, 1000)
    y = np.linspace(-5.12, 5.12, 1000)
    xx, yy = np.meshgrid(x, y)

    rastrigin = Rastrigin(10)
    zz = rastrigin(xx, yy)

    fig = plt.figure(1, (6, 5))
    plt.contour(xx, yy, zz, alpha=0.5, colors='white')
    plt.contourf(xx, yy, zz, alpha=0.9, levels=30)
    plt.colorbar(ticks=np.arange(0, 81, 10), pad=0.01)
    if(xs):
        plt.plot(*tuple(zip(*xs[::10])), '*-', color='red')

    if(final_points):
        points, counts = np.unique(np.array(final_points).round(decimals=2), axis=0, return_counts=True)
        plt.scatter(*tuple(zip(*points)), color='red', s=counts/4)
        for index, (i, j) in enumerate(points):
            plt.annotate(str(counts[index]), xy=(i+0.01,j+0.01), color='white')

    plt.axis('scaled')

    plt.xlabel('x')
    plt.ylabel('y')


def visualize_rastrigin_3d():
    '''
    Visualize the Rastrigin function, with A=10, in 3D.

    Args:
        xs (list): points crossed by gradient descent, over iterations of a single run.
        final_points (list): points to which gradient descent converged, over multiple runs.
    '''
    x = np.linspace(-5.12, 5.12, 1000)
    y = np.linspace(-5.12, 5.12, 1000)
    xx, yy = np.meshgrid(x, y)

    rastrigin = Rastrigin(10)
    zz = rastrigin(xx, yy)

    fig = plt.figure(1, (12, 8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='jet')




def plot_w_CI(BO_losses, GD_losses, max_num_iterations, title, xlabel, ylabel, figname):
    '''
       Helper for plotting losses from bayesian optimization compared to random initialization
       
       Args:
           * BO_losses: numpy array with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * GD_losses: numpy array with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * title: str, title of plot
           * xlabel: str, x-label of plot
           * ylabel: str, y-label of plot
           * figname: str, path to save image
    '''
    
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
    '''
       Helper for plotting several settings of bayesian optimization compared to random initialization
       
       Args:
           * BO_losses: list of numpy arrays with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * GD_losses: list of numpy arrays with shape (num_trials, num_iterations), raw gradient descent losses of experiment
           * main_title: Title for entire figure
           * titles: list of str, title of each subplot
           * xlabel: list of str, x-label of each subplot (note only bottom is displayed)
           * ylabel: list of str, y-label of each subplot
           * figname: str, path to save image
    '''
    
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



def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    bound = [-5.12, 5.12]
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((bound[0], bound[1]))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((bound[0], bound[1]))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def visualize_bayesian(n_steps=10, kappa=5, seed=None):
    ra1d = Rastrigin1d(10)
    bound = [-5.12, 5.12]

    xx = np.linspace(bound[0], bound[1], 1000).reshape(-1, 1)
    yy = ra1d(xx)
    optimizer = BayesianOptimization(f=ra1d, pbounds={"x": bound}, random_state=seed, verbose=2)
    optimizer.maximize(init_points=1, n_iter=0)

    for iter in range(n_steps):
        optimizer.maximize(init_points=0, n_iter=1, kappa=kappa)
        plot_gp(optimizer, xx, yy)
        plt.show()