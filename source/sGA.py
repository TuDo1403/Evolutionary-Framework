import numpy as np
from enum import Enum
import fitness_function as ff

from GA import *
from plot import *


def variate(pop, crossover_mode):
    (num_inds, num_params) = pop.shape
    indices = np.arange(num_inds)

    offsprings = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        index1 = indices[i]
        index2 = indices[i+1]
        offspring1 = pop[index1].copy()
        offspring2 = pop[index2].copy()

        if crossover_mode == 'onepoint':
            point = np.random.randint(low=0, high=num_params-1)
            offspring1[:point], offspring2[:point] = offspring2[:point], offspring1[:point].copy()
        else:
            for j in range(num_params):
                if np.random.randint(low=0, high=2) == 1:
                    offspring1[j], offspring2[j] = offspring2[j], offspring1[j]

        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return np.reshape(offsprings, (num_inds, num_params))

def tournament_selection(f_pool, tournament_size, selection_size, maximize=False):
    num_inds = len(f_pool)
    indices = np.arange(num_inds)
    selected_indices = []
    comparer = max if maximize else min

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)

        for i in range(0, num_inds, tournament_size):
            idx_tournament = indices[i : i+tournament_size]
            best_idx = list(filter(lambda idx : f_pool[idx] == comparer(f_pool[idx_tournament]), idx_tournament))
            selected_indices.append(np.random.choice(best_idx))

    return selected_indices


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def optimize(params, plot=False, print_scr=True):
    """

    """

    # Initialize required parameters from dictionary
    num_inds = params['N']
    tournament_size = params['ts']
    max_gen = params['g']
    seed = params['s']
    maximize = params['maximize']
    crossover_mode = params['cm']

    f_func = params['f']    # Dictionary of fitness function data
    real_valued = f_func['real valued']
    (lower_bound, upper_bound) = f_func['D']
    num_params = f_func['d']

    # Plot search space
    plottable = plot and num_params == 2
    if plottable:
        data = get_plot_data(f_func)
        fig, ax = plt.subplots()
        if plottable and plot == 2:
            ax = Axes3D(fig)

    # Initialize
    comparer = np.argmax if maximize else np.argmin
    np.random.seed(seed)
    epsilon = 10**-5
    pop = initialize(num_inds, num_params, 
                     domain=[lower_bound, upper_bound], 
                     real_valued=real_valued)
    f_pop = evaluate(pop, f_func['function'])
    selection_size = len(pop)
    gen = 0
    num_f_func_calls = len(f_pop)
    #
    while not pop_converge(pop):
        gen += 1
        if max_gen_reached(gen, max_gen):
            break

        # Variate
        offs = variate(pop, crossover_mode)

        # Evaluate
        f_offs = evaluate(offs, f_func['function'])    
        num_f_func_calls += len(f_offs)

        # Selection
        pool = np.vstack((pop, offs))
        f_pool = np.hstack((f_pop, f_offs))

        pool_indices = tournament_selection(f_pool, tournament_size, 
                                            selection_size, maximize)
        pop = pool[pool_indices]
        f_pop = f_pool[pool_indices]
        #

        # Visualize / log result
        if print_scr and gen % 100 == 0:
            print('## Gen {}: {} (Fitness: {})'.format(gen, pop[comparer(f_pop)].reshape(1, -1), 
                                                       f_pop[comparer(f_pop)]))
        if plottable:
            ax.clear()
            if plot == 1:
                contour_plot(ax, data, f_func, hold=True)
                ax_lim = (xlim, ylim) = ax.get_xlim(), ax.get_ylim()
                scatter_plot(ax_lim, ax, pop, hold=True)
            else:
                contour_3D(ax, data, f_func, hold=True)
                ax_lim = (xlim, ylim, zlim) = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                scatter_3D(ax_lim, ax, pop, f_pop, hold=True)
            plt.pause(epsilon)
        #

    if plottable: 
        plt.show() 

    solution =  pop[comparer(f_pop)].reshape(1, -1).flatten()
    opt_sol_found = None

    optimize_goal = 'global maximum' if maximize else 'global minimum'
    if type(f_func[optimize_goal]) != type(None):
        epsilon = 10**-5
        diffs = np.abs(f_func[optimize_goal] - solution).sum(axis=1)
        opt_sol_found = len(np.where(diffs <= num_params*epsilon)[0]) != 0

    result = { 'solution' : solution, 
               'evaluate function calls' : num_f_func_calls, 
               'global optima found' : opt_sol_found }
    return result


def get_parameters(**params):
    mode = 'ux' if params['mode'] == "" or params['mode'] not in ['ux', 'onepoint'] else params['mode']
    default_params = { 'N' : params['N'],
                       's' : params['s'],
                       'g' : params['g'],
                       'ts' : params['t_size'],
                       'maximize' : params['maximize'],
                       'f' : params['f'],
                       'cm' : mode }
    return default_params