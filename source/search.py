import click
import numpy as np

import optimizers.PSO as pso
import optimizers.ECGA as ecga
import optimizers.sGA as sga
import test_functions.fitness_function_dictionary as f_dict

func_dict = { 'himmelblau' : f_dict.himmelblau_dict,
            'onemax' : f_dict.onemax_dict,
            'cross_in_tray' : f_dict.cross_in_tray_dict,
            'beale' : f_dict.beale_dict,
            'booth' : f_dict.booth_dict,
            'rastrigin' : f_dict.rastrigin_dict,
                    }
opt_dict = { 'pso' : pso, 'ecga' : ecga, 'sga' : sga }

@click.command()
@click.option('--optimizer', '-opt', required=True, type=click.Choice(opt_dict.keys(), case_sensitive=False), 
            help='Choose optimization method')
@click.option('--func', '-f', required=True, type=click.Choice(func_dict.keys(), case_sensitive=False), 
            help='Choose which function to evaluate')
@click.option('-max', 'optimize', flag_value='max', default=False, 
            help='Define whether to maximize or minimize the output')
@click.option('-min', 'optimize', flag_value='min')
@click.option('--seed', '-s', default=1, type=int, 
            help='Random seed for the random number generator (default value : 1)')
@click.option('--gen', '-g', default=-1, type=int, 
            help='Max generations to evaluate population (default value : -1)')
@click.option('--pshape', '-ps', default=(100, 2), type=(int, int), 
            help='Define initial population shape (N, d) (default value : (100, 2))')
@click.option('--tsize', '-ts', default=4, type=int, 
            help='Define tournament size for some method')
@click.option('--mode', type=str, 
            help='Choose which mode in optimization method to use')
@click.option('--printscr', '-prnscr', default=True, type=bool, 
            help='Print result to command line')
@click.option('--plot', '-plt', default=0, type=click.IntRange(0, 3), 
            help='0 (no plot), 1 (2d plot), 2 (3d plot)')
def cli(optimizer, func, optimize, seed, gen, pshape, 
        tsize, mode, printscr, plot):
    f_dict = func_dict[func]
    if f_dict['d'] == 0:
        f_dict['d'] = pshape[1]
        if f_dict['name'] == 'One Max' or 'Trap' in f_dict['name']:
            f_dict['global minimum'] = np.zeros((1, f_dict['d']))
            f_dict['global maximum'] = np.ones((1, f_dict['d']))
    params = opt_dict[optimizer].get_parameters(N=pshape[0], s=seed, g=gen, 
                                                 mode=mode, f=f_dict, 
                                                 maximize=optimize, t_size=tsize)

    opt_method = opt_dict[optimizer]
    result = opt_method.optimize(params, plot, printscr)
    print(result)