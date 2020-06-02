import click
import numpy as np
import PSO as pso
import ECGA as ecga
import sGA as sga
import fitness_function_dictionary as f_dict

functions_dict = { 'himmelblau' : f_dict.himmelblau_dict,
                   'onemax' : f_dict.onemax_dict,
                   'cross_in_tray' : f_dict.cross_in_tray_dict,
                   'beale' : f_dict.beale_dict,
                   'booth' : f_dict.booth_dict,
                   'rastrigin' : f_dict.rastrigin_dict,
                    }
methods_dict = { 'pso' : pso, 'ecga' : ecga, 'sga' : sga }

@click.command()
@click.option('--maximize', '-max', default=False, type=bool, help='Define whether to maximize or minimize the output')
@click.option('--function', '-f', required=True, type=str, help='Choose which function to evaluate')
@click.option('--method', '-m', required=True, type=str, help='Choose optimization method')
@click.option('--seed', '-s', default=1, type=int, help='Random seed for the random number generator (default value : 1)')
@click.option('--gen', '-g', default=-1, type=int, help='Max generations to evaluate population (default value : -1)')
@click.option('--pop_shape', '-ps', default=(100, 2), type=(int, int), help='Define initial population shape (N, d) (default value : (100, 2))')
@click.option('--tournament_size', '-ts', default=4, type=int, help='Define tournament size for some method')
@click.option('--mode', type=str, help='Choose which mode in optimization method to use')
@click.option('--print_scr', '-prnscr', default=True, type=bool, help='Print result to command line')
@click.option('--plot', '-plt', default=0, type=click.IntRange(0, 3), help='0 (no plot), 1 (2d plot), 2 (3d plot)')
def run(method, function, maximize, seed, gen, pop_shape, 
        tournament_size, mode, print_scr, plot):
    f_func = functions_dict[function]
    if f_func['d'] == 0:
        f_func['d'] = pop_shape[1]
        if f_func['name'] == 'One Max' or 'Trap' in f_func['name']:
            f_func['global minimum'] = np.zeros((1, f_func['d']))
            f_func['global maximum'] = np.ones((1, f_func['d']))
    params = methods_dict[method].get_parameters(N=pop_shape[0], s=seed, g=gen, 
                                                 mode=mode, f=f_func, 
                                                 maximize=maximize, t_size=tournament_size)

    opt_method = methods_dict[method]
    result = opt_method.optimize(params, plot, print_scr)
    print(result)

if __name__ == '__main__':
    run()



