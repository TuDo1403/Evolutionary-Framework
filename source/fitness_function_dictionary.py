import fitness_function as ff
import numpy as np


himmelblau_dict = { 'name' : 'Himmelblau',
                    'function' : ff.himmelblau,
                    'd' : 2,
                    'global minimum' : np.array([[3, 2],
                                                 [-2.805118, 3.131312],
                                                 [-3.779310, -3.283186],
                                                 [3.584428, -1.848126]]),
                    'global maximum' : None,
                    'D' : (-5, 5),
                    'real valued' : True,
                    'multi dims' : False }

onemax_dict = { 'name' : 'One Max',
                'function' : ff.onemax,
                'd' : 0,
                'global minimum' : None,
                'global maximum' : None,
                'D' : (0, 2),
                'real valued' : False,
                'multi dims' : True }

cross_in_tray_dict = { 'name' : 'Himmelblau',
                       'function' : ff.cross_in_tray,
                       'd' : 2,
                       'global minimum' : np.array([[1.34941, -1.34941],
                                                    [1.34941, 1.34941],
                                                    [-1.34941, -1.34941],
                                                    [-1.34941, 1.34941]]),
                       'global maximum' : None,
                       'D' : (-10, 10),
                       'real valued' : True,
                       'multi dims' : False }

booth_dict = { 'name' : 'Booth',
               'function': ff.booth,
               'd' : 2,
               'global minimum' : np.array([[1, 3]]),
               'global maximum' : None,
               'D' : (-10, 10),
               'real valued' : True,
               'multi dims' : False }

rastrigin_dict = { 'name' : 'Rastrigin',
                   'function' : ff.rastrigin,
                   'd' : 0,
                   'global minimum' : None,
                   'global maximum' : None,
                   'D' : (-5.12, 5.12),
                   'real valued' : True,
                   'multi dims' : True
                   }

beale_dict = { 'name' : 'Beale',
               'function' : ff.beale,
               'd' : 2,
               'global minimum' : np.array([[3, 0.5]]),
               'global maximum' : None,
               'D' : (-4.5, 4.5),
               'real valued' : True,
               'multi dims' : False}

happy_cat_dict = { }

