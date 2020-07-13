import os
import shutil
import time
import numpy as np
import random

os.system('search -opt pso -f rastrigin -g 500 -ps 200 2 -max False > ./test.txt &')

while True:
    time.sleep(1)

    number_of_processes = int(os.popen('pgrep -c search').read())
    if number_of_processes == 0:
        break

    time.sleep(1)

print('Second process')
os.system('search -opt pso -f rastrigin -g 200 -ps 200 5 -max True > ./test1.txt &')
