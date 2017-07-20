import numpy as np
from numpy.random import randint, randn, rand
from joblib import Parallel, delayed
import multiprocessing


J = 1
H_ext = 0.1

def energy(field, J = J, H_ext = H_ext):
    energy = 0.0
    size = len(field)
    for x in range(size):
        for y in range(size):
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                energy += - J * field[(x + dx)%size][(y + dy)%size] * field[x][y] - H_ext * field[x][y] /2
    
    energy = energy / 2.0
    return energy


def specific_heat(samples, temp, J = J, H_ext = H_ext, parallel = False):
    samples_energy = []
    
    if not parallel:
        for sample in samples:
            field = sample
            samples_energy.append(energy(field , J = J, H_ext = H_ext))
    else:
        num_cores = multiprocessing.cpu_count()
        samples_energy = Parallel(n_jobs=num_cores)(delayed(energy)(field = samples[i], J = J, H_ext = H_ext) for i in range(len(samples)))
        
    samples_energy = np.array(samples_energy)
    
    return 1/temp**2*((samples_energy**2).mean()-(samples_energy.mean())**2)

def spin_direction(field, x, y, Temp, J = J, H_ext = H_ext):
    d_energy = 0.0
    size = len(field)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        d_energy += 2 * J * field[(x + dx)%size][(y + dy)%size] * field[x][y] + 2 * H_ext * field[x][y]
        
    p = np.exp(- d_energy / Temp)
    
    if rand() <= p:
        spin = -field[x][y]
    else:
        spin = field[x][y]
    return spin



def run_gibbs_sampling(field, Temp, seed, iternum, J = J, H_ext = H_ext):
    np.random.seed(seed)
    for _ in range(iternum):
        lattice = [(x,y) for x in range(Size) for y in range(Size)]
        np.random.shuffle(lattice)
        for x, y in lattice:
            field[x][y] = spin_direction(field, x, y, Temp, J = J, H_ext = H_ext)
            
            
def sampling(temps, seed, J = J, H_ext = H_ext, iternum = 100):
    np.random.seed(seed)
    samples = []
    field = randint(2,size=(Size,Size))*2-1
    for temp in temps:
        run_gibbs_sampling(field = field, Temp = temp, seed = seed, iternum = iternum, J = J, H_ext = H_ext)
        samples.append(list(field.copy().reshape(Size*Size)) + [temp])
    return samples


def get_samples(temps, size = 16, num_samples = 1000, iternum = 100):
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(delayed(sampling)(temps = temps, seed = None, J = J, H_ext = H_ext, iternum = iternum)  for i in range(num_samples))
    results = np.array(results)
    results = results.reshape(num_samples*len(temps), size*size+1)
    np.savetxt('samples.csv', results, delimiter=",",fmt='%f')