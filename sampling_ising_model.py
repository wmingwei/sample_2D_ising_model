import numpy as np
from numpy.random import randint, randn, rand
from joblib import Parallel, delayed
import multiprocessing


Size = 40
J = 1

def energy(field, J):
    energy = 0.0
    size = len(field)
    for x in range(size):
        for y in range(size):
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                energy += - J * field[(x + dx)%size][(y + dy)%size] * field[x][y]
    
    energy = energy / 4.0
    return energy


def specific_heat(samples, temp, J, parallel = False):
    
    samples_energy = []
    if not parallel:
        for sample in samples:
            samples_energy.append(energy(sample))
    else:
        num_cores = multiprocessing.cpu_count()
        samples_energy = Parallel(n_jobs=num_cores)(delayed(energy)(field = samples[i]) for i in range(len(samples)))
        
    samples_energy = np.array(samples_energy)
    
    return 1/temp**2*((samples_energy**2).mean()-(samples_energy.mean())**2)

def spin_direction(field, x, y, Temp, J):
    energy = 0.0
    size = len(field)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        energy += - J * field[(x + dx)%size][(y + dy)%size]

    if Temp == 0:
        p = (np.sign(energy) + 1) * 0.5
    else:
        p = 1/(1+np.exp(2*(1/Temp)*energy))
    if rand() <= p:
        spin = 1
    else:
        spin = -1
    return spin



def run_gibbs_sampling(field, Temp, seed, iternum, J):
    np.random.seed(seed)
    for _ in range(iternum):
        lattice = [(x,y) for x in range(Size) for y in range(Size)]
        np.random.shuffle(lattice)
        for x, y in lattice:
            field[x][y] = spin_direction(field, x, y, Temp, J = J)
            
            
def sampling(temps, seed, J, iternum = 100):
    np.random.seed(seed)
    samples = []
    field = randint(2,size=(Size,Size))*2-1
    for temp in temps:
        run_gibbs_sampling(field = field, Temp = temp, seed = seed, iternum = iternum, J = J)
        samples.append(list(field.copy().reshape(Size*Size)) + [temp])
    return samples


def get_samples(temps, size = Size, num_samples = 1000, iternum = 100):
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(delayed(sampling)(temps = temps, seed = None, J = J, iternum = iternum) for i in range(num_samples))
    results = np.array(results)
    results = results.reshape(num_samples*len(temps), size*size+1)
    np.savetxt('samples.csv', results, delimiter=",",fmt='%f')