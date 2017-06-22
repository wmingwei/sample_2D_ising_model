import numpy as np
from numpy.random import randint, randn, rand
from joblib import Parallel, delayed
import multiprocessing


Size = 40
J = 1
H = 0.0
Temp = 0

def energy(field):
    energy = H
    size = len(field)
    for x in range(size):
        for y in range(size):
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                energy += - J * field[(x + dx)%size][(y + dy)%size] * field[x][y]
    
    energy = energy / 4.0
    return energy


def specific_heat(samples, temp, parallel = False):
    
    samples_energy = []
    if not parallel:
        for sample in samples:
            samples_energy.append(energy(sample))
    else:
        num_cores = multiprocessing.cpu_count()
        samples_energy = Parallel(n_jobs=num_cores)(delayed(energy)(field = samples[i]) for i in range(len(samples)))
        
    samples_energy = np.array(samples_energy)
    
    return 1/temp**2*((samples_energy**2).mean()-(samples_energy.mean())**2)

def spin_direction(field, x, y, Temp):
    energy = H
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



def run_gibbs_sampling(field, Temp, seed, iternum):
    np.random.seed(seed)
    for _ in range(iternum):
        lattice = [(x,y) for x in range(Size) for y in range(Size)]
        np.random.shuffle(lattice)
        for x, y in lattice:
            field[x][y] = spin_direction(field, x, y, Temp)
            
            
def sampling(temps, seed, iternum = 20):
    np.random.seed(seed)
    samples = []
    field = randint(2,size=(Size,Size))*2-1
    for temp in temps:
        run_gibbs_sampling(field = field, Temp = temp, seed = seed, iternum = iternum)
        samples.append(list(field.copy().reshape(Size*Size)) + [temp])
    return samples

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(sampling)(temps = temps, seed = None) for i in range(100))
results = np.array(results)
results = results.reshape(100*27, 1600+1)
np.savetxt('samples.csv', results, delimiter=",",fmt='%f')
for j in range(9):
    results = Parallel(n_jobs=num_cores)(delayed(sampling)(temps = temps, seed = None) for i in range(100))
    results = np.array(results)
    results = results.reshape(100*27, 1600+1)
    samples = list(np.genfromtxt('samples.csv', delimiter=","))
    samples += list(results)
    samples = np.array(samples)
    np.savetxt('samples.csv', samples, delimiter=",",fmt='%f')
    print(j)