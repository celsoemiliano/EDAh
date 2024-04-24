import numpy as np
from numba import jit
from joblib import Parallel, delayed
import numpy.random as npr
from random import randint, sample
from queue import SimpleQueue

class Utils:

    @staticmethod
    def initialize(path, jobs, resources):
        array = np.array(open(path).readlines(), dtype=float)

        ET = np.reshape(array, (jobs, resources))
        CT = ET.copy()

        maquinas = np.zeros(resources, dtype=float)

        return ET, CT, maquinas

    @staticmethod
    def get_min_in_matrix(matrix):
        return np.unravel_index(matrix.argmin(), matrix.shape)

    @staticmethod
    def get_max_in_matrix(matrix):
        return np.unravel_index(matrix.argmax(), matrix.shape)

    @staticmethod
    def get_min_in_array(array):
        return np.where(array == array.min())[0][0]

    @staticmethod
    def get_max_in_array(array):
        return np.where(array == array.max())[0][0]


def create_prob_matrix(jobs, machines):
    return np.zeros((jobs, machines), dtype=int)


@jit(nopython=True)
def get_first_pop_random(ni, nm, nj):
    return [list(np.random.randint(low=0, high=nm, size=nj)) for _ in range(ni)]


def order_pop(ET, individuals, seq_jobs):
    makespans = []
    makespans = Parallel(n_jobs=4)(delayed(get_fitness)(ET, i, seq_jobs) for i in individuals)
    makespans = np.array(makespans)
    inds = makespans.argsort()
    return individuals[inds]


def get_fitness(ET, individual, seq_jobs):
    qtd_maxima = max(seq_jobs)
    sub_jobs, start_indexes = get_sub_jobs(individual, seq_jobs)
    makespans = np.zeros(ET.shape[1])
    ult_maquinas = SimpleQueue()
    
    for i in range(0, qtd_maxima):
        maquinas = recuperar_maquinas(sub_jobs, i)

        for maquina in enumerate(maquinas):
            if(maquina[1] != -1):
                tempo = 0
                
                if i != 0:
                    ult_tempo = ult_maquinas.get()

                    tempo = max(
                        ult_tempo,
                        makespans[maquina[1]]
                    )

                    tempo += ET[start_indexes[maquina[0]] + i, maquina[1]]
                else:
                    tempo = makespans[maquina[1]] + ET[start_indexes[maquina[0]] + i, maquina[1]]
                
                makespans[maquina[1]] = tempo
                
                # Adiciona o tempo da última 
                if(i + 1 < len(sub_jobs[maquina[0]])):
                    ult_maquinas.put(tempo)

    return makespans[get_max_in_array(makespans)]


def get_max_in_array(array):
    """
    Esta função a posição do menor elemento do array passado por parametro
    """
    return np.where(array == array.max())[0][0]


@jit(nopython=True)
def fill_prob_matrix(prob_matrix, individuals, num_jobs):
    for i in individuals:
        for j in range(len(i)):
            prob_matrix[j][i[j]] += 1

    return prob_matrix


@jit(nopython=True)
def create_new_individual(prob_matrix, ET):
    new_individual = np.random.randint(0, 0, 0)
    count = 0

    for i in prob_matrix:
        count += 1
        choice = select_with_choice(i)
        new_individual = np.append(new_individual, choice)

    return new_individual


def create_new_pop(prob_matrix, qtde, ET):
    individuals = []
    for i in range(qtde):
        individuals.append(create_new_individual(prob_matrix, ET))
    return individuals


@jit(nopython=True)
def select_with_choice(population):
    weights = population / np.sum(population)
    return np.arange(len(population))[np.searchsorted(np.cumsum(weights), np.random.random(), side="right")]


@jit(nopython=True)
def mutate(individuals, mut_prob, num_machines, ET):
    for i in individuals:
        p = npr.randint(1, 100)
        if p <= mut_prob:
            pos = npr.randint(0, len(i) - 1)
            pos2 = npr.randint(0, len(i) - 1)
            i[pos], i[pos2] = i[pos2], i[pos]


def generate_new_with_crossover(pop, qtde):
    inds = []
    for i in range(qtde):
        pos1 = randint(0, len(pop[0]))
        pos2 = randint(pos1, len(pop[0]))

        ind1, ind2 = np.random.choice(len(pop), 2)

        new_ind1 = np.concatenate([pop[ind1][:pos1], pop[ind2][pos1:pos2], pop[ind1][pos2:]],
                                  axis=0)  # pop[ind1][:pos1] + pop[ind2][pos1:pos2] + pop[ind1][pos2:]
        new_ind2 = np.concatenate([pop[ind2][:pos1], pop[ind1][pos1:pos2], pop[ind2][pos2:]], axis=0)

        inds.append(new_ind1)
        inds.append(new_ind2)
    return inds

def get_sub_jobs(individual, seq_jobs):
    sub_jobs = []
    start_indexes = []
    start_index = 0

    for i in seq_jobs:
        sub_jobs.append(
            individual[start_index: start_index + i]
        )

        start_indexes.append(start_index)
        start_index+=i

    return sub_jobs, start_indexes

def recuperar_maquinas(sub_jobs, job_number):
    maquinas = []

    for sub_job in sub_jobs:
        if job_number < sub_job.shape[0]:
            maquinas.append(sub_job[job_number])
        else:
            maquinas.append(-1)

    return maquinas