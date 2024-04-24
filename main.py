import argparse
import numpy as np
from utils import Utils, get_first_pop_random, order_pop, get_fitness, create_prob_matrix, fill_prob_matrix, \
    create_new_pop, mutate
import time
import sys
import glob

#individuo
#array de inteiros de [0,qtde_maquinas]: tamanho qtde_tarefas

np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()

parser.add_argument('--jobs', help='number of jobs')
parser.add_argument('--machines', help='number of jobs to schedule')
parser.add_argument('--path', help='path for the jobs instance')
parser.add_argument('--numInd', help='number of individuals')
parser.add_argument('--numGen', help='number of generations')
parser.add_argument('--toMatrix', help='percentual da matriz')
parser.add_argument('--elitism', help='percentual de individuos que passam de Geração')
parser.add_argument('--mutation', help='percentual de individuos que sofrem mutacao')
parser.add_argument('--jobsSequence', help='particionamento de execução de jobs')

args = parser.parse_args()

jobs = [int(i) for i in args.jobs.split(',')]
machines = [int(i) for i in args.machines.split(',')]
numInd = [int(i) for i in args.numInd.split(',')]
numGen = [int(i) for i in args.numGen.split(',')]
toMatrix = [float(i) for i in args.toMatrix.split(',')]
path = args.path
path = glob.glob(f'{path}/*')
elitism = [int(i) for i in args.elitism.split(',')]
mutacao = [int(i) for i in args.mutation.split(',')]
seqJobs = [int(i) for i in args.jobsSequence.split(',')]

for j in jobs:
    for m in machines:
        for ni in numInd:
            for ng in numGen:
                for tm in toMatrix:
                    for p in path:
                        for e in elitism:
                            for mu in mutacao:
                                # ------------ inicialização ------------
                                array = np.array(open(p).readlines(), dtype=float)
                                ET, CT, maquinas = Utils.initialize(p, j, m)
                                # ------------ timer ------------
                                start = time.time()
                                # ------------ 1 pop ------------
                                pop = get_first_pop_random(ni, m, j)
                                pop = np.array(pop)
                                pop = order_pop(ET, pop, seqJobs)
                                #best_makespan = get_fitness(ET, pop[0])
                                pop = pop[:ni]#do 0 ate essa posicao ni
                                # ------------  ------------
                                for i in range(ng):#0 -> numero de geracoes
                                    pb = create_prob_matrix(j, m)

                                    pop = order_pop(ET, pop, seqJobs)
                                    pb_to_matrix = pop.copy()[:int(len(pop) * tm)]
                                    pb = fill_prob_matrix(pb, pb_to_matrix, j)

                                    pop = pop[:e]

                                    new_pop = create_new_pop(pb, ni - len(pop), ET)

                                    pop = np.append([pop], [new_pop], axis=1)[0]
                                    mutate(pop, mu, m, ET)
                                    pop = order_pop(ET, pop, seqJobs)


                                    #if i == 48:
                                        #print(pop)
                                print("melhor ind: ", get_fitness(ET, pop[0], seqJobs))
                                    #print("pior ind: ", get_fitness(ET, pop[-1], seqJobs))
                                #print('------------------------')
                                end = time.time()
                                # ------------ end ------------

                                print("Elapsed time = %s" % (end - start))
                                print('------------------------')
