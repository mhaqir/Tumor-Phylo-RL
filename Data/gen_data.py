
import configparser
import numpy as np
import pandas as pd
from random import uniform
from copy import deepcopy
import itertools
import os

config = configparser.ConfigParser()
config.read('config.ini')


nMuts = eval(config['General']['nMuts'])
nCells = eval(config['General']['nCells'])
nMat = eval(config['General']['nMat'])

p_dir = config['Directory']['p_dir']
n_dir = config['Directory']['n_dir']
p_na_dir = config['Directory']['p_na_dir']
ms_dir = config['Directory']['ms_dir']


def count3gametes(matrix,nCells, nMuts):
    columnPairs = list(itertools.permutations(range(nMuts), 2))
    nColumnPairs = len(columnPairs)
    columnReplicationList = np.array(columnPairs).reshape(-1)
    replicatedColumns = matrix[:, columnReplicationList].transpose()
    x = replicatedColumns.reshape((nColumnPairs, 2, nCells), order="A")
    col10 = np.count_nonzero( x[:,0,:]<x[:,1,:]     , axis = 1)
    col01 = np.count_nonzero( x[:,0,:]>x[:,1,:]     , axis = 1)
    col11 = np.count_nonzero( (x[:,0,:]+x[:,1,:]==2), axis = 1)
    eachColPair = col10 * col01 * col11
    return np.sum(eachColPair)


def run_ms(nMat, nCells, nMuts, ms_dir):
    m = []
    matrices = np.zeros((nMat, nCells, nMuts), dtype = np.int8)
    for i in range(1, nMat + 1):
        cmd = "{ms_dir} {nCells} 1 -s {nMuts} | tail -n {nCells} > {p_dir}/m.txt".format(ms_dir = ms_dir, nCells = nCells, nMuts = nMuts, p_dir = p_dir)
        os.system(cmd)
        with open(p_dir + '/m.txt', 'r') as f:
            l = [line for line in f]
        l1 = [s.strip('\n') for s in l]
        l2 = np.array([[int(s) for s in q] for q in l1])  # Original matrix
        matrices[i-1,:,:] = l2
        m.append(tuple(l2.flatten()))
    os.remove(p_dir + '/m.txt')
    m1 = list(set(m))
    matrices_u = np.zeros((len(m1), nCells, nMuts), dtype = np.int8)
    for j in range(len(m1)):
        matrices_u[j,:,:] = np.asarray(m1[j]).reshape((nCells, nMuts))
    return matrices_u  # returns all of generated unique matrices
        

def add_noise(matrices_u, nCells, nMuts, fp_fn_fixed = True):
    fp_r = []
    fn_r = []
    if fp_fn_fixed == True:
        alpha = eval(config['Data1']['alpha'])
        betta = eval(config['Data1']['betta'])
    else:
        alpha = eval(config['Data2']['alpha'])
        betta = eval(config['Data2']['betta'])
        
    k = 0   
    for i in range(np.shape(matrices_u)[0]):
        v = 0
        matrix_n = deepcopy(matrices_u[i,:,:].reshape(1, -1))
        while ((count3gametes(matrix_n.reshape(nCells, nMuts), nCells, nMuts) == 0) and (v < nCells*nMuts)):
            if fp_fn_fixed == True:
                fp = alpha
                fn = betta
            else:
                fp = uniform(alpha[0], alpha[1])
                fn = uniform(betta[0], betta[1])

            matrix_n = deepcopy(matrices_u[i,:,:].reshape(1, -1))
            Zs = np.where(matrix_n  == 0)[1]
            s_fp = np.random.choice([True, False], (1, len(Zs)), p = [fp, 1 - fp])  # must be flipped from 0 to 1
            Os = np.where(matrix_n  == 1)[1] 
            s_fn = np.random.choice([True, False], (1, len(Os)), p = [fn, 1 - fn]) # must be flipped from 1 to 0
            matrix_n[0, Zs[np.squeeze(s_fp)]] = 1
            matrix_n[0, Os[np.squeeze(s_fn)]] = 0
            v += 1
            
        if count3gametes(matrix_n.reshape(nCells, nMuts), nCells, nMuts) != 0:
            k += 1
            df1 = pd.DataFrame(matrices_u[i,:,:] , index = ['cell' + str(k1) for k1 in range(np.shape(matrices_u[i,:,:])[0])], \
                          columns = ['mut' + str(h1) for h1 in range(np.shape(matrices_u[i,:,:])[1])])
            df1.index.rename('cellID/mutID', inplace=True)
            df1.to_csv(p_dir + '/mp{}.txt'.format(k), sep='\t')    # Write perfect matrix in txt file

            df2 = pd.DataFrame(matrix_n.reshape(nCells, nMuts) , index = ['cell' + str(k1) for k1 in range(np.shape(matrix_n.reshape(nCells, nMuts))[0])], \
                      columns = ['mut' + str(h1) for h1 in range(np.shape(matrix_n.reshape(nCells, nMuts))[1])])
            df2.index.rename('cellID/mutID', inplace=True)
            df2.to_csv(n_dir + '/mn{}.txt'.format(k), sep='\t') # Write noisy matrix in txt file

            fp_r.append(fp)
            fn_r.append(fn)
        
    with open(n_dir + '/fp_r.txt', 'w') as f:
        for item in fp_r:
            f.write("%s\n" % item)
    with open(n_dir + '/fn_r.txt', 'w') as f:
        for item in fn_r:
            f.write("%s\n" % item)
    
    
def add_noise_NA(matrices_u, nCells, nMuts, fp_fn_fixed = True):
    fp_r = []
    fn_r = []
    if fp_fn_fixed == True:
        alpha = eval(config['Data1']['alpha'])
        betta = eval(config['Data1']['betta'])
        gamma = eval(config['Data1']['gamma'])
    else:
        alpha = eval(config['Data2']['alpha'])
        betta = eval(config['Data2']['betta'])
        gamma = eval(config['Data2']['gamma'])
        
    for i in range(np.shape(matrices_u)[0]):
        if fp_fn_fixed == True:
            fp = alpha
            fn = betta
            NA_r = gamma
        else:
            fp = uniform(alpha[0], alpha[1])
            fn = uniform(betta[0], betta[1])
            NA_r = gamma
            
        matrix_na = deepcopy(matrices_u[i,:,:].reshape(1, -1))
        s_NA = np.random.choice([True, False], (1, np.shape(matrix_na)[1]), p = [NA_r, 1 - NA_r])  # must be flipped from 0 or 1 to 2
        matrix_na[0, np.squeeze(s_NA)] = 2
        matrix_n = deepcopy(matrix_na)
        Zs = np.where(matrix_n  == 0)[1]
        s_fp = np.random.choice([True, False], (1, len(Zs)), p = [fp, 1 - fp])  # must be flipped from 0 to 1
        Os = np.where(matrix_n  == 1)[1] 
        s_fn = np.random.choice([True, False], (1, len(Os)), p = [fn, 1 - fn]) # must be flipped from 1 to 0
        matrix_n[0, Zs[np.squeeze(s_fp)]] = 1
        matrix_n[0, Os[np.squeeze(s_fn)]] = 0
        
        df1 = pd.DataFrame(matrices_u[i,:,:] , index = ['cell' + str(k1) for k1 in range(np.shape(matrices_u[i,:,:])[0])], \
                      columns = ['mut' + str(h1) for h1 in range(np.shape(matrices_u[i,:,:])[1])])
        df1.index.rename('cellID/mutID', inplace=True)
        df1.to_csv(p_dir + '/mp{}.txt'.format(i+1), sep='\t')    # Write perfect matrix in txt file

        df2 = pd.DataFrame(matrix_n.reshape(nCells, nMuts) , index = ['cell' + str(k1) for k1 in range(np.shape(matrix_n.reshape(nCells, nMuts))[0])], \
                  columns = ['mut' + str(h1) for h1 in range(np.shape(matrix_n.reshape(nCells, nMuts))[1])])
        df2.index.rename('cellID/mutID', inplace=True)
        df2.to_csv(n_dir + '/mn{}.txt'.format(i+1), sep='\t') # Write noisy matrix in txt file

        df3 = pd.DataFrame(matrix_na.reshape(nCells, nMuts) , index = ['cell' + str(k1) for k1 in range(np.shape(matrix_na.reshape(nCells, nMuts))[0])], \
                  columns = ['mut' + str(h1) for h1 in range(np.shape(matrix_na.reshape(nCells, nMuts))[1])])
        df3.index.rename('cellID/mutID', inplace=True)
        df3.to_csv(p_na_dir + '/mpna{}.txt'.format(i+1), sep='\t') # Write NA matrix in txt file

        fp_r.append(fp)
        fn_r.append(fn)
        
    with open(n_dir + '/fp_r.txt', 'w') as f:
        for item in fp_r:
            f.write("%s\n" % item)
    with open(n_dir + '/fn_r.txt', 'w') as f:
        for item in fn_r:
            f.write("%s\n" % item)

def main():
    matrices_u = run_ms(nMat, nCells, nMuts, ms_dir)
    add_noise(matrices_u, nCells, nMuts, fp_fn_fixed = True)
    
if __name__ == '__main__':
    main()
    
# for i in range(5):
#     l2 = pd.read_csv(p_dir + '/mp{}.txt'.format(i+1), sep="\t")
#     l2.set_index('cellID/mutID', inplace=True)
#     l3 = pd.read_csv(n_dir + '/mn{}.txt'.format(i+1), sep="\t")
#     l3.set_index('cellID/mutID', inplace=True)
#     l4 = pd.read_csv(p_na_dir + '/mpna{}.txt'.format(i+1), sep="\t")
#     l4.set_index('cellID/mutID', inplace=True)
#     print(l2.values)
#     print("------------------------------------------")
#     print(l3.values)
#     print("------------------------------------------")
#     print(l4.values)
#     print(np.logical_xor(l3.values , l4.values))
#     print("------------------------------------------")
#     print(np.sum(np.logical_xor(l3.values , l4.values)))