from imports import *


def run_ms(nMat, nCells, nMuts, ms_dir):
    m = []
    matrices = np.zeros((nMat, nCells, nMuts), dtype = np.int8)
    os.mkdir(p_dir + '/tmp')
    for i in range(1, nMat + 1):
        cmd = "{ms_dir} {nCells} 1 -s {nMuts} | tail -n {nCells} > {tmp_dir}/m{i}.txt".format(ms_dir = ms_dir, nCells = nCells, nMuts = nMuts, tmp_dir = p_dir + '/tmp', i = i)
        os.system(cmd)
    for j in range(1, nMat + 1):
        f = open(p_dir + "/tmp/m{j}.txt".format(j = j), 'r')
        l = [line for line in f]
        l1 = [s.strip('\n') for s in l]
        l2 = np.array([[int(s) for s in q] for q in l1])  # Original matrix
        matrices[j-1,:,:] = l2
        m.append(tuple(l2.flatten()))
        f.close()
    shutil.rmtree(p_dir + '/tmp')
    m1 = list(set(m))
    matrices_u = np.zeros((len(m1), nCells, nMuts), dtype = np.int8)
    for k in range(len(m1)):
        matrices_u[k,:,:] = np.asarray(m1[k]).reshape((nCells, nMuts))
    return matrices_u  # returns all of generated unique matrices


def add_noise(matrix, n):
    if isViolated(matrix) == True:
        v = True
        return matrix
    else:
        v = False
    r = 0
    Os = np.where(matrix  == 1)[1]
    while v == False and r < len(Os):
        matrix_n = deepcopy(matrix.reshape(1, -1))
        Os = np.where(matrix_n  == 1)[1]
        s_n = sample(Os, n)
        matrix_n[0, s_n] = 0
        matrix_n = matrix_n.reshape(np.shape(matrix)[0], np.shape(matrix)[1])
        v = isViolated(matrix_n)
        r += 1
    return matrix_n


def count3gametes(matrix):
    nCells = np.shape(matrix)[0]
    nMuts = np.shape(matrix)[1]
    columnPairs = list(itertools.permutations(range(nMuts), 2))
    nColumnPairs = len(columnPairs)
    columnReplicationList = np.array(columnPairs).reshape(-1)
    replicatedColumns = matrix[:, columnReplicationList].transpose()
    x = replicatedColumns.reshape((nColumnPairs, 2, nCells), order="A")
    col10 = np.count_nonzero(x[:,0,:]<x[:,1,:]     , axis = 1)
    col01 = np.count_nonzero(x[:,0,:]>x[:,1,:]     , axis = 1)
    col11 = np.count_nonzero((x[:,0,:]+x[:,1,:]==2), axis = 1)
    eachColPair = col10 * col01 * col11
    return np.sum(eachColPair)


def isViolated(matrix):
    if count3gametes(matrix) > 0:
    	return True
    else:
        return False


## number 2

# def greedyVC(matrix):
#     m = deepcopy(matrix)
#     nMuts = np.shape(m)[1]
#     columnPairs = list(itertools.permutations(range(nMuts), 2))
#     if isViolated(m) == True:
#         tVio = True
#     else:
#         tVio = False
#     while tVio == True:
#         columnPairs = sample(columnPairs, len(columnPairs))
#         vio = 0
#         while vio == 0:
#             col10 = np.count_nonzero(m[:, columnPairs[0][0]] > m[:,columnPairs[0][1]])
#             col01 = np.count_nonzero(m[:, columnPairs[0][0]] < m[:,columnPairs[0][1]])
#             col11 = np.count_nonzero((m[:, columnPairs[0][0]] + m[:,columnPairs[0][1]] == 2))    
#             vio = col10 * col01 * col11
#             if vio == 0:
#                 del columnPairs[0]
#             if len(columnPairs) == 0:
#                 return m
#         idx10 = np.where(m[:, columnPairs[0][0]] > m[:,columnPairs[0][1]])[0]
#         idx01 = np.where(m[:, columnPairs[0][0]] < m[:,columnPairs[0][1]])[0]
#         r10 = sample(idx10, 1)
#         r01 = sample(idx01, 1)
#         m1 = deepcopy(m)
#         m2 = deepcopy(m)
#         m1[r10[0], columnPairs[0][1]] = 1
#         m2[r01[0], columnPairs[0][0]] = 1
#         c10 = count3gametes(m1)
#         c01 = count3gametes(m2)
#         if c10 < c01:
#             m[r10[0], columnPairs[0][1]] = 1
#         elif c01 < c10:
#             m[r01[0], columnPairs[0][0]] = 1
#         else:
#             coin = np.random.randint(2, size = 1)
#             if coin[0] == 0:
#                 m[r10[0], columnPairs[0][1]] = 1
#             else:
#                 m[r01[0], columnPairs[0][0]] = 1
#         if len(idx10) == 1 and len(idx01) == 1:
#             del columnPairs[0]
#         tVio = isViolated(m)
#     return m


## number 1

def greedyVC(matrix):
    m = deepcopy(matrix)
    nMuts = np.shape(m)[1]
    columnPairs = list(itertools.permutations(range(nMuts), 2))
    if isViolated(m) == True:
        tVio = True
    else:
        tVio = False
    while tVio == True:
        columnPairs = sample(columnPairs, len(columnPairs))
        i = 0
        vio = 0
        while vio == 0:
            col10 = np.count_nonzero(m[:, columnPairs[i][0]] > m[:,columnPairs[i][1]])
            col01 = np.count_nonzero(m[:, columnPairs[i][0]] < m[:,columnPairs[i][1]])
            col11 = np.count_nonzero((m[:, columnPairs[i][0]] + m[:,columnPairs[i][1]] == 2))    
            vio = col10 * col01 * col11
            i += 1
            if i == len(columnPairs):
                return m
        idx10 = np.where(m[:, columnPairs[i-1][0]] > m[:,columnPairs[i-1][1]])[0]
        idx01 = np.where(m[:, columnPairs[i-1][0]] < m[:,columnPairs[i-1][1]])[0]
        r10 = sample(idx10, 1)
        r01 = sample(idx01, 1)
        m1 = deepcopy(m)
        m2 = deepcopy(m)
        m1[r10[0], columnPairs[i-1][1]] = 1
        m2[r01[0], columnPairs[i-1][0]] = 1
        c10 = count3gametes(m1)
        c01 = count3gametes(m2)
        if c10 < c01:
            m[r10[0], columnPairs[i-1][1]] = 1
        elif c01 < c10:
            m[r01[0], columnPairs[i-1][0]] = 1
        else:
            coin = np.random.randint(2, size = 1)
            if coin[0] == 0:
                m[r10[0], columnPairs[i-1][1]] = 1
            else:
                m[r01[0], columnPairs[i-1][0]] = 1
        if len(idx10) == 1 and len(idx01) == 1:
            del columnPairs[i-1]
        tVio = isViolated(m)
    return m



def main():
    # df = pd.read_csv('/data/mhaghir/Deep_data_noisy_10_25x25/mn2.txt', sep = '\t')
    # df.set_index('cellID/mutID', inplace = True)
    # print(df.values)
    # print(isViolated(df.values))
    # s = time()
    # o = greedyVC(df.values)
    # print(o)
    # print(np.sum(np.logical_xor(o, df.values)))
    # print(time() - s)
    nCells = 50
    nMuts = 50
    nTest = 500
    fRange = [1, 101]

    u = run_ms(nTest, nCells, nMuts, msdir)
    flips_g = np.zeros((nTest, fRange[1] - fRange[0]), dtype = np.int32)
    time_g = np.zeros((nTest, fRange[1] - fRange[0]), dtype = np.float32)
    for i in range(fRange[0], fRange[1]):
        for j in range(nTest):
            m_n = add_noise(u[j], i)
            st = time()
            o = greedyVC(m_n)
            dur = time() - st
            flips_g[j, i-1] = np.sum(np.logical_xor(o, m_n))
            time_g[j, i-1] = dur

    df = pd.DataFrame(flips_g, index = ['test{}'.format(k) for k in range(nTest)],\
        columns = ['num_flips_{}'.format(q) for q in range(fRange[0], fRange[1])])

    df1 = pd.DataFrame(time_g, index = ['test{}'.format(k1) for k1 in range(nTest)],\
        columns = ['num_flips_{}'.format(q1) for q1 in range(fRange[0], fRange[1])])

    df.to_csv('test_{nCells}x{nMuts}_1.csv'.format(nCells = nCells, nMuts = nMuts), sep = ',')
    df1.to_csv('test_{nCells}x{nMuts}_time_1.csv'.format(nCells = nCells, nMuts = nMuts), sep = ',')

if __name__ == '__main__':
    main()