
import numpy as np
import pandas as pd

from config import get_config, print_config

class DataSet(object):

    # Initialize a DataSet
    def __init__(self, config):
        self.config = config

    # Read a batch for training procedure
    def train_batch(self, batch_num, fp, fn, l):
        matrices_n = np.zeros((self.config.batch_size, self.config.nCells, self.config.nMuts), dtype = np.int8)
#         matrices_p = np.zeros((self.config.batch_size, self.config.nCells, self.config.nMuts), dtype = np.int8)
        fp_fn = np.zeros((self.config.batch_size, self.config.nCells, self.config.nMuts), dtype = np.float32)
        k = 0
        for i in range(batch_num * self.config.batch_size + 1, (batch_num + 1) * self.config.batch_size + 1):
            
            l3 = pd.read_csv(self.config.input_dir_n + '/mn{}.txt'.format(i), sep="\t")
            l3.set_index('cellID/mutID', inplace = True)
            matrices_n[k,:,:] = l3.values
            fp_fn[k, matrices_n[k,:,:] == 1] = fp[i-1]
            fp_fn[k, matrices_n[k,:,:] == 0] = fn[i-1]
            k += 1
            
        a = np.expand_dims(matrices_n.reshape(-1, self.config.nCells*self.config.nMuts),2)
        b = np.expand_dims(fp_fn.reshape(-1, self.config.nCells*self.config.nMuts),2)
        x = np.tile(l,(self.config.batch_size,1,1))
        c = np.squeeze(np.concatenate([x,b,a], axis = 2))
        d = np.asarray([np.take(c[i,:,:],np.random.permutation(c[i,:,:].shape[0]),axis=0,out=c[i,:,:]) for i in range(np.shape(c)[0])])
        del matrices_n
        return d

    # Generate random batch for testing procedure
    def test_batch(self, batch_size, nCells, nMuts, seed=0):
        # Generate random TSP instance
        pass
    