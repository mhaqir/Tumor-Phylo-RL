#!/usr/bin/env python

# importing data from a file

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import pandas as pd
from random import sample
from random import seed
seed(30)
import copy
from time import time


from actor import Actor
from config import get_config, print_config
from dataset import DataSet


### Model: Critic (state value function approximator) = slim mean Attentive (parametric baseline ***)
###        w/ moving baseline (init_value default = 7 for TSP20, 20 for TSP40)       
###        Encoder = w/ FFN ([3] num_stacks / [16] num_heads / inner_FFN = 4*hidden_dim / [0.1] dropout_rate)
###        Decoder init_state = train, mean(enc)                                
###        Decoder inputs = Encoder outputs
###        Decoder Glimpse = Attention_g on ./(mask - first) + Residual connection



def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)
    
                     
    
    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0, max_to_keep= 1000)  

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())
    
        # Restore variables from disk.
        if config.restore_model==True:
            saver.restore(sess, config.restore_from)
            print("Model restored.")

        # Testing mode
        if config.inference_mode:

                  
            fn = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fn_r.txt')]
            fp = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fp_r.txt')]
            matrices_n_t = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.int8)
            matrices_p_t = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.int8)
            fp_fn = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.float32)
            s_m = sample(range(config.nLow ,config.nHigh), config.nTest)
            k = 0
            for i in s_m:
                
                l3 = pd.read_csv(config.input_dir_n + "/mn{}.txt".format(i), sep="\t")
                l3.set_index('cellID/mutID', inplace = True)
                matrices_n_t[k,:,:] = l3.values
                l2 = pd.read_csv(config.input_dir_p + "/mp{}.txt".format(i), sep="\t")
                l2.set_index('cellID/mutID', inplace = True)
                matrices_p_t[k,:,:] = l2.values

                fp_fn[k, matrices_n_t[k,:,:] == 1] = fp[i-1]
                fp_fn[k, matrices_n_t[k,:,:] == 0] = fn[i-1]
             
                k += 1
                     
            l = []
            for i in range(config.nCells):
                for j in range(config.nMuts):
                    l.append([i,j])
            l = np.asarray(l)

            a = np.expand_dims(matrices_n_t.reshape(-1, actor.max_length),2)
            b = np.expand_dims(fp_fn.reshape(-1, actor.max_length),2)
            x = np.tile(l,(config.nTest,1,1))
            c = np.squeeze(np.concatenate([x,b,a], axis = 2))
            d = np.asarray([np.take(c[i,:,:],np.random.permutation(c[i,:,:].shape[0]),axis=0,out=c[i,:,:]) for i in range(np.shape(c)[0])])
            
            f_1_0_rl = []
            f_0_1_rl = []
            V_o = []
            M_n = []
            t = []

            for j in tqdm(range(config.nTest)): # num of examples
                start_t = time()

                input_batch = np.tile(d[j,:,:],(actor.batch_size,1,1))
                
                feed = {actor.input_: input_batch}

                
                pos  = sess.run([actor.positions] , feed_dict=feed)[0]

                inp_ = tf.convert_to_tensor(input_batch, dtype=tf.float32)
                pos =  tf.convert_to_tensor(pos, dtype=tf.int32)

                
                r = tf.range(start = 0, limit = actor.batch_size, delta = 1)
                r = tf.expand_dims(r ,1)
                r = tf.expand_dims(r ,2)
                r3 = tf.cast(tf.ones([actor.max_length , 1]) * tf.cast(r, tf.float32), tf.int32)
                r4 = tf.squeeze(r, axis = 2)
                i = 0
                
                idx = tf.concat([r3, tf.cast(inp_[:,:,0:2], tf.int32)], axis = 2)
                m = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = inp_[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))
                c_v_o = actor.count3gametes(m)
                c_n = c_v_o[0].eval()

                
                while i < 35:
#                     print("Test for number of flips equal to {} ...".format(i))
#                     time1 = time()
                    r5 = tf.expand_dims(tf.fill([actor.batch_size], i), axis = 1)
                    u = tf.ones_like(r5)
                    r4_r5 = tf.concat([r4, r5], axis = 1)

                    pos_mask = tf.squeeze(tf.scatter_nd(indices = r4_r5, updates = u, shape = [actor.batch_size, actor.max_length, 1]), axis = 2)

                    pos_mask_cum1 = tf.cumsum(pos_mask, reverse = True, exclusive = True, axis = 1)
                    pos_mask_cum2 = tf.cumsum(pos_mask, reverse = False, exclusive = False, axis = 1) # for calculating NLL

                    per_pos = tf.concat([r3, tf.expand_dims(pos, axis = 2)], axis = 2)

                    per_ = tf.gather_nd(inp_, indices = per_pos)
            
                    per_matrix = per_[:,:,3:4]

                    # flipping the input
                    m1 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum1, tf.float32))
                    m1 = tf.subtract(tf.cast(pos_mask_cum1, tf.float32) , m1)
                    m2 = tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum2, tf.float32))
                    T_f = tf.add(m1, m2)

                    per_flipped = tf.concat([per_[:,:,0:3], tf.expand_dims(T_f, axis = 2)], axis = 2)
                    idx = tf.concat([r3, tf.cast(per_flipped[:,:,0:2], tf.int32)], axis = 2)
                    m_f = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = per_flipped[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))           
                    c_v = actor.count3gametes(m_f) # cost for flipped matrix
                    V_rl = c_v.eval()
                    g = np.min(V_rl)  

                    if g == 0:
#                         print("0")
                        gg = np.where(V_rl == g)[0][0]
                       # time2 = time()
                        c_v_rl = V_rl[gg]
                        m_rl = m_f.eval()[gg]                   
                        N10 = tf.reduce_sum(tf.multiply(tf.squeeze(per_matrix, axis = 2), tf.cast(pos_mask_cum1, tf.float32)), axis = 1, keepdims = True)
                        f_1_to_0_rl = tf.squeeze(N10, axis = 1).eval()[gg]
                        sum_mask_cum1 = tf.reduce_sum(tf.cast(pos_mask_cum1, tf.float32), axis = 1, keepdims = True )
                        N01 = tf.subtract(sum_mask_cum1, N10)
                        f_0_to_1_rl = tf.squeeze(N01, axis = 1).eval()[gg]
                        
                        # cost of original
                        idx = tf.concat([r3, tf.cast(inp_[:,:,0:2], tf.int32)], axis = 2)
                        m = tf.scatter_nd(indices = tf.expand_dims(idx,2), updates = inp_[:,:,3:4], shape = tf.constant([actor.batch_size, actor.config.nCells, actor.config.nMuts]))
                        c_v_o = actor.count3gametes(m)
                        c_n = c_v_o.eval()[0]
            
                        f_1_0_rl.append(f_1_to_0_rl)
                        f_0_1_rl.append(f_0_to_1_rl)
                        V_o.append(c_n)
                        M_n.append(s_m[j])
                        df = pd.DataFrame(m_rl.astype(int) , index = ['cell' + str(k1) for k1 in range(np.shape(m_rl)[0])], \
                                          columns = ['mut' + str(h1) for h1 in range(np.shape(m_rl)[1])])
                        df.index.rename('cellID/mutID', inplace=True)
                        df.to_csv(config.output_dir + '/mrl_{}.txt'.format(s_m[j]), sep='\t')
                        dur = time() - start_t
                        t.append(dur)
                       # time2_dur = time() - time2
                       # print("Time for 00 is {}".format(time2_dur))
#                         print("00")
                        break     
                    i += 1  
            df = pd.DataFrame(index = ["Test" + str(k) for k in range(len(M_n))], columns = ["V_o", "f_1_0_rl",\
                                                                                            "f_0_1_rl", "M_n", "time"])
            df["V_o"] = V_o
            df["f_1_0_rl"] = f_1_0_rl
            df["f_0_1_rl"] = f_0_1_rl
            df["M_n"] = M_n
            df["time"] = t
            df.to_csv(config.output_dir + '/test_{nCells}x{nMuts}.csv'.format(nCells = config.nCells, nMuts = config.nMuts), sep = ',')
            
if __name__ == "__main__":
    main()
