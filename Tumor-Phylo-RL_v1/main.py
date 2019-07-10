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
    dataset = DataSet(config)
    
    # Creating dataset
    if not config.inference_mode:
        
        fn = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fn_r.txt')]
        fp = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fp_r.txt')]  
        l = []
        for i in range(config.nCells):
            for j in range(config.nMuts):
                l.append([i,j])
        l = np.asarray(l)


    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0, max_to_keep= 1000)  

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model==True:
            saver.restore(sess, config.restore_from + "/actor.ckpt")
            print("Model restored.")

        # Training mode
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.starting_num, config.nb_epoch)): 
                # Get feed dict
                
                # Save the variables to disk
                if i % 500 == 0:
                    save_path = saver.save(sess, config.save_to + "/tmp.ckpt", global_step=i)
                    print("\n Model saved in file: %s" % save_path)
                    
                
#                 input_batch = sess.run(next_element)

                feed = {actor.input_: dataset.train_batch(i, fp, fn, l)}

                # Forward pass & train step
                summary, train_step1, train_step2 = sess.run([actor.merged, actor.train_step1, actor.train_step2], feed_dict=feed)

                if i % (config.nb_epoch-1) == 0:
                    writer.add_summary(summary,i)


            print("Training COMPLETED !")
            saver.save(sess, config.save_to + "/actor.ckpt")


        # Inference mode
        else:
            
            f_1_to_0_o = np.zeros((config.nTest, 1), dtype = np.float64)
            f_0_to_1_o = np.zeros((config.nTest, 1), dtype = np.float64)
            M_n = np.zeros((config.nTest, 1), dtype = np.float64)
            
            
            fn = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fn_r.txt')]
            fp = [float(line.rstrip('\n')) for line in open(config.input_dir_n + '/fp_r.txt')] 
            matrices_n_t = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.int8)
            matrices_p_t = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.int8)
            fp_fn = np.zeros((config.nTest, config.nCells, config.nMuts), dtype = np.int16)
            s_m = sample(range(config.nLow ,config.nHigh), config.nTest)
            
            k = 0
            for i in s_m:
                l3 = pd.read_csv(config.input_dir_n + "/mn{}.txt".format(i), sep="\t")
                l3.set_index('cellID/mutID', inplace=True)
                matrices_n_t[k,:,:] = l3.values
                l2 = pd.read_csv(config.input_dir_p + "/mp{}.txt".format(i), sep="\t")
                l2.set_index('cellID/mutID', inplace=True)
                matrices_p_t[k,:,:] = l2.values
                
                N01_o_ = np.sum(matrices_n_t[k,:,:] - matrices_p_t[k,:,:] == -1) 
                N10_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == -1)
                N11_o_ = np.sum(matrices_p_t[k,:,:] + matrices_n_t[k,:,:] == 2)
                N00_o_ = np.sum(matrices_p_t[k,:,:] - matrices_n_t[k,:,:] == 0) - N11_o_
                
                f_1_to_0_o[k, 0] = N10_o_
                f_0_to_1_o[k, 0] = N01_o_
                M_n[k, 0] = i
                
                fp_fn[k, matrices_n[k,:,:] == 1] = fp[i-1]
                fp_fn[k, matrices_n[k,:,:] == 0] = fn[i-1]
                
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
            np.take(c,np.random.permutation(c.shape[1]),axis=1,out=c)

            
            output_ = np.zeros((config.nTest, 10), dtype = np.float64)
            for j in tqdm(range(config.nTest)): # num of examples
                
                start_t = time()

                input_batch = np.tile(c[j,:,:],(actor.batch_size,1,1))

                
                feed = {actor.input_: input_batch}

                
                pos  = sess.run([actor.positions] , feed_dict=feed)[0]

                inp_ = tf.convert_to_tensor(input_batch, dtype=tf.float32)
                pos =  tf.convert_to_tensor(pos, dtype=tf.int32)

                pos = tf.expand_dims(pos, 2)
                

                alpha = 1
                beta = 0.001
                x = tf.zeros([int(actor.max_length/2), actor.batch_size], tf.float32)
                i = 0
                c2 = 0
                
                while i < 20:     #actor.max_length:
                    
                    inpu_ = tf.identity(inp_)
                    po = pos[:,:i,:]
                    r = tf.range(start = 0, limit = actor.batch_size, delta = 1)
                    r = tf.expand_dims(r ,1)
                    r = tf.expand_dims(r ,2)


                    r1 = tf.cast(tf.ones([actor.max_length, 1]) * tf.cast(r, tf.float32), tf.int32)
                    idx = tf.concat([r1, tf.cast(inpu_[:,:,0:2], tf.int32)], axis = 2)
                    m = tf.scatter_nd(indices = tf.expand_dims(idx,2) ,updates = inpu_[:,:,3:4], shape = tf.constant([actor.batch_size, config.nCells, config.nMuts])) # niosy batch


                    r2 = tf.cast(tf.ones([i, 1]) * tf.cast(r, tf.float32), tf.int32)            
                    s = tf.stack([r2, po], axis = 2)
                    s = tf.squeeze(s,axis = 3)  #indices

                    o1 = tf.gather_nd(inpu_, s)
                    fp_fn = tf.squeeze(o1[:,:,2:3], axis = 2)
                    count_flip = tf.squeeze(o1[:,:,3:4], axis = 2)
                    t1_flip = tf.reduce_sum(tf.cast(tf.equal(count_flip, 0), tf.int32), axis = 1) # count the number of 
                    s_flip = tf.fill(t1_flip.get_shape(),i)                                  # flips from 0 to 1 and 
                    t2_flip = tf.subtract(s_flip, t1_flip)                                  # 1 to 0
                    
                    flip_1_to_0 = tf.identity(t2_flip)
                    flip_0_to_1 = tf.identity(t1_flip)                       
                    
                    c_fp_fn = tf.expand_dims(tf.scalar_mul(beta,tf.reduce_sum(fp_fn, axis = 1)), axis = 0)

                    idx1 = tf.concat([r2, tf.cast(o1[:,:,0:2], tf.int32)], axis = 2)
                    sc1 = tf.scatter_nd(indices = tf.expand_dims(idx1,2) ,updates = o1[:,:,3:4], shape = m.get_shape())

                    m_flip = tf.subtract(tf.ones_like(m), m)
                    o2 = tf.gather_nd(m_flip , idx1)

                    sc2 = tf.scatter_nd(indices = idx1 ,updates = o2, shape = m.get_shape())
                    new_val = tf.subtract(m , sc1)
                    final_val = tf.add(new_val, sc2)


                    c_v = tf.cast(actor.count3gametes(final_val) , tf.float32)
                    c_v = tf.expand_dims(c_v, axis = 0)  # Violation cost after flipping
                    c_v_m = tf.reduce_min(c_v, axis = 1)
                    if c_v_m.eval() == 0:  # once it can solve this instance
#                         print("0")
                        c_v_am = tf.argmin(c_v, axis=1).eval()
                        m_rl = final_val[tf.cast(c_v_am[0], tf.int32),:,:].eval()  # output matrix after flipping
                        m_m = m[tf.cast(c_v_am[0], tf.int32),:,:].eval() # matrix before flipping
                        c1 = actor.count3gametes(m)
                        
                        c_f = (alpha*i)/(int(actor.max_length/2))
                        c_f_t = tf.fill(c_v.get_shape(), c_f)
                        c_f_fp_fn = tf.add_n([c_f_t, c_fp_fn]) # cost of number of flips and fp-fn

                        
                        c2 = copy.deepcopy(c_f_fp_fn[:,tf.cast(c_v_am[0], tf.int32)].eval())  # which is always 0 since it could solve the instance
                        c_v1 = c_v_m.eval()
                        n_f = copy.deepcopy(i)  # required number of flips to solve this instance
                        c_n = c1[tf.cast(c_v_am[0], tf.int32)].eval()  # noisy matrix cost
                        f_0_to_1 = flip_0_to_1[tf.cast(c_v_am[0], tf.int32)].eval()
                        f_1_to_0 = flip_1_to_0[tf.cast(c_v_am[0], tf.int32)].eval()


                        df = pd.DataFrame(m_rl.astype(int) , index = ['cell' + str(k1) for k1 in range(np.shape(m_rl)[0])], \
                                          columns = ['mut' + str(h1) for h1 in range(np.shape(m_rl)[1])])
                        df.index.rename('cellID/mutID', inplace=True)
                        df.to_csv(config.output_dir + '/mrl_{}.txt'.format(s_m[j]), sep='\t') 
#                         print("00")
                        break

                    c_f = (alpha*i)/(int(actor.max_length/2))
                    c_f_t = tf.fill(c_v.get_shape(), c_f)
                    c_f_fp_fn = tf.add_n([c_v, c_f_t, c_fp_fn]) # cost of number of flips and fp-fn
                
                    V_rl = tf.squeeze(c_v, axis = 0).eval()
                    g = np.min(V_rl)
                    g_w = np.where(V_rl == g)[0]
                    g_w_f_fp_fn = np.argmin(tf.squeeze(c_f_fp_fn, axis = 0).eval()[g_w])
                    gg = g_w[g_w_f_fp_fn]
                    
                    if i == 0:
#                         print("1")
                        c2 = copy.deepcopy(tf.squeeze(c_f_fp_fn, axis = 0).eval()[gg])
                        c_v1 = V_rl[gg]
                        n_f = copy.deepcopy(i)
                        f_0_to_1 = 0
                        f_1_to_0 = 0
                        m_rl = final_val.eval()[gg]

#                         print("11")

                    if c2 > tf.squeeze(c_f_fp_fn, axis = 0).eval()[gg]:
#                         print("2")
                        c2 = copy.deepcopy(tf.squeeze(c_f_fp_fn, axis = 0).eval()[gg])  # minimum cost we can get
                        c_v1 = V_rl[gg]
                        n_f = copy.deepcopy(i)
                        f_0_to_1 = flip_0_to_1[gg].eval()  #tf.cast(c_t_am[0], tf.int32)
                        f_1_to_0 = flip_1_to_0[gg].eval()
                        m_rl = final_val.eval()[gg]

#                         print("22")
                    
                    if i == 19:      #(actor.max_length - 1):
#                         print("3")
                        c1 = actor.count3gametes(m)
                        c_n = c1[0].eval()
                        df = pd.DataFrame(m_rl.astype(int) , index = ['cell' + str(k1) for k1 in range(np.shape(m_rl)[0])], \
                                          columns = ['mut' + str(h1) for h1 in range(np.shape(m_rl)[1])])
                        df.index.rename('cellID/mutID', inplace=True)
                        df.to_csv(config.output_dir + '/mrl_{}.txt'.format(s_m[j]), sep='\t') 
#                         print("33")
                    i += 1
    
                dur_t = time() - start_t
    
                output_[j,0] = c2
                output_[j,1] = c_v1
                output_[j,2] = c_n
                output_[j,3] = n_f
                output_[j,4] = f_0_to_1
                output_[j,5] = f_1_to_0
                output_[j,6] = dur_t
            output_[:,7] = np.squeeze(f_1_to_0_o)
            output_[:,8] = np.squeeze(f_0_to_1_o)
            output_[:,9] = np.squeeze(M_n)
                
            df = pd.DataFrame(output_, index = ["test" + str(k) for k in range(config.nTest)], \
                             columns = ["c_f_fp_fn", "V_rl", "V_o", "n_f", "f_0_to_1_rl", "f_1_to_0_rl", "time", "f_1_to_0_o", "f_0_to_1_o", "M_n"])
            df.to_csv(config.output_dir + '/test_{nCells}x{nMuts}.csv'.format(nCells = config.nCells, nMuts = config.nMuts), sep = ',')
            
            


if __name__ == "__main__":
    main()