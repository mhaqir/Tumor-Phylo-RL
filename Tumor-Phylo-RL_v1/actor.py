import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

from encoder import Attentive_encoder
from decoder import Pointer_decoder
from critic import Critic
from config import get_config, print_config



# Tensor summaries for TensorBoard visualization
def variable_summaries(name,var, with_max_min=False):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        if with_max_min == True:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))



class Actor(object):


    def __init__(self, config):
        self.config=config

        # Data config
        self.batch_size = config.batch_size # batch size
        self.max_length = config.nCells*config.nMuts # input sequence length 
        self.input_dimension = config.input_dimension # dimension of a input

        # Reward config
        #self.avg_baseline = tf.Variable(config.init_baseline, trainable=False, name="moving_avg_baseline") # moving baseline for Reinforce
        self.alpha = config.alpha # moving average update

        # Training config (actor)
        self.global_step= tf.Variable(0, trainable=False, name="global_step") # global step
        self.lr1_start = config.lr1_start # initial learning rate
        self.lr1_decay_rate= config.lr1_decay_rate # learning rate decay rate
        self.lr1_decay_step= config.lr1_decay_step # learning rate decay step

        # Training config (critic)
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # global step
        self.lr2_start = config.lr1_start # initial learning rate
        self.lr2_decay_rate= config.lr1_decay_rate # learning rate decay rate
        self.lr2_decay_step= config.lr1_decay_step # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_coordinates")
        self.inp = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_coordinates")
        
        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()
        self.merged = tf.summary.merge_all()

    
    def count3gametes(self, input_):
        #input_ = tf.reshape(input_ , [input_.get_shape()[0], self.config.nCells, self.config.nMuts])
        columnPairs = list(itertools.permutations(range(self.config.nMuts), 2))
        nColumnPairs = len(columnPairs)
        columnReplicationList = np.array(columnPairs).reshape(-1)
        l = []
        for i in range(input_.get_shape()[0]):
            for j in range(self.config.nCells):
                for k in columnReplicationList:
                    l.append([i,j,k])
        replicatedColumns = tf.reshape(tf.gather_nd(input_, l), [input_.get_shape()[0], self.config.nCells, len(columnReplicationList)])
        replicatedColumns = tf.transpose(replicatedColumns, perm = [0,2,1])
        x = tf.reshape(replicatedColumns, [input_.get_shape()[0], nColumnPairs, 2, self.config.nCells])
        col10 = tf.count_nonzero(tf.math.greater(x[:,:,0,:], x[:,:,1,:]), axis = 2)# batch_size * nColumnPairs
        col01 = tf.count_nonzero(tf.math.greater(x[:,:,1,:], x[:,:,0,:]), axis = 2)# batch_size * nColumnPairs
        col11 = tf.count_nonzero(tf.math.equal(x[:,:,0,:]+x[:,:,1,:],2), axis = 2)# batch_size * nColumnPairs
        eachColPair = col10 * col01 * col11 # batch_size * nColumnPairs
        return tf.reduce_sum(eachColPair, axis = 1) # batch_size 

    
    def build_permutation(self):

        with tf.variable_scope("encoder"):

            Encoder = Attentive_encoder(self.config)
            encoder_output = Encoder.encode(self.input_)

        with tf.variable_scope('decoder'):
            # Ptr-net returns permutations (self.positions), with their log-probability for backprop
            self.ptr = Pointer_decoder(encoder_output, self.config)
            self.positions, self.log_softmax = self.ptr.loop_decode()
            variable_summaries('log_softmax',self.log_softmax, with_max_min = True)
            


    def build_critic(self):

        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config)
            self.critic.predict_rewards(self.input_)
            variable_summaries('predictions',self.critic.predictions, with_max_min = True)


    def build_reward(self):

        with tf.name_scope('permutations'):
            
            # Reorder input % tour
            alpha = 1
            beta = 0.001
            x = tf.zeros([int(self.max_length/2), self.batch_size], tf.float32)
            for i in range(int(self.max_length/2)):
                inp = tf.identity(self.input_)
                pos = tf.expand_dims(self.positions, 2)
                po = tf.identity(pos[:,:i,:])
                r = tf.range(start = 0, limit = self.batch_size, delta = 1)
                r = tf.expand_dims(r ,1)
                r = tf.expand_dims(r ,2)
                
                r1 = tf.cast(tf.ones([self.max_length, 1]) * tf.cast(r, tf.float32), tf.int32)
                idx = tf.concat([r1, tf.cast(inp[:,:,0:2], tf.int32)], axis = 2)
                m = tf.scatter_nd(indices= tf.expand_dims(idx,2) ,updates = inp[:,:,3:4], shape = tf.constant([self.batch_size, self.config.nCells, self.config.nMuts]))
                
                r2 = tf.cast(tf.ones([i, 1]) * tf.cast(r, tf.float32), tf.int32)            
                
                s = tf.stack([r2, po], axis = 2)
                s = tf.squeeze(s, axis = 3)  #indices
                o1 = tf.gather_nd(inp, s)
                fp_fn = tf.squeeze(o1[:,:,2:3], axis = 2)
                c_fp_fn = tf.expand_dims(tf.scalar_mul(beta,tf.reduce_sum(fp_fn, axis = 1)), axis = 0)
                idx1 = tf.concat([r2, tf.cast(o1[:,:,0:2], tf.int32)], axis = 2)
                sc1 = tf.scatter_nd(indices = tf.expand_dims(idx1,2) ,updates = o1[:,:,3:4], shape = m.get_shape())

                m_flip = tf.subtract(tf.ones_like(m), m)
                o2 = tf.gather_nd(m_flip , idx1)

                sc2 = tf.scatter_nd(indices = idx1 ,updates = o2, shape = m.get_shape())

                new_val = tf.subtract(m , sc1)
                final_val = tf.add(new_val, sc2)

                c_g = tf.cast(self.count3gametes(final_val) , tf.float32)
                c_g = tf.expand_dims(c_g, axis = 0)
                c_f = (alpha*i)/(int(self.max_length/2))
                c_f_t = tf.fill(c_g.get_shape(), c_f)
                c_t = tf.add_n([c_g, c_f_t, c_fp_fn])
                ind = []
                for i1 in range(x.get_shape()[1]):
                    ind.append([i,i1])
                ind = tf.convert_to_tensor(ind)
                ind = tf.expand_dims(ind , axis = 0)
                x_n = tf.scatter_nd(indices = ind , updates = c_t, shape = x.get_shape())
                x = tf.add(x_n, x)
            x_m = tf.reduce_min(x, axis = 0)
            self.cost = tf.identity(x_m)
            
        with tf.name_scope('environment'):
            
            #cost = self.count3gametes(self.inp)
            cost = tf.identity(self.cost)

            # Define reward from tour length
            self.reward = tf.cast(cost,tf.float32)
            variable_summaries('reward',self.reward, with_max_min = True)


    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            with tf.name_scope('baseline'):
                # Update baseline
                reward_mean, reward_var = tf.nn.moments(self.reward,axes=[0])
                #self.base_op = tf.assign(self.avg_baseline, self.alpha(1.0-self.alpha)*reward_mean)
                #tf.summary.scalar('average baseline',self.avg_baseline)

            with tf.name_scope('reinforce'):
                # Actor learning rate
                self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,self.lr1_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1,beta1=0.9,beta2=0.99, epsilon=0.0000001)
                # Discounted reward
                self.reward_baseline = tf.stop_gradient(self.reward  - self.critic.predictions) # [Batch size, 1] 
                variable_summaries('reward_baseline',self.reward_baseline, with_max_min = True)
                # Loss
                self.loss1 = tf.reduce_mean(self.reward_baseline*self.log_softmax,0)
                tf.summary.scalar('loss1', self.loss1)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None] # L2 clip
                self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Critic learning rate
                self.lr2 = tf.train.exponential_decay(self.lr2_start, self.global_step2, self.lr2_decay_step,self.lr2_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2,beta1=0.9,beta2=0.99, epsilon=0.0000001)
                # Loss
                weights_ = 1.0 #weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
                self.loss2 = tf.losses.mean_squared_error(self.reward, self.critic.predictions, weights = weights_)
                tf.summary.scalar('loss2', self.loss1)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None] # L2 clip
                self.train_step2 = self.opt1.apply_gradients(capped_gvs2, global_step=self.global_step2)





if __name__ == "__main__":
    # get config
    config, _ = get_config()

    # Build Model and Reward from config
    actor = Actor(config)

    print("Starting training...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print_config()

        solver = [] #Solver(actor.max_length)
        training_set = DataGenerator(solver)

        nb_epoch=2
        for i in tqdm(range(nb_epoch)): # epoch i

            # Get feed_dict
            input_batch  = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
            feed = {actor.input_: input_batch}
            #print(' Input \n', input_batch)

            permutation, distances = sess.run([actor.positions, actor.distances], feed_dict=feed) 
            print(' Permutation \n',permutation)
            print(' Tour length \n',distances)


        variables_names = [v.name for v in tf.global_variables() if 'Adam' not in v.name]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k, "Shape: ", v.shape)