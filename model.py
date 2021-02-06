
#import sys
#sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.io
import time

np.random.seed(1)
tf.set_random_seed(1)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t,p0,x1,y1,t1,p,x2,y2,t2,x3,y3,t3,pb,layers,layers_2):
        
        X = np.concatenate([x2, y2, t2], 1)
       # X0 = np.concatenate([x1, y1, t1], 1)
        
        self.lb = X.min(0)   #The normalized
        self.ub = X.max(0)  
        
        
        self.X = X
        
        self.x = x
        self.y = y
        self.t = t

        self.x1 = x1
        self.y1 = y1
        self.t1 = t1

        self.x2 = x2
        self.y2 = y2
        self.t2 = t2
        
        self.x3 = x3
        self.y3 = y3
        self.t3 = t3
        #self.u = u
        self.p0 = p0
        self.p = p
        self.pb = pb
#        self.v = v
        
        self.layers = layers
        self.layers_2=layers_2
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        self.weights_2, self.biases_2 = self.initialize_NN(layers_2)
        # Initialize parameters
        #self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)
        #self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.x1_tf = tf.placeholder(tf.float32, shape=[None, self.x1.shape[1]])
        self.y1_tf = tf.placeholder(tf.float32, shape=[None, self.y1.shape[1]])
        self.t1_tf = tf.placeholder(tf.float32, shape=[None, self.t1.shape[1]])
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        self.x2_tf = tf.placeholder(tf.float32, shape=[None, self.x2.shape[1]])
        self.y2_tf = tf.placeholder(tf.float32, shape=[None, self.y2.shape[1]])
        self.t2_tf = tf.placeholder(tf.float32, shape=[None, self.t2.shape[1]])    
        self.x3_tf = tf.placeholder(tf.float32, shape=[None, self.x3.shape[1]])
        self.y3_tf = tf.placeholder(tf.float32, shape=[None, self.y3.shape[1]])
        self.t3_tf = tf.placeholder(tf.float32, shape=[None, self.t3.shape[1]])
        self.pb_tf = tf.placeholder(tf.float32, shape=[None, self.pb.shape[1]])
        self.keep_prob = tf.placeholder(tf.float32)
#        self.h_tf = tf.placeholder(tf.float32, shape=[None, self.h.shape[1]])
#        self.h1_tf = tf.placeholder(tf.float32, shape=[None, self.h1.shape[1]])
  
        self.p0_tf = tf.placeholder(tf.float32, shape=[None, self.p0.shape[1]])
       # self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
       # self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])
#        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.u_pred, self.f_u0_pred = self.net_NS_0(self.x_tf, self.y_tf, self.t_tf)
        self.u1_pred,self.ur_pred = self.net_NS(self.x1_tf, self.y1_tf, self.t1_tf) 
        self.f_ub_pred,self.f_u_pred = self.net_NS1(self.x2_tf, self.y2_tf, self.t2_tf)
        self.f_b0_pred = self.net_NS2(self.x3_tf, self.y3_tf, self.t3_tf)
        
        self.loss_1 =  10**-6*tf.sqrt(tf.reduce_sum(tf.square(self.f_ub_pred)/3000)) + 10**-6*tf.sqrt(tf.reduce_sum(tf.square(self.f_b0_pred)/3000))+tf.reduce_sum( tf.maximum(0.0,self.f_u_pred-20000000)/3000)
        self.loss_2 =  10**5*tf.sqrt(tf.reduce_sum(tf.square(self.f_u0_pred)/3000))+ tf.reduce_sum(tf.maximum(0.0, self.p0_tf - self.u_pred)/3000) + tf.reduce_sum( tf.maximum(0.0,self.u_pred-20000000)/3000)
        self.loss_3 =  10**-6*tf.sqrt(tf.reduce_sum(tf.square(self.p_tf-self.ur_pred)/6000))
        self.loss_4 =  10**-5*tf.sqrt(tf.reduce_sum(tf.square(self.p_tf-self.u1_pred)/6000))
        #self.loss_4 =  10**-5*tf.sqrt(tf.reduce_sum(tf.square(self.u1_pred)/6000))
        
        
        self.loss1 =  self.loss_1+ self.loss_2 
        self.loss2 =  self.loss_4  +self.loss_3   # +0.1*tf.abs(self.lambda_1-3.8)
        self.loss=self.loss1 + self.loss2
                    
                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)       



             
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers)
        weights.append(())
        biases.append(()) 
        
        W = self.xavier_init(size=[layers[1], layers[2]])
        b = tf.Variable(tf.zeros([1,layers[2]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
        
        for l in range(2,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b =self.xavier_init(size=[1 , layers[l+1]] )
            #b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        #h=2.0*(h-h.min(0))/(h.max(0)-h.min(0))-1.0
        t=X[:,2:3]
        t=2.0*t/(600000) - 1.0
        #H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  #The normalized
        x = X[:,0:1] 
        y = X[:,1:2]
        r = tf.sqrt(x**2+y**2)
        h = tf.exp(0.2*(0.1 - r))   #  0.3 error 4.084460E-03 0.1 error 1.64051E-02 0.14 error 8E-03
       
        r = 2.0* r /(200*1.414)- 1.0 
        
        H = tf.concat([r,t,h],1)
        
        #H = tf.concat([r,t],1)
        
        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
           # H = tf.nn.dropout(H, rate=1-self.keep_prob) 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        Y =tf.tanh(Y)
        Y=(1+Y)*16e6
        return Y
    
    def neural_net_2(self, X, weights, biases):
        num_layers = len(weights) + 1
        t=X[:,2:3]
        t=2.0*t/(600000) - 1.0

        x = X[:,0:1] 
        y = X[:,1:2]
        r = tf.sqrt(x**2+y**2)
        r = 2.0* r /(200*1.414)- 1.0 
        
        H = tf.concat([r,t],1)
        
        for l in range(1,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
           # H = tf.nn.dropout(H, rate=1-self.keep_prob) 
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        Y =tf.tanh(Y)
        Y=Y*10e6
        return Y    
        
    
    
    def net_NS_0(self, x, y, t):
     
        #lambda_1 = self.lambda_1

        
        u = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
       # u = psi_and_p[:,0:1]
        
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        #u=10**6*u 
        
        B=1.05/(1+2*10**-9*(u-20000000))
        B0=1+2*10**-9*(u-20000000)
        fai=0.2*(1+1.5*10**-10*(u-20000000))
        det=2*10**-9*(u_x**2+u_y**2)
        
        f_u0 = (fai*2*10**-9/1.05 + 0.2*1.5*10**-10/B) * u_t - (1000*10**-14*3.8/ 1.05) * (det+B0 * (u_xx + u_yy))
        
        return u ,f_u0
    
    def net_NS(self, x, y, t):          #The internal boundary
        #lambda_1 = self.lambda_1
        #lambda_2 = self.lambda_2
        
        x1=4*x
        y1=4*y
        
        r=0.1
        h=10 
        Q=100/3600/24
        S=0
        #S=1
        #S=-1
       # C=10**-18
       #Cf=2*10**-9
       #Pref=20000000
        mu=10**-3
        u_0=self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        u_2 = self.neural_net_2(tf.concat([x,y,t], 1), self.weights_2, self.biases_2)
        u1 = self.neural_net(tf.concat([x1,y1,t], 1), self.weights, self.biases)
       # u_1 = self.neural_net_2(tf.concat([x1,y1,t], 1), self.weights_2, self.biases_2)
        #u = psi_and_p[:,0:1]
        #p = psi_and_p[:,1:2]
        
#        v = -tf.gradients(psi, x)[0]  
        ur=u_0 + u_2
        B=1.05/(1+2*10**-9*(ur-20000000))
        
        u_r = Q*B*mu/(2*np.pi*r*h*3.8*10**-14) #+ C*p_t
        f_w=u1 -u_r*3*r
        u_w= ur - Q*B*S*mu/(2*np.pi*h*3.8*10**-14)
    
        return  f_w, u_w #u ,
    
    def net_NS1(self, x, y, t):           #Outer boundary
        
        u = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        
        u_x = tf.gradients(u, x)[0]
         
        u_y = tf.gradients(u, y)[0]
        f_ub=tf.abs(u_x)+tf.abs(u_y)
        
         
        return f_ub , u
    
    def net_NS2(self, x, y, t):           #Initial pressure
        
        u = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        
        f_b0=u-20000000
         
        return f_b0
    
    
    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
  

      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t, self.x1_tf: self.x1, 
                   self.y1_tf: self.y1, self.t1_tf: self.t1,self.p_tf: self.p , self.p0_tf: self.p0,
                   self.x2_tf: self.x2, self.y2_tf: self.y2, self.t2_tf: self.t2,
                   self.x3_tf: self.x3,self.y3_tf: self.y3, self.t3_tf: self.t3,self.pb_tf: self.pb,
                   self.keep_prob: 1
                  }
        
           
           
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            #self.sess.run(self.train_op_Adam_2, tf_dict)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                #loss_value2 = self.sess.run(self.loss_2, tf_dict)
               # loss_value3 = self.sess.run(self.loss_3, tf_dict)
                loss_value3 = self.sess.run(self.loss_3, tf_dict)
                loss_value1 = self.sess.run(self.loss_1, tf_dict)
                
                loss_value = self.sess.run(self.loss, tf_dict)
                #loss2_value = self.sess.run(self.loss2, tf_dict)
               # lambda_1_value = self.sess.run(self.lambda_1)
      #          lambda_2_value = self.sess.run(self.lambda_2)

                print('It: %d, Loss: %.3e , Loss_1: %.3e, Loss_3: %.3e, Time: %.2f' % 
                      (it, loss_value, loss_value1, loss_value3, elapsed))
                
#                with open(f'./folder_for_nn/loss_L.txt','a', encoding='utf-8') as f:  
#                    f.write(str(it) + '   '+ str(loss_value) +'   '+ str(loss_value3) +'\n')
                    
                
                start_time = time.time()
            


#        self.optimizer.minimize(self.sess,
#                                feed_dict = tf_dict,
#                                fetches = [self.loss_3],
#                                loss_callback = self.callback)

        
#        f.close()
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        

        
#        save_path = self.saver.save(self.sess, "folder_for_nn/save_net2.ckpt")
#        print("Save to path: ", save_path)    
    
    def predict(self, x_star, y_star, t_star,x1_star, y1_star, t1_star):     
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star,self.keep_prob: 1}
        tf_dict1 = {self.x1_tf: x1_star, self.y1_tf: y1_star, self.t1_tf: t1_star,self.keep_prob: 1}
        u_star = self.sess.run(self.u_pred, tf_dict)
#        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.ur_pred, tf_dict1)
        
        return u_star   ,p_star    #, v_star, p_star

        

             

