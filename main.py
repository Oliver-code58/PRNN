# -*- coding: utf-8 -*-
import numpy as np
import model
import scipy.io

if __name__ == "__main__": 
      
    N_train = 3000
    N1_train = 6000
    N2_train=3000
    
    layers = [2, 3, 10, 10, 10,10,10,10, 1]
    #layers = [3, 3, 10, 10, 10,10,10,10, 1]
    layers_2 = [2,2, 10, 10 ,10, 10 , 1]
    # Load Data
    data = scipy.io.loadmat(R'./sentou0705.mat')
           
    U_star = data['U'] # N x T
    P0_star = 10**6*data['PP']
    Pb_star = 10**6*data['P0']
    P_star = 10**6*data['P'] # N x T
    t_star = data['tt'] # T x 1
    t0_star = data['t0'] # T x 1
    X_star = data['X'] # N x 2
    X1_star = data['X1'] # N1 x 2
    X2_star = data['X2'] # N2 x 2
    X3_star = data['X3'] # N2 x 2
    
    N = X_star.shape[0]
    N1 = X1_star.shape[0]
    N2 = X2_star.shape[0]
    N3 = X3_star.shape[0]
    T = t_star.shape[0]
    T1 = t0_star.shape[0]
    # Rearrange Data 
    #num=213
    
    XX = np.tile(X_star[:,0:1], (1,T)) # Num x T
    YY = np.tile(X_star[:,1:2], (1,T)) # Num x T
    XX1 = np.tile(X1_star[:,0:1], (1,T)) # Num x T
    YY1 = np.tile(X1_star[:,1:2], (1,T)) # Num x T
    XX2 = np.tile(X2_star[:,0:1], (1,T)) # Num x T
    YY2 = np.tile(X2_star[:,1:2], (1,T)) # Num x T  
    XX3 = np.tile(X3_star[:,0:1], (1,T1)) # Num x T1
    YY3 = np.tile(X3_star[:,1:2], (1,T1)) # Num x T1 
    TT = np.tile(t_star, (1,N)).T # Num x T
    TT1 = np.tile(t_star, (1,N1)).T # Num x T
    TT2 = np.tile(t_star, (1,N2)).T # Num x T
    TT3 = np.tile(t0_star, (1,N3)).T # Num x T
    
   # TT0 = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,:] # N x T
    P0 = P0_star[:,:]
    PB = Pb_star[:,:]
    PP = P_star[:,:] # N1 x T
#    VV = U_star[:,1,:] # N x T
  #  PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    x1 = XX1.flatten()[:,None] # NT x 1
    y1 = YY1.flatten()[:,None] # NT x 1
    t1 = TT1.flatten()[:,None] # NT x 1
    
    x2 = XX2.flatten()[:,None] # NT x 1
    y2 = YY2.flatten()[:,None] # NT x 1
    t2 = TT2.flatten()[:,None] # NT x 1    
    
    x3 = XX3.flatten()[:,None] # NT x 1
    y3 = YY3.flatten()[:,None] # NT x 1
    t3 = TT3.flatten()[:,None] # NT x 1   
    
    u = UU.flatten()[:,None] # NT x 1
    p0 = P0.flatten()[:,None]
    pb = PB.flatten()[:,None]
#    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # N1T x 1
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
   # u_train = u[idx,:]
    p0_train = p0[idx,:]
#    v_train = v[idx,:]
   
    idx = np.random.choice(N1*T, N1_train, replace=False)
    x1_train = x1[idx,:]
    y1_train = y1[idx,:]
    t1_train = t1[idx,:]
    p_train = p[idx,:]
    
    idx = np.random.choice(N2*T, N2_train, replace=False)
    x2_train = x2[idx,:]
    y2_train = y2[idx,:]
    t2_train = t2[idx,:]  
    
    idx = np.random.choice(N3*T1, N2_train, replace=False)
    x3_train = x3[idx,:]
    y3_train = y3[idx,:]
    t3_train = t3[idx,:]  
    pb_train = pb[idx,:]
    
    # Training
    model = model.PhysicsInformedNN(x_train, y_train, t_train,p0_train, x1_train, y1_train, t1_train,p_train, x2_train,y2_train,t2_train,x3_train, y3_train, t3_train,pb_train, layers,layers_2)
    model.train(5000)
    
    # Test Data
    snap = np.array([93])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    x1_star = X1_star[:,0:1]
    y1_star = X1_star[:,1:2]
    t1_star = TT1[:,snap]
    
    u_star = U_star[:,0:1]
#    v_star = U_star[:,1,snap]
#    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, p_pred= model.predict(x_star, y_star, t_star,x1_star, y1_star, t1_star)
    
    snap1 = np.array([186])
    x_star1 = X_star[:,0:1]
    y_star1 = X_star[:,1:2]
    t_star1 = TT[:,snap1]
    x1_star1 = X1_star[:,0:1]
    y1_star1 = X1_star[:,1:2]
    t1_star1 = TT1[:,snap1]
    u_pred1, p_pred1= model.predict(x_star1, y_star1, t_star1,x1_star1, y1_star1, t1_star1)
    
    snap2 = np.array([322])
    x_star2 = X_star[:,0:1]
    y_star2 = X_star[:,1:2]
    t_star2 = TT[:,snap2]
    x1_star2 = X1_star[:,0:1]
    y1_star2 = X1_star[:,1:2]
    t1_star2 = TT1[:,snap2]
    u_pred2, p_pred2= model.predict(x_star2, y_star2, t_star2,x1_star2, y1_star2, t1_star2)
    

    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    
    print('Error u: %e' % (error_u))    
