# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:31:33 2020

@author: 10619
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


data1_loss =np.loadtxt(r'C:\Users\10619\Desktop\11\folder_for_nn\loss.txt')  
data2_loss = np.loadtxt(r'C:\Users\10619\Desktop\11\folder_for_nn\loss_L.txt') 


x = data1_loss[:,0]
y = data1_loss[:,1]
x1 = data2_loss[:,0]
y1 = data2_loss[:,1]

x2 = data1_loss[:,0]
y2 = data1_loss[:,2]
x3 = data2_loss[:,0]
y3 = data2_loss[:,2]



fig = plt.figure(figsize = (7,5))       #sizi
#ax1 = fig.add_subplot(1, 1, 1)


plt.axes(yscale = "log") 
pl.plot(x,y*1e-3,'g-',label=u'Loss with g_model', linestyle='--')

p2 = pl.plot(x1, y1*1e-3,'r-', label = u'Loss without g_model', linestyle='--')
pl.legend()

#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'Iters')
pl.ylabel(u'Training loss')
#plt.title('Compare loss for different models in training')

fig2 = plt.figure(figsize = (7,5))       #figsize
#ax1 = fig.add_subplot(1, 1, 1) 


plt.axes(yscale = "log") 
pl.plot(x2,y2*1e-3,'g-',label=u'MSE with g_model',linestyle='--')

p2 = pl.plot(x3, y3*1e-3,'r-', label = u'MSE without g_model',linestyle='--')
pl.legend()

#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'Iters')
pl.ylabel(u'MSE_f loss')
#plt.title('Compare loss for different models in training')






#import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  
#
##u_pred=np.load('./u_pred.npy')
##x_star=np.load('./x_star.npy')
##y_star=np.load('./y_star.npy') 


#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(x, y, z)

#ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
#ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
#ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
#plt.show()



