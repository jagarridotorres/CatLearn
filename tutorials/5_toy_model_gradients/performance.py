import numpy as np
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer
from scipy.spatial import distance

# Method 1: broadcast, Method 2: for_train_loop, Method3: old for loops.
# Method 4: broadcast+cdist

method = '3'
np.random.seed(1)
m1 = []
train_points = 500
dimensions = 500
iterations = 5

m1= 1.2*np.random.randint(5.0, size=(train_points,
dimensions))


kwidth = np.zeros(np.shape(m1)[1])+2.0
time=[]

for i in range(0,iterations):
    start=timer()
    loop_number = i

    if method=='1':
        size = np.shape(m1)
        d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        den = np.sum(invsqkwidth*d**2,axis=-1)
        k1 = np.exp(-den/2)
        # print(k1)
        big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
        size[0]*size[1])
        # print(big_kgd)

    if method=='2':
        size = np.shape(m1)
        big_kgd = np.zeros((size[0],size[0]*size[1]))
        d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        den = np.sum(invsqkwidth*d**2,axis=-1)
        k1 = np.exp(-den/2)
        # print(k1)
        for i in range(size[0]):
            big_kgd_i = ((invsqkwidth * d[i]).T * k1[i]).T
            big_kgd[:,(size[1]*i):(size[1]+size[1]*i)] = big_kgd_i
        # print(big_kgd)

    if method=='3':
         k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
         k1 = distance.squareform(np.exp(-.5 * k1))
         np.fill_diagonal(k1, 1)
         # print(k1)
         big_kgd = gkernels.big_kgd('squared_exponential',kwidth,m1)
         # print(big_kgd)

    if method=='4':
        size = np.shape(m1)
        k1 = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k1 = distance.squareform(np.exp(-.5 * k1))
        np.fill_diagonal(k1, 1)
        d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
        invsqkwidth = 1/kwidth**2
        # print(k1)
        big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
        size[0]*size[1])
        # print(big_kgd)


    end = timer()
    time_one_loop = end-start
    time.append(time_one_loop)
    print('Elapsed time iteration',loop_number,time_one_loop)


print('Average time', np.average(time))
print('Best', np.min(time))









