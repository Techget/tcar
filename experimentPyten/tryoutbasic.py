from pyten.tools import create  # Import the problem creation function

problem = 'basic'  # Define Problem As Basic Tensor Completion Problem
siz = [20, 20, 20]  # Size of the Created Synthetic Tensor
r = [4, 4, 4]  # Rank of the Created Synthetic Tensor
miss = 0.8  # Missing Percentage
tp = 'CP'  # Define Solution Format of the Created Synthetic Tensor As 'CP decomposition'
[X1, Omega1, sol1] = create(problem, siz, r, miss, tp)
print("Omega1 shape: ", Omega1.shape)
print(X1.data)
# print(Omega1)

# Basic Tensor Completion with methods: CP-ALS,Tucker-ALS, FaLRTC, SiLRTC, HaLRTC, TNCP
from pyten.method import *

r = 4  # Rank for CP-based methods
R = [4, 4, 4]  # Rank for tucker-based methods
# CP-ALS
[T1, rX1] = cp_als(X1, r, Omega1)  # if no missing data just omit Omega1 by using [T1,rX1]=cp_als.cp_als(X1,r)
# print sol1.totensor().data
# print rX1.data

# Tucker-ALS
[T2, rX2] = tucker_als(X1, R, Omega1)  # if no missing data just omit Omega1
# FalRTC, SiLRTC, HaLRTC
rX3 = falrtc(X1, Omega1)
rX4 = silrtc(X1, Omega1)
rX5 = halrtc(X1, Omega1)
# TNCP
self1 = TNCP(X1, Omega1, rank=r)
self1.run()

# Error Testing
from pyten.tools import tenerror

realX = sol1.totensor()
[Err1, ReErr11, ReErr21] = tenerror(rX1, realX, Omega1)
[Err2, ReErr12, ReErr22] = tenerror(rX2, realX, Omega1)
[Err3, ReErr13, ReErr23] = tenerror(rX3, realX, Omega1)
[Err4, ReErr14, ReErr24] = tenerror(rX4, realX, Omega1)
[Err5, ReErr15, ReErr25] = tenerror(rX5, realX, Omega1)
[Err6, ReErr16, ReErr26] = tenerror(self1.X, realX, Omega1)
# print ('\n', 'The Relative Error of the Six Methods are:', ReErr21, ReErr22, ReErr23, ReErr24, ReErr25, ReErr26)
print ('\n', 'The frobenius norm of Error of the Six Methods are:', Err1, Err2, Err3, Err4, Err5, Err6)
