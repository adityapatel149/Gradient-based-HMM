from math import e, log
import numpy as np
from itertools import product
from numba import njit


N = 2
M = 27
T = 50000
minIters = 100
threshold = 0.00001
ta = 4
W = np.zeros((N,N))
V = np.zeros((N,M))
a = 5


#Generate observation sequence from Brown corpus
f = open('browncorpus.txt', 'r')
filtered_f = open('filtered_browncorpus.txt', 'w')
for line in f:
    filtered_line = ''.join([char.lower() for char in line if char.isalpha() or char == ' '])
    filtered_f.write(filtered_line)

filtered_f.close()

filtered_f = open('filtered_browncorpus.txt', 'r')
char_obs_seq = filtered_f.read(T)

#Map characters to integers. stored as a dictionary
unique_chars = sorted(set(char_obs_seq))
char_to_int = {char:i for i,char in enumerate(unique_chars)}
#convert observation sequence to sequence of integers
obs_seq = [char_to_int[char] for char in char_obs_seq]



# Initialise Model
rand_values = np.random.uniform(-0.01,0.01, (N, N))
A = np.full((N,N), 1.0/N)
A += rand_values #add or subtract random values from all elements of A
A = np.clip(A, 0, None) # clip array A with minimum 0, maximum None (A should only have positive values)
A = A / A.sum(axis=1, keepdims=True) # normalise values in A so row sums to 1

rand_values = np.random.uniform(-0.01,0.01, (N, M))
B = np.full((N,M), 1.0/M)
B += rand_values
B = np.clip(B, 0, None)
B = B / B.sum(axis =1, keepdims = True)

rand_values = np.random.uniform(-0.1,0.1, N)
pi = np.full(N, 1.0/N)
pi += rand_values
pi = np.clip(pi, 0, None)
pi = pi / pi.sum()

@njit
def hmm(obs_seq, pi: np.ndarray, A:np.ndarray, B: np.ndarray, c: np.ndarray,W: np.ndarray,V:np.ndarray):
    oldLogProb = float('-inf')
    for itera in range(minIters):
        print(itera)
        #Forward algorithm    
        alpha = np.zeros((T,N))
        c[0] = 0.0

        alpha[0] = pi * B[:, obs_seq[0]]   
        c[0] = 1/alpha[0].sum()
        alpha[0] *= c[0] 
        
        for t in range(1, T):
            c[t] = 0.0
            alpha[t] = np.dot(alpha[t-1],A) * B [:, obs_seq[t]]
            c[t] = 1/alpha[t].sum()
                
            alpha[t] *= c[t]
        
        #Backward algorithm:
        beta = np.zeros((T,N))
        beta[T-1] = c[T-1]

        for t in range(T-2,-1,-1):
            beta[t] = np.dot(A, (B[:,obs_seq[t+1]]*beta[t+1]))
            beta[t] *= c[t]
    
        #Calculate gamma and di-gamma
        gamma = np.zeros((T,N))
        digamma = np.zeros((T,N,N))
    
        for t in range(T-1):
            denom = 0
            denom = np.dot(alpha[t], np.dot(A, B[:, obs_seq[t+1]]*beta[t+1]))
            digamma[t] = (alpha[t][:,None] * A * (B[:, obs_seq[t+1]] * beta[t+1])) / denom
            gamma[t] = digamma[t].sum(axis=1)

        gamma[T-1] = alpha[T-1] / alpha[T-1].sum()

        #Calculate A1
        A1 = digamma[:T-1].sum(axis=0) 

        #Calculate B1
        B1 = np.zeros((B.shape[0], B.shape[1]), dtype=B.dtype)
        for t in range(T):
             B1[:,obs_seq[t]] += gamma[t]
        
        # Calculate C
        C = np.sum(np.log10(c))

        # Update W
        A1s = A1.sum(axis=1)
        W += (a / C) * (A1 - (A1s[:, np.newaxis] * A))

        # Update V
        B1s = B1.sum(axis=1)
        V += (a/C) * (B1 - B1s[:,np.newaxis]*B)

        # Re-estimate pi
        pi = gamma[0]
        
        # Re-estimate A
        A = np.zeros((A.shape[0], A.shape[1]), dtype=A.dtype) # np.zeros_like(A)
        numer = e**(ta*W)
        denom = numer.sum(axis=1)
        denom = denom.reshape(-1, 1)
        A = numer/denom

        # Re-estimate B
        B = np.zeros((B.shape[0], B.shape[1]), dtype=B.dtype) # np.zeros_like(B)
        numer = e**(ta*V)
        denom = numer.sum(axis=1)
        denom = denom.reshape(-1, 1)
        B = numer/denom

        #Compute log(P(O|lamda))
        logProb = -np.sum(np.log10(c))
    
    return C, pi, A, B



c = np.zeros((T), dtype=np.float32)
trained_C, trained_pi, trained_A,trained_B = hmm(obs_seq,pi,A,B,c,W,V) 

print(N,M,T)
print("tau = ", ta, ", alpha = ",a)
print("prob \n", trained_C)
print("pi\n", trained_pi)
print("A \n", trained_A)
print("B \n", trained_B)
