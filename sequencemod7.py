# Import libraries.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity

def divisors(n):
    D = []
    P = []
    for x in range(2, n + 1):
        if n % x == 0:
            D.append(x)
    return D


def prime_factors(d):
    P = []
    for d in divisors(d):
        if len(divisors(d)) < 2:
            P.append(d)
    return P

def prime_factorization(n):
    pf={}
    for p in prime_factors(n):
        if n % p == 0:
            k = 1
            while n % (p ** k) == 0:
                k = k + 1
            pf[p]=k-1
    return pf

S=[]
T=[]
C=[]
M=input('Enter a bound:')
M=int(M)
for n in range(1,M+1):
    T.append(n)
    #print('n =', n)
    #print('prime factorization of n =', prime_factorization(n))
    i=0
    while n!=1:
        if n%7==0:
            n=n/7
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==1:
            n=5*n+6
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==2:
            n=3*n+1
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==3:
            n=5*n-1
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==4:
            n=3*n+1
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==5:
            n=3*n+1
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        elif n%7==6:
            n=n+1
            n=int(n)
            T.append(n)
            if T.count(n)>=2:
                print('Cycle at n =', n)
                C.append(n)
                break
            #print('n =', n)
            #print('prime factorization of n =', prime_factorization(n))
        i=i+1
    S.append(i+1)
    #print('Trajectory of n =', T)
    T=[]

#print(F)
print('C=', C)
if len(C)==0:
    print('There are no cycles other than the trivial cycle.')

# Print data.
# print('S=', S) #Stopping times. For large M it is not always a good idea to print out S.
print("The maximum element is", max(S))  # Maximum stopping time.
print("The mean of the stopping times is", np.mean(S))  # Mean stopping time.
print("The variance of the stopping times is", np.var(S))  # Variance.
print(
    "The standard deviation of the stopping times is", np.std(S)
)  # Standard deviation.
print("The mode of the stopping times is", stats.mode(S))  # Mode.

# x=x-axis, y=stopping times.
x = np.arange(1, M + 1)
y = S

# Create visualization.
plt.title("Stopping Times for Sequence")
plt.xlabel("n")
plt.ylabel("Stopping Time")
plt.plot(x, y, ".")
plt.show()

# Create model.
sample1 = S
sample2 = S
sample = hstack((sample1, sample2))
model = KernelDensity(bandwidth=2, kernel="gaussian")
sample = sample.reshape((len(sample), 1))
model.fit(sample)
values = asarray([value for value in np.arange(1, max(S))])
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)
plt.title("Distribution of Stopping Times")
plt.xlabel("Stopping Time")
plt.ylabel("Frequency of Stopping Time")
plt.hist(sample, bins=np.arange(0, max(S) + 1, 1), density=True)
plt.plot(values[:], probabilities)
plt.show()  # Visualize the model.

# Initialize lists for calculation.
maximum = []
mu = []
s = []
v = []

# Calculate successive maximums, means, standard deviations, and variances and append them to their respective lists.
for j in range(len(S)):
    maximum.append(max(S[: j + 1]))
    mu.append(np.mean(S[: j + 1]))
    s.append(np.std(S[: j + 1]))
    v.append(np.var(S[: j + 1]))

# Create maximum DataFrame and plot the successive maximums.
df_maximum = pd.DataFrame(maximum, columns=["maximum"])
plt.title("Sum of Maximums of Stopping Times vs. Maximum")
plt.xlabel("Sum of Maximums of Stopping Times")
plt.ylabel("Maximum")
plt.plot(maximum)
plt.show()

# Create mean DataFrame and plot the successive means.
df_mu = pd.DataFrame(mu, columns=["mu"])
plt.title("Sum of Means of Stopping Times vs. Mean")
plt.xlabel("Sum of Means of Stopping Times")
plt.ylabel("Mean")
plt.plot(mu)
plt.show()

# Create variance DataFrame and plot the successive variances.
df_v = pd.DataFrame(v, columns=["v"])
plt.title("Sum of Variances of Stopping Times vs. Variance")
plt.xlabel("Sum of Variances of Stopping Times")
plt.ylabel("Variance")
plt.plot(v)
plt.show()

# Create standard deviation DataFrame and plot successive standard deviations.
df_s = pd.DataFrame(s, columns=["s"])
plt.title("Sum of Standard Deviations of Stopping Times vs. Standard Deviation")
plt.xlabel("Sum of Standard Deviations of Stopping Times")
plt.ylabel("Standard Deviation")
plt.plot(s)
plt.show()

# Observation: The sum of the maximums, means, variances, and stand deviations are increasing functions.

# Create DataFrame that encompasses the previous data, print it to the console, and save it to a file.
df = pd.DataFrame({"S": S, "mu": mu, "v": v, "s": s}, index=np.arange(1, len(S) + 1, 1))
print(df)
df.to_csv("SequenceMod7.csv")
