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

S=[]
M=input('Enter a bound:')
M=int(M)
for n in range(1,M+1):
    print(n)
    i=0
    while n!=1:
        if n%7==0:
            n=n/7
            print(n)
        elif n%7==1:
            n=5*n+6
            print(n)
        elif n%7==2:
            n=3*n+1
            print(n)
        elif n%7==3:
            n=5*n+1
            print(n)
        elif n%7==4:
            n=3*n+1
            print(n)
        elif n%7==5:
            n=3*n+2
            print(n)
        elif n%7==6:
            n=n+1
            print(n)
        i=i+1
    S.append(i+1)
print(S)

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
plt.plot(x, y, "o")
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
plt.title("Histogram of Stopping Times")
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