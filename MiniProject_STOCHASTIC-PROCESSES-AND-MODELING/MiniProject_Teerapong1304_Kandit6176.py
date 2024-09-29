import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import * #binom,chi2,t

# Parameters
n = 10000           # sample size
x2 = 40             # Example value for x2, replace with actual value
p = 0.31 + x2/1000  # probability of success
N = 10              # number of trials

# Generate binomial random variable sequence
X = np.random.binomial(N, p, n) # generate binomial random variable sequence
Xmin = X.min()                  # find the minimum value of X
Xmax = X.max()                  # find the maximum value of X

# Plot histogram
e = np.arange(Xmin - 0.5, Xmax + 1) # create bins
plt.hist(X, bins=e)                 # plot histogram
print(e)                            # print the bins  
print("bin = ",len(e)-1)            # print the number of bins
plt.title('Histogram with bins') 

# Calculate sample mean
Mn = np.mean(X)                                 # calculate sample mean
pn = Mn / N                                     # calculate estimated p
print(f"Sample Mean: {Mn}, Estimated p: {pn}")  # print sample mean and estimated p

# Values of k (possible outcomes, from 0 to N)
k = np.arange(0, N + 1) 

# PMF (Probability Mass Function) of the Binomial distribution
PMF = binom.pmf(k, N, pn)   # calculate the PMF
print("pmf",PMF)            # print the PMF

# Plotting the PMF
fig, ax = plt.subplots(1, 1)
ax.plot(k, PMF, 'ro', ms=5, mec='r')    # Red circles for PMF points
ax.vlines(k, 0, PMF, colors='r', lw=1)  # Vertical lines for each PMF value
# Show the plot
plt.title('PMF of Binomial Distribution (N=10, p={pn:.5f})'.format(pn=pn))
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.grid(True)
# plt.show()

#2.2_______________________________________________________
print("#2 การจำลองแบบ --------------------------------------------------------------------------------")
print("p = 0.31 + x2/1000 = 0.31 + 40/1000 = ", p)                           
print("X = ",X)
print("While Xmin = ", Xmin, ", Xmax = ", Xmax,", Mean = ", Mn, f", Estimated p = {pn:.5f}")            
print("e = ",e, "with number of bins = ", len(e)-1)  
print("K = ",k)
print("PMF = ", PMF)
plt.show()

print("#3.5 Statistic Z --------------------------------------------------------------------------------")
#3.5_______________________________________________________
H = np.bincount(X, minlength=len(PMF))      # calculate the histogram
print(f"H = {H}")                           # print the histogram
Z = np.sum((H - n * PMF) ** 2 / (n * PMF))  # calculate the statistics
print(f"Statistics Z = {Z:.5f}")            # print the statistics

print("#3.6 degrees of the freedom threshold zα -----------------------------------------------------------")
#3.6_______________________________________________________
# Degrees of freedom = Number of bins - 1
# m = len(e) - 1 = N
m = len(e) - 1                          # number of bins
r = 1                                   # number of estimated parameters
dof = ((m) - 1 - r)                     # calculate the degrees of freedom
alpha = 0.05                            # significance level
threshold = chi2.ppf(1 - alpha, dof)    # calculate the threshold
print(f"Degrees of freedom: {dof}, Chi-squared threshold: {threshold:.5f}")

# compare the statistics with the threshold
if Z < threshold:   
    print(f"While {Z:.5f} < {threshold:.5f}(Z < threshold), the candidate PMF is a good fit to the data.")
else:
    print(f"While {Z:.5f} > {threshold:.5f}(Z > threshold), the candidate PMF is not a good fit to the data.")

print("#4 Sn, yα and Confidence Interva ---------------------------------------------------------------")
#4.7-4.8___________________________________________________
# Calculate standard deviation
Sn = np.std(X, ddof=1)
# Calculate confidence interval for the mean
dof_new = n-1                                                   # degrees of freedom
y_alpha_over_2 = t.ppf(1 - alpha / 2, dof_new)                  # t-distribution value
mean_lower_bound = Mn - (y_alpha_over_2 * (Sn / np.sqrt(N)))    # lower bound
mean_upper_bound = Mn + (y_alpha_over_2 * (Sn / np.sqrt(N)))    # upper bound

print(f"y_alpha_over_2 = {y_alpha_over_2:.5f}")
print(f"Sn = {Sn:.5f}")
print(f"Confidence Interval: [{mean_lower_bound:.5f}, {mean_upper_bound:.5f}]")


