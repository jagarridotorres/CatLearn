"""This tutorial is intended to show that the resulting estimates are
improved by including first derivative observations.

First we set up a known underlying function in one dimension.
Then, we pick some values to train.
Finally we will use CatLearn to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt
import gptools


# A known underlying function in one dimension (y) and first derivative (dy).
def afunc(x):
    """Function (y) and first derivative (dy)."""
    y = 100 + (x-4) * np.sin(x)
    dy = (x-4) * np.cos(x) + np.sin(x)
    return [y, dy]


# Setting up data.

# A number of training points in x.
train_points = 5

# Each element in the list train can be referred to as a fingerprint.
np.random.seed(2)
train = 2*np.random.randn(train_points, 1) + 3.0
train = np.concatenate((train, [[0.0], [7.0]]))


# Call the underlying function to produce the target values.
target = np.array(afunc(train)[0])

# Generate test datapoints x.
test_points = 500
test = np.linspace(0.0, 7.0, test_points)
test = np.reshape(test, (test_points, 1))

# Make a copy of the original features and targets.
org_train = train.copy()
org_target = target.copy()
org_test = test.copy()

# Call the underlying function to produce the gradients of the target values.

gradients = []
for i in org_train:
    gradients.append(afunc(i)[1])
org_gradients = np.asarray(gradients)
gradients = org_gradients

# Gaussian Process.

k = gptools.SquaredExponentialKernel(param_bounds=[(1e-3, 2.0), (1e-3, 4.0)])
gp = gptools.GaussianProcess(k)


gp.add_data(train, target.flatten())

for i in range(0, np.shape(gradients)[1]):
    g_i = gradients[:, i]
    n_i = np.zeros(np.shape(gradients))
    n_i[:, i] = 1.0
    gp.add_data(train, g_i, n=n_i)

# gp.sample_hyperparameter_posterior(nsamp=5)

y_star, err_y_star = gp.predict(test)



linex = np.linspace(0.0, 7.0, test_points)
linex = np.reshape(linex, (1, np.shape(linex)[0]))
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i)[0])

fig = plt.figure(figsize=(5, 5))

# Example
ax = fig.add_subplot(111)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(org_train, org_target, 'o', alpha=0.2, color='black')
ax.plot(org_test, y_star, 'g-', lw=1, alpha=0.4)
# ax.fill_between(org_test[:, 0], upper, lower, interpolate=True, color='red',
#                 alpha=0.2)
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Plot gradients (when included).

if gradients is not None:
    size_bar_gradients = (np.abs(np.max(linex) - np.min(linex))/2.0)/25.0

    def lineary(m, linearx, train, target):
            """Define some linear function."""
            lineary = m*(linearx-train)+target
            return lineary

    for i in range(0, np.shape(org_gradients)[0]):
        linearx_i = np.linspace(
            org_train[i]-size_bar_gradients,
            org_train[i]+size_bar_gradients, num=10)
        lineary_i = lineary(org_gradients[i], linearx_i, org_train[i],
                            org_target[i])
        ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=0.5, color='black')


plt.show()
