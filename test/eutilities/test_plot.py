# from matplotlib import pyplot as plot
#
# plot.plot([1,2,3,4], [4,5,6,7])
# plot.plot([4,1,3,2], [4,5,6,7])
# plot.show()

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the impurity-based feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


if __name__ == '__main__':
    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)

    plt.figure(12)
    plt.subplot(221)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
    # plt.xticks(rotation=45)
    plt.xticks(fontsize=5, rotation=90)
    plt.subplot(222)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    plt.xlabel('xxx', loc='right', labelpad=2)
    plt.subplot(212)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

    plt.show()

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

plt.close('all')

rc('text', usetex=True)

mu = np.linspace(0, 10, 100)
eta = mu ** 2

fig, ax = plt.subplots()
ax.plot(mu, eta, label=r'$\eta (\mu)$')
ax.set_title('Test')
ax.legend()
fig.show()
