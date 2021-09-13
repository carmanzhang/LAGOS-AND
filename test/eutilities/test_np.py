import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[2, 2, 2], [8, 8, 8]])
z = np.hstack([x, y])
print(z)

x = np.array([['a', 'b'], [''], ['a']])
import collections
y = collections.Counter([n for n in np.hstack(x) if len(n) > 0])
print(y)
