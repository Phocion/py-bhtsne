import bhtsne as bh
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
res = bh.bh_tsne(samples=iris['data'], perplexity=5, theta=0.15, verbose=True)
fctr = list(res)[0]
z = [y for (x,y) in fctr]
z = np.asarray(z)
z -= z.min(axis=0)
z /= z.max(axis=0)

plt.scatter(z[:,0], z[:,1])
for label, x, y in zip(iris['target'], z[:, 0], z[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle = 'round,pad=0.15', fc = 'yellow', alpha = 0.3),
                 arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()
