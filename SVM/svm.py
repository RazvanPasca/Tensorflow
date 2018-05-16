from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.svm import SVC

sns.set()
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

xfit = np.linspace(-1,3.5)
plt.plot([0.6],[2.1],'x',color = 'red', markeredgewidth = 2, markersize = 10)

for m,b,d in [(1,0.65,0.3),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color = '#AAAAAA',
                     alpha = 0.4)

plt.show()

model = SVC(kernel = 'linear', C=1E10)
model.fit(X,y)