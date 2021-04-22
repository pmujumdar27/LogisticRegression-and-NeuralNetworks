import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# import seaborn as sns

plt.style.use('ggplot')
# plt.rcParams["figure.figsize"] = (10, 10)

digits = load_digits()
data = digits['data']
target = digits['target']

plt.figure()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(data)
plt.scatter(proj.T[0], proj.T[1], c=target, cmap="Paired")
plt.colorbar()
plt.title("Scatterplot of different digits")
# plt.show()
plt.savefig("./plots/pca_digits.png")