import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap

X = []
labels = []

with open('iris/iris.data', 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if len(row) == 5:
            X.append([float(value) for value in row[:4]])
            labels.append(row[4])

X = np.array(X)
labels = np.array(labels)

colors = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'green',
    'Iris-virginica': 'blue'
}

isomap = Isomap(n_neighbors=3, n_components=2)
X_isomap = isomap.fit_transform(X)

plt.figure(figsize=(8, 6))

for species, color in colors.items():
    subset = X_isomap[labels == species]

    plt.scatter(
        subset[:, 0],
        subset[:, 1],
        color=color,
        label=species,
        alpha=0.7
    )

plt.xlabel('Isomap Component 1')
plt.ylabel('Isomap Component 2')
plt.title('Iris Dataset: Isomap (3 Nearest Neighbors)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('fig/iris_isomap_2d.png', dpi=300)
plt.close()

print("Isomap reconstruction error:", isomap.reconstruction_error())
print("Isomap figure saved in fig folder.")