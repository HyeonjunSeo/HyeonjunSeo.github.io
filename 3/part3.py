import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

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

os.makedirs('fig', exist_ok=True)

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

print("PCA Vectors:")
for i, vector in enumerate(pca.components_):
    print(f"PC{i+1}: {vector}")

print("\nExplained Variance Ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio * 100:.2f}%")

print(f"Total variance explained by PC1: "
      f"{pca.explained_variance_ratio_[0] * 100:.2f}%")

print(f"Total variance explained by PC1 and PC2: "
      f"{sum(pca.explained_variance_ratio_[:2]) * 100:.2f}%")

print(f"Total variance explained by PC1, PC2 and PC3: "
      f"{sum(pca.explained_variance_ratio_[:3]) * 100:.2f}%")

plt.figure(figsize=(8, 5))

for species, color in colors.items():
    subset = X_pca[labels == species]

    plt.scatter(
        subset[:, 0],
        np.zeros(len(subset)),
        color=color,
        label=species,
        alpha=0.7
    )

plt.xlabel('PC1')
plt.yticks([])
plt.title('Iris Dataset: Projection onto PC1')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('fig/iris_pca_1d.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))

for species, color in colors.items():
    subset = X_pca[labels == species]

    plt.scatter(
        subset[:, 0],
        subset[:, 1],
        color=color,
        label=species,
        alpha=0.7
    )

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris Dataset: Projection onto PC1 and PC2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('fig/iris_pca_2d.png', dpi=300)
plt.close()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

for species, color in colors.items():
    subset = X_pca[labels == species]

    ax.scatter(
        subset[:, 0],
        subset[:, 1],
        subset[:, 2],
        color=color,
        label=species,
        alpha=0.7
    )

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Iris Dataset: Projection onto PC1, PC2 and PC3')
ax.legend()

plt.tight_layout()

plt.savefig('fig/iris_pca_3d.png', dpi=300)
plt.close()

print("\nFigures saved in the fig folder.")

components = ['PC1', 'PC2', 'PC3', 'PC4']
variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(variance)

plt.figure(figsize=(8, 5))

plt.bar(components, variance, alpha=0.7, label='Individual variance')

plt.plot(
    components,
    cumulative_variance,
    marker='o',
    color='red',
    label='Cumulative variance'
)

for i, value in enumerate(variance):
    plt.text(i, value + 1, f'{value:.2f}%', ha='center')

plt.ylabel('Explained Variance (%)')
plt.xlabel('Principal Component')
plt.title('PCA Explained Variance - Iris Dataset')
plt.ylim(0, 110)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('fig/iris_pca_variance.png', dpi=300)
plt.close()

components = ['PC1', 'PC2', 'PC3', 'PC4']
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

plt.figure(figsize=(10, 6))

for i, vector in enumerate(pca.components_):
    plt.plot(
        features,
        vector,
        marker='o',
        label=components[i]
    )

    for j, value in enumerate(vector):
        plt.text(j, value + 0.025, f'{value:.3f}',
                 ha='center', fontsize=8)

plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel('Original Features')
plt.ylabel('PCA Vector Coefficient')
plt.title('PCA Vectors - Iris Dataset')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('fig/iris_pca_vectors.png', dpi=300)
plt.close()