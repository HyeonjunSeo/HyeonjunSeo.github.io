import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'class'
]

df = pd.read_csv(
    'iris/iris.data',
    header=None,
    names=columns
)

# Remove any empty rows
df = df.dropna()

# Assign a different color to each class
colors = {
    'Iris-setosa': 'red',
    'Iris-versicolor': 'green',
    'Iris-virginica': 'blue'
}

plt.figure(figsize=(8, 5))

for species, color in colors.items():
    subset = df[df['class'] == species]

    plt.hist(
        subset['petal_length'],
        bins=15,
        alpha=0.5,
        color=color,
        label=species
    )

plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.title('Iris Dataset: Petal Length')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig/iris_1_feature.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))

for species, color in colors.items():
    subset = df[df['class'] == species]

    plt.scatter(
        subset['petal_length'],
        subset['petal_width'],
        color=color,
        label=species,
        alpha=0.7
    )

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset: Petal Length vs Petal Width')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig/iris_2_features.png', dpi=300)
plt.close()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

for species, color in colors.items():
    subset = df[df['class'] == species]

    ax.scatter(
        subset['sepal_length'],
        subset['petal_length'],
        subset['petal_width'],
        color=color,
        label=species,
        alpha=0.7
    )

ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
ax.set_title('Iris Dataset: Three-Feature Visualization')
ax.legend()

plt.tight_layout()
plt.savefig('fig/iris_3_features.png', dpi=300)
plt.close()