import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# LOAD DATASET
iris = load_iris()

X = iris.data
y = iris.target


# BAGI DATA TRAIN DAN TEST
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# FUNGSI HITUNG JARAK
def euclidean_distance(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


# CLASS KNN MANUAL
class KNN:

    def __init__(self, k=3):

        self.k = k

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for x in X:

            distances = []

            for x_train in self.X_train:

                distance = euclidean_distance(
                    x,
                    x_train
                )

                distances.append(distance)

            k_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = [
                self.y_train[i]
                for i in k_indices
            ]

            prediction = max(
                set(k_nearest_labels),
                key=k_nearest_labels.count
            )

            predictions.append(prediction)

        return predictions


# NILAI K YANG DITEST
k_values = [1, 3, 5, 7, 10, 15]

manual_accuracies = []
sklearn_accuracies = []

print("HASIL AKURASI")
print("========================")


for k in k_values:

    # KNN MANUAL
    model_manual = KNN(k=k)

    model_manual.fit(
        X_train,
        y_train
    )

    predictions_manual = model_manual.predict(
        X_test
    )

    acc_manual = accuracy_score(
        y_test,
        predictions_manual
    )

    manual_accuracies.append(acc_manual)

    # KNN SKLEARN
    model_sklearn = KNeighborsClassifier(
        n_neighbors=k
    )

    model_sklearn.fit(
        X_train,
        y_train
    )

    predictions_sklearn = model_sklearn.predict(
        X_test
    )

    acc_sklearn = accuracy_score(
        y_test,
        predictions_sklearn
    )

    sklearn_accuracies.append(acc_sklearn)

    print(f"K = {k}")

    print(f"Manual  : {acc_manual}")

    print(f"Sklearn : {acc_sklearn}")

    print("------------------------")


# GRAFIK
plt.plot(
    k_values,
    manual_accuracies,
    marker='o',
    label='KNN Manual'
)

plt.plot(
    k_values,
    sklearn_accuracies,
    marker='s',
    label='KNN Sklearn'
)

plt.title("Perbandingan K vs Accuracy")

plt.xlabel("Nilai K")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()