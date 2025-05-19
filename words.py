import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
from scipy.sparse import dok_matrix
from sklearn import svm

# Load word list
with open('words.txt', 'r') as f:
    words = [line.strip() for line in f]
vocab_size = len(words)

# Load data
def load_data(data_file, label_file, vocab_size, num_docs):
    data = dok_matrix((num_docs, vocab_size), dtype=np.float32)
    with open(data_file, 'r') as f:
        for line in f:
            doc_id, word_id = map(int, line.strip().split())
            data[doc_id - 1, word_id - 1] = 1.0
    labels = np.loadtxt(label_file, dtype=int)
    return data.toarray(), labels

# Load training and test data
num_train_docs = sum(1 for _ in open('trainLabel.txt'))
X_train, y_train = load_data('trainData.txt', 'trainLabel.txt', vocab_size, num_train_docs)
num_test_docs = sum(1 for _ in open('testLabel.txt'))
X_test, y_test = load_data('testData.txt', 'testLabel.txt', vocab_size, num_test_docs)

nb = GaussianNB()
nb.fit(X_train, y_train)

train_acc_nb = nb.score(X_train, y_train)
test_acc_nb = nb.score(X_test, y_test)

print(f"Naive Bayes - Train Accuracy: {train_acc_nb:.2f}, Test Accuracy: {test_acc_nb:.2f}")

mean_0 = nb.theta_[0]
mean_1 = nb.theta_[1]
epsilon = 1e-10
log_diff = np.abs(np.log(mean_0 + epsilon) - np.log(mean_1 + epsilon))
top_indices = np.argsort(log_diff)[-10:][::-1]

print("\nTop 10 Discriminative Words:")
for idx in top_indices:
    print(f"{words[idx]}: {log_diff[idx]:.4f}")



pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_train)

def plot_training_data_with_decision_boundary(
    kernel, ax=None, long_title=True, support_vectors=True
):
    
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X_vis, y_train)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    common_params = {"estimator": clf, "X": X_vis, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    if long_title:
        ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")
    else:
        ax.set_title(kernel)

    if ax is None:
        plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plot_training_data_with_decision_boundary('linear', ax=axs[0])
plot_training_data_with_decision_boundary('poly', ax=axs[1])
plt.tight_layout()
plt.show()

svm_linear = SVC(kernel='linear', gamma='auto')
svm_linear.fit(X_train, y_train)
train_acc_svm_linear = accuracy_score(y_train, svm_linear.predict(X_train))
test_acc_svm_linear = accuracy_score(y_test, svm_linear.predict(X_test))

svm_poly = SVC(kernel='poly', degree=3, gamma='auto')
svm_poly.fit(X_train, y_train)
train_acc_svm_poly = accuracy_score(y_train, svm_poly.predict(X_train))
test_acc_svm_poly = accuracy_score(y_test, svm_poly.predict(X_test))

# Print SVM accuracies
print(f"SVM (Linear) - Train Accuracy: {train_acc_svm_linear:.2f}, Test Accuracy: {test_acc_svm_linear:.2f}")
print(f"SVM (Poly) - Train Accuracy: {train_acc_svm_poly:.2f}, Test Accuracy: {test_acc_svm_poly:.2f}")
