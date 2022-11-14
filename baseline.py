"""
Baseline: Implemenatin of Baselines
"""
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def KNN(train_x, train_y, test_x, test_y, K = 3):
    """
    KNN
    Parameters:
       train_x: train input
       train_y: train label
       test_x: test input
       test_y: test label
    """
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, -1)
    num_test_sample = test_x.shape[0]
    test_x = test_x.reshape(num_test_sample, -1)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(train_x, train_y)
    score = neigh.score(test_x, test_y)
    print(f"KNN: mean accuracy: {score:.4f}")


def KNN_PCA(train_x, train_y, test_x, test_y, K = 3):
    """
    KNN + PCA
    Parameters:
       train_x: train input
       train_y: train label
       test_x: test input
       test_y: test label
    """
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, -1)
    num_test_sample = test_x.shape[0]
    test_x = test_x.reshape(num_test_sample, -1)

    pca = PCA(n_components=100)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(train_x, train_y)
    score = neigh.score(test_x, test_y)
    print(f"KNN + PCA: mean accuracy: {score:.4f}")


def LR(train_x, train_y, test_x, test_y):
    """
    Logistic Regression
    Parameters:
       train_x: train input
       train_y: train label
       test_x: test input
       test_y: test label
    """
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, -1)
    num_test_sample = test_x.shape[0]
    test_x = test_x.reshape(num_test_sample, -1)
    train_y = train_y.argmax(axis=1)
    test_y = test_y.argmax(axis=1)

    pca = PCA(n_components=100)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    logistic_regression = LogisticRegression()
    logistic_regression.fit(train_x, train_y)
    score = logistic_regression.score(test_x, test_y)
    print(f"LR: mean accuracy: {score:.4f}")


def SVM(train_x, train_y, test_x, test_y):
    """
    SVM
    Parameters:
       train_x: train input
       train_y: train label
       test_x: test input
       test_y: test label
    """
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, -1)
    num_test_sample = test_x.shape[0]
    test_x = test_x.reshape(num_test_sample, -1)
    train_y = train_y.argmax(axis=1)
    test_y = test_y.argmax(axis=1)

    pca = PCA(n_components=100)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    clf = SVC(gamma='auto')
    clf.fit(train_x, train_y)
    score = clf.score(test_x, test_y)
    print(f"SVM: mean accuracy: {score:.4f}")


def MLP(train_x, train_y, test_x, test_y):
    """
    MLP
    Parameters:
       train_x: train input
       train_y: train label
       test_x: test input
       test_y: test label
    """
    num_train_sample = train_x.shape[0]
    train_x = train_x.reshape(num_train_sample, -1)
    num_test_sample = test_x.shape[0]
    test_x = test_x.reshape(num_test_sample, -1)
    train_y = train_y.argmax(axis=1)
    test_y = test_y.argmax(axis=1)

    pca = PCA(n_components=100)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=(512, 128, 32),
                        alpha=0.01)
    clf.fit(train_x, train_y)
    # score = clf.score(train_x, train_y)
    # print(score)
    score = clf.score(test_x, test_y)
    print(f"MLP: mean accuracy: {score:.4f}")


if __name__ == "__main__":
    PATH = "data_dict.pkl"
    with open(PATH, "rb") as f:
        data_dict = pickle.load(f)
    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]

    KNN(train_x, train_y, test_x, test_y)
    KNN_PCA(train_x, train_y, test_x, test_y)
    LR(train_x, train_y, test_x, test_y)
    SVM(train_x, train_y, test_x, test_y)
    MLP(train_x, train_y, test_x, test_y)
