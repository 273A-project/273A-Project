import csv
import numpy as np
from numpy import linalg
from sklearn import svm
from sklearn.metrics import confusion_matrix
from VCA import *
import matplotlib.pyplot as plt

def generate_data(num_samples=1000):
    features  = np.empty(shape=(num_samples,3),dtype=np.float64)
    transformed_targets = np.empty(shape=(num_samples,),dtype=np.float64)
    for k in range(num_samples):
        sample = np.random.uniform(size=(3,))
        features[k] = sample
        transformed_targets[k] = sample[0]**2 + sample[0]*sample[1] - sample[2]**2 + sample[1]*sample[2]
    
    labels = np.zeros(shape=(num_samples,),dtype=int)

    max_target = np.max(transformed_targets)
    min_target = np.min(transformed_targets)
    delta = (max_target - min_target) / 4
    thresholds = [min_target + delta * i for i in range(1, 4)]

    labels[np.logical_and(thresholds[0] < transformed_targets, transformed_targets <= thresholds[1])] = 1
    labels[thresholds[2] < transformed_targets] = 1

    noisy_indices = np.random.choice(num_samples, size=num_samples // 100, replace=False)
    labels[noisy_indices] = ~labels[noisy_indices]

    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    normalized_features = (features - mean) / std_dev

    return normalized_features, labels, (mean, std_dev)

def data_visualize():
    # Generate the data
    X, Y, _ = generate_data(1000)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color mapping from Y
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap='coolwarm', edgecolor='k')

    # Adding labels and title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('3D Scatter Plot of Generated Data')

    # Adding a color bar
    plt.colorbar(scatter, ax=ax, label='Labels')

    plt.show()    

def thorsten_joachims_heuristic_C(X):
    m = X.shape[0]
    (U,S,V) = linalg.svd(X,full_matrices=False)
    Z = U@np.diag(S*S)@U.T
    C = 1/np.mean(np.sqrt(S))
    return C

def evaluate_header():
    print("{:<30s} | {:<20s} | {:<20s}".format("Model","In-sample accuracy","Out-sample accuracy"))
    
def evaluate_performance(name,X_train,Y_train,X_test,Y_test,model):
    in_R2 = model.score(X_train,Y_train)
    out_R2 = model.score(X_test,Y_test)
    print("{:<30s} | {:<20f} | {:<20f}".format(name,in_R2,out_R2))
    
def main():
    # (X,Y,scale) = load_data()
    (X,Y,scale) = generate_data()

    dataset_size  = X.shape[0]
    test_size = dataset_size // 3
    training_size = dataset_size - test_size

    X_train, Y_train = X[0:training_size], Y[0:training_size]
    X_test, Y_test = X[training_size:], Y[training_size:]

    print("Total dataset size %d" % dataset_size)
    print("Training size %d" % training_size)
    print("Test size %d" % test_size)
    print("---------------------")
    
    C = thorsten_joachims_heuristic_C(X_train)
    tol = 1e-4
    random_state = 12345
    evaluate_header()

    # linear SVC
    clf = svm.LinearSVC(C=C,random_state=random_state,dual=False,tol=tol)
    clf.fit(X_train,Y_train)
    evaluate_performance("Linear SVC",X_train,Y_train,X_test,Y_test,clf)

    # VCA
    print("Computing VCA...")
    epsilon = 0.1
    vca = VCA(X_train,epsilon=epsilon)
    vca.fit(print_D=True)
    print("# of vanishing components %d, using threshold=%g" % (len(vca.f_V),epsilon))

    X_vca_train = vca.transform(X_train)
    # scale the result again
    mu = np.mean(X_vca_train,axis=0)
    sigma = np.std(X_vca_train,axis=0)
    X_vca_train = (X_vca_train-mu)/sigma

    X_vca_test = vca.transform(X_test)
    X_vca_test = (X_vca_test-mu)/sigma
    
    C = thorsten_joachims_heuristic_C(X_vca_train)
    clf = svm.LinearSVC(C=C,random_state=random_state,dual=False,tol=tol)
    clf.fit(X_vca_train,Y_train)
    evaluate_header()
    evaluate_performance("Linear SVC + VCA",X_vca_train,Y_train,X_vca_test,Y_test,clf)

    return clf, X_test, Y_test, vca, X_vca_test, Y_test

if __name__ == "__main__":
    np.random.seed(12345)
    data_visualize()
    main()