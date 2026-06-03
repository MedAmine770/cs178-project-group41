import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_flattened_data, class_names


def run_knn():
    X_train, y_train, X_val, y_val, X_test, y_test = get_flattened_data()

    X_train_small = X_train[:5000]
    y_train_small = y_train[:5000]

    X_val_small = X_val[:1000]
    y_val_small = y_val[:1000]

    k_values = [1, 3, 5, 7, 9]
    validation_accuracies = []

    best_k = None
    best_accuracy = 0

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_small, y_train_small)

        val_predictions = model.predict(X_val_small)
        val_accuracy = accuracy_score(y_val_small, val_predictions)

        validation_accuracies.append(val_accuracy)

        print(f"k = {k}, validation accuracy = {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_k = k

    plt.figure()
    plt.plot(k_values, validation_accuracies, marker="o")
    plt.xlabel("k value")
    plt.ylabel("Validation accuracy")
    plt.title("kNN Validation Accuracy")
    plt.savefig("results/knn_k_vs_accuracy.png")
    plt.close()

    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train_small, y_train_small)

    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    cm = confusion_matrix(y_test, test_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(xticks_rotation=45)
    plt.title("kNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/knn_confusion_matrix.png")
    plt.close()

    print()
    print("Best k:", best_k)
    print("Best validation accuracy:", best_accuracy)
    print("Test accuracy:", test_accuracy)

    return best_k, best_accuracy, test_accuracy