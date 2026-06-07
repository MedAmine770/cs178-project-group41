import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_flattened_data, class_names


def run_random_forest():
    X_train, y_train, X_val, y_val, X_test, y_test = get_flattened_data()

    n_estimators_values = [50, 100, 150, 200]
    validation_accuracies = []

    best_n_estimators = None
    best_accuracy = 0

    for n_estimators in n_estimators_values:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        validation_accuracies.append(val_accuracy)

        print(
            f"n_estimators = {n_estimators}, "
            f"validation accuracy = {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_n_estimators = n_estimators

    plt.figure()
    plt.plot(
        n_estimators_values,
        validation_accuracies,
        marker="o"
    )
    plt.xlabel("Number of trees")
    plt.ylabel("Validation accuracy")
    plt.title("Random Forest Validation Accuracy")
    plt.savefig("results/random_forest_trees_vs_accuracy.png")
    plt.close()

    best_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train, y_train)

    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    cm = confusion_matrix(y_test, test_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(xticks_rotation=45)
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/random_forest_confusion_matrix.png")
    plt.close()

    print()
    print("Best number of trees:", best_n_estimators)
    print("Best validation accuracy:", best_accuracy)
    print("Test accuracy:", test_accuracy)

    return best_n_estimators, best_accuracy, test_accuracy