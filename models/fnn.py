import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_flattened_data, class_names


def run_fnn():
    X_train, y_train, X_val, y_val, X_test, y_test = get_flattened_data()

    # hidden layer sizes to try
    # each tuple defines the architecture neurons per layer
    architectures = [
        (128,),
        (256,),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ]

    validation_accuracies = []
    best_arch = None
    best_accuracy = 0
    best_model = None

    for arch in architectures:
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="adam",
            max_iter=30,          # kept low so tuning is fast; adjust later since a final model uses more
            random_state=42,
            early_stopping=False,
        )
        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        validation_accuracies.append(val_accuracy)

        arch_str = str(arch)
        print(f"Architecture {arch_str:20s}, validation accuracy = {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_arch = arch
            best_model = model

    # validation accuracy vs architecture plot
    arch_labels = [str(a) for a in architectures]
    plt.figure(figsize=(8, 4))
    plt.bar(arch_labels, validation_accuracies)
    plt.xlabel("Hidden layer architecture")
    plt.ylabel("Validation accuracy")
    plt.title("FNN Validation Accuracy by Architecture")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("results/fnn_arch_vs_accuracy.png")
    plt.close()

    # retrain best architecture with more iterations on full training set 
    final_model = MLPClassifier(
        hidden_layer_sizes=best_arch,
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
        early_stopping=False,
    )
    final_model.fit(X_train, y_train)

    # test evaluation
    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    cm = confusion_matrix(y_test, test_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    display.plot(xticks_rotation=45)
    plt.title("FNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/fnn_confusion_matrix.png")
    plt.close()

    # training loss curve
    plt.figure()
    plt.plot(final_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("FNN Training Loss Curve")
    plt.tight_layout()
    plt.savefig("results/fnn_loss_curve.png")
    plt.close()

    print()
    print("Best architecture :", best_arch)
    print("Best val accuracy :", best_accuracy)
    print("Test accuracy     :", test_accuracy)

    return best_arch, best_accuracy, test_accuracy