import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from dataset import get_flattened_data, class_names


def _bar(current, total, width=30):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run_fnn():
    print("=" * 55)
    print("  FNN Classifier — Fashion-MNIST")
    print("=" * 55)
    # ADDED VISUALS YAY!
    # Load data
    print("\n[1/3] Loading data...", end=" ", flush=True)
    t0 = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = get_flattened_data()
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"      Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # Use a subset for fast hyperparameter search 
    X_search, y_search = X_train[:10000], y_train[:10000]
    print(f"      Using {len(X_search):,} samples for architecture search")

    # Hyperparameter search
    architectures = [
        (128,),
        (256,),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ]

    print(f"\n[2/3] Hyperparameter search — {len(architectures)} architectures × 20 iters each")
    print("-" * 55)

    validation_accuracies = []
    best_arch = None
    best_accuracy = 0
    total = len(architectures)

    for i, arch in enumerate(architectures, 1):
        arch_str = str(arch)
        print(f"  {_bar(i - 1, total)}  training {arch_str}...", end=" ", flush=True)
        t1 = time.time()

        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="adam",
            max_iter=20,          # reduced for speed during search
            random_state=42,
            early_stopping=False,
        )
        model.fit(X_search, y_search)  # subset only

        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        validation_accuracies.append(val_accuracy)

        elapsed = time.time() - t1
        marker = " ◄ best so far" if val_accuracy > best_accuracy else ""
        print(f"val acc = {val_accuracy:.4f}  ({elapsed:.1f}s){marker}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_arch = arch

    print(f"  {_bar(total, total)}  search complete")

    # Architecture bar chart 
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

    # Final model on full data 
    print(f"\n[3/3] Retraining best architecture {best_arch} on full data (100 iters)...")
    print("      verbose=True can see loss dropping every 10 iters", flush=True)
    t2 = time.time()

    final_model = MLPClassifier(
        hidden_layer_sizes=best_arch,
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
        early_stopping=False,
        verbose=True,
    )
    final_model.fit(X_train, y_train)
    print(f"      Retraining done ({time.time() - t2:.1f}s)")

    # Test evaluation 
    print("\n  Evaluating on test set...", end=" ", flush=True)
    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("done")

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

    plt.figure()
    plt.plot(final_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("FNN Training Loss Curve")
    plt.tight_layout()
    plt.savefig("results/fnn_loss_curve.png")
    plt.close()

    # Summary
    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"  Best architecture : {best_arch}")
    print(f"  Best val accuracy : {best_accuracy:.4f}")
    print(f"  Test accuracy     : {test_accuracy:.4f}")
    print("  Plots saved to results/")
    print("=" * 55)

    return best_arch, best_accuracy, test_accuracy