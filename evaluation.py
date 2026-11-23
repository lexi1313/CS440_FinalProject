from tabulate import tabulate

REQUIRED_METRICS = ("precision", "recall", "micro_f1")

def main():
    # Display metrics saved in CS440_FinalProject.ipynb
    import json
    import re
    from pathlib import Path

    notebook_path = Path(__file__).with_name("CS440_FinalProject.ipynb")

    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))

    def extract_metrics(output):
        patterns = {
            "precision": r"Precision:\s*([0-9.]+)",
            "recall": r"Recall:\s*([0-9.]+)",
            "micro_f1": r"Micro F1:\s*([0-9.]+)",
            "macro_f1": r"Macro F1:\s*([0-9.]+)",
        }

        return {
            key: float(match.group(1))
            for key, pattern in patterns.items()
            if (match := re.search(pattern, output))
        }

    results = {}

    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        output = "\n".join(
            "".join(o.get("text", []))
            for o in cell.get("outputs", [])
        )

        if "rf_model.predict(X_val_tfidf)" in source:
            results["Random Forest"] = extract_metrics(output)

        elif "lr_model.predict(X_val_tfidf)" in source:
            results["Logistic Regression"] = extract_metrics(output)

        
        elif "for threshold in [0.68, 0.69, 0.70, 0.71]" in source:
            pattern = (
                r"Threshold:\s*([0-9.]+)\s+"
                r"Micro F1:\s*([0-9.]+)\s*\|\s*Macro F1:\s*([0-9.]+)\s+"
                r"Precision:\s*([0-9.]+)\s*\|\s*Recall:\s*([0-9.]+)"
            )

            for match in re.finditer(pattern, output):
                threshold, micro, macro, precision, recall = match.groups()

                results[f"MLP ({threshold})"] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "micro_f1": float(micro),
                    "macro_f1": float(macro),
                }


    print("\n=== SAVED MODEL RESULTS ===")

    for name, metrics in results.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    baseline = results.get("Random Forest")

    if baseline:
        print("\n=== BASELINE COMPARISON ===")

        for name, metrics in results.items():
            if name == "Random Forest":
                continue

            if not all(k in metrics for k in REQUIRED_METRICS):
                print(f"{name}: Missing required metrics")
                continue

            passed = all(
                metrics[k] >= baseline[k]
                for k in REQUIRED_METRICS
            )

            print(f"{name}: {'PASS' if passed else 'FAIL'}")


    print("\n=== MODEL COMPARISON TABLE ===")

    table_data = []

    for name, metrics in results.items():
        if name == "Random Forest":
            status = "Baseline"
        elif all(k in metrics for k in REQUIRED_METRICS):
            passed = all(
                metrics[k] >= baseline[k]
                for k in REQUIRED_METRICS
            )
            status = "No" if passed else "Yes"
        else:
            status = "N/A"

        table_data.append([
            name,
            f'{metrics["precision"]:.4f}',
            f'{metrics["recall"]:.4f}',
            f'{metrics["micro_f1"]:.4f}',
            f'{metrics["macro_f1"]:.4f}',
            status
        ])

    headers = [
        "Model", "Precision", "Recall",
        "Micro-F1", "Macro-F1", "Degradation?"
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
