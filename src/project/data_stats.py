import os

import matplotlib.pyplot as plt
from data import DisasterTweetData


def data_statistics(datadir: str = "data"):
    os.makedirs("reports/figures", exist_ok=True)

    out_report = "reports/report.md"

    train_data = DisasterTweetData(data_path=f"{datadir}/train.csv")
    val_data = DisasterTweetData(data_path=f"{datadir}/valid.csv")

    # Write dataset statistics to report
    with open(out_report, "w") as f:
        f.write("# Data Statistics Report\n\n")

        f.write("## Training Dataset\n\n")
        f.write(f"Number of samples: {len(train_data)}\n\n")

        f.write("## Validation Dataset\n\n")
        f.write(f"Number of samples: {len(val_data)}\n\n")

    # Plot training class distribution
    train_labels = train_data.data["target"].tolist()

    plt.pie(
        [train_labels.count(0), train_labels.count(1)],
        labels=["Not Disaster", "Disaster"],
        autopct="%1.1f%%",
        colors=["green", "red"],
    )
    plt.title("Training Dataset Class Distribution")
    plt.savefig("reports/figures/train_class_distribution_pie.png")
    plt.close()

    # Plot validation class distribution
    val_labels = val_data.data["target"].tolist()

    plt.pie(
        [val_labels.count(0), val_labels.count(1)],
        labels=["Not Disaster", "Disaster"],
        autopct="%1.1f%%",
        colors=["green", "red"],
    )
    plt.title("Validation Dataset Class Distribution")
    plt.savefig("reports/figures/val_class_distribution_pie.png")
    plt.close()

    # Add plots to the report
    with open(out_report, "a") as f:
        f.write("## Training Class Distribution\n\n")
        f.write("![Training class distribution]" "(figures/train_class_distribution_pie.png)\n\n")

        f.write("## Validation Class Distribution\n\n")
        f.write("![Validation class distribution]" "(figures/val_class_distribution_pie.png)\n\n")


if __name__ == "__main__":
    data_statistics("data/processed_data")
