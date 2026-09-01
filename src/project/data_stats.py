import matplotlib.pyplot as plt
from data import DisasterTweetData


def data_statistics(datadir: str = "data"):
    train_data = DisasterTweetData(data_path=f"{datadir}/train.csv")
    val_data = DisasterTweetData(data_path=f"{datadir}/valid.csv")

    print("Train dataset:")
    print(f"Number of tweets: {len(train_data)}")
    print("\n")
    print("Validation dataset:")
    print(f"Number of tweets: {len(val_data)}")

    # plot how much of each class is in the training dataset
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

    # plot how much of each class is in the validation dataset
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


if __name__ == "__main__":
    data_statistics("data/processed_data")
