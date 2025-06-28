import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_class_distribution(df):
    plt.figure(figsize=(8,5))
    sns.countplot(x="bloom_label", data=df)
    plt.title("Bloom Label Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_loss_curves(log_history):
    train_loss = [log["loss"] for log in log_history if "loss" in log]
    val_loss = [log["eval_loss"] for log in log_history if "eval_loss" in log]

    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.show()
