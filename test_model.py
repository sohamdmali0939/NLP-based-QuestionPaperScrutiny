import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Setup ===

model_name = "models/bloom_classifier"  # Path to your fine-tuned model directory

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load and preprocess test data
df_test = pd.read_csv("data/test.csv")

# Create label mapping (string label -> numeric index)
label_map = {label: idx for idx, label in enumerate(sorted(df_test["bloom_label"].unique()))}
df_test["bloom_label"] = df_test["bloom_label"].map(label_map)

# Save mapped test data for Hugging Face datasets
os.makedirs("data", exist_ok=True)
df_test.to_csv("data/test_mapped.csv", index=False)

# Load dataset with Hugging Face datasets
dataset = load_dataset("csv", data_files={"test": "data/test_mapped.csv"})

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["question"], padding=True, truncation=True)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Rename label column and remove text column for Trainer compatibility
tokenized_dataset = tokenized_dataset.rename_column("bloom_label", "labels")
tokenized_dataset = tokenized_dataset.remove_columns(["question"])
tokenized_dataset.set_format("torch")

# === Evaluation Setup ===

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# Detect if GPU available, else use CPU
no_cuda = not torch.cuda.is_available()

args = TrainingArguments(
    output_dir=model_name,
    per_device_eval_batch_size=16,
    no_cuda=no_cuda,
    logging_dir=f'{model_name}/logs',
    report_to=[],  # Disable wandb or other logging if unwanted
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# === Model Evaluation ===

eval_results = trainer.evaluate()
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")

# === Additional Analysis & Visualizations ===

# Get predictions on test set
predictions_output = trainer.predict(tokenized_dataset["test"])

# Extract true and predicted labels
y_true = predictions_output.label_ids
y_pred = np.argmax(predictions_output.predictions, axis=1)

# Reverse label_map: index -> label string
idx2label = {v: k for k, v in label_map.items()}

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[idx2label[i] for i in range(len(idx2label))],
    yticklabels=[idx2label[i] for i in range(len(idx2label))]
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))]))

# Per-class accuracy line plot
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
plt.plot(
    [idx2label[i] for i in range(len(idx2label))],
    per_class_accuracy,
    marker='o',
    linestyle='-',
    color='tab:blue',
    linewidth=2,
)
plt.title('Per-class Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.savefig("per_class_accuracy.png")
plt.show()

# Boxplots of confidence scores per true class
probs = torch.nn.functional.softmax(torch.tensor(predictions_output.predictions), dim=1).numpy()

df_probs = pd.DataFrame(probs, columns=[idx2label[i] for i in range(len(idx2label))])
df_probs['true_label'] = [idx2label[i] for i in y_true]
df_probs['pred_label'] = [idx2label[i] for i in y_pred]

for class_label in idx2label.values():
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_probs, x='true_label', y=class_label)
    plt.title(f'Confidence Scores for Class "{class_label}" by True Label')
    plt.xlabel('True Label')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"confidence_scores_{class_label}.png")
    plt.show()


# python test_model.py
# Generating test split: 96 examples [00:00, 3177.88 examples/s]
# Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 1842.88 examples/s]
# C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew - Copy\venv\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  3.24it/s]
# Test Accuracy: 0.9000
# Test F1 Score: 0.8000
# PS C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew - Copy> 
