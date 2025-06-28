import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import json
import matplotlib.pyplot as plt

# Load the trained model and tokenizer
model_name = "models/bloom_classifier"  # Path to the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the test data
df_test = pd.read_csv("data/test.csv")

# Ensure consistent label mapping
label_map = {label: idx for idx, label in enumerate(sorted(df_test["bloom_label"].unique()))}
reverse_label_map = {v: k for k, v in label_map.items()}
df_test["bloom_label"] = df_test["bloom_label"].map(label_map)

# Save the mapped version
df_test.to_csv("data/test_mapped.csv", index=False)

# Load the dataset using Hugging Face Datasets
dataset = load_dataset("csv", data_files={"test": "data/test_mapped.csv"})

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["question"], padding=True, truncation=True)

# Tokenize the test set
tokenized_dataset = dataset.map(tokenize, batched=True)

# Prepare dataset for Trainer
tokenized_dataset = tokenized_dataset.rename_column("bloom_label", "labels")
tokenized_dataset = tokenized_dataset.remove_columns(["question"])
tokenized_dataset.set_format("torch")

# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# Training arguments (for evaluation)
args = TrainingArguments(
    output_dir="models/bloom_classifier",
    per_device_eval_batch_size=16,
    no_cuda=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"\n✅ Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"✅ Test F1 Score: {eval_results['eval_f1']:.4f}")

# Get prediction output
pred_output = trainer.predict(tokenized_dataset["test"])
labels = pred_output.label_ids
preds = np.argmax(pred_output.predictions, axis=1)
confidences = np.max(torch.nn.functional.softmax(torch.tensor(pred_output.predictions), dim=1).numpy(), axis=1)

# Print classification report
print("\n📊 Classification Report:\n")
print(classification_report(labels, preds, target_names=[str(label) for label in label_map.keys()]))

# Show confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=label_map.keys())
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save predictions with confidence scores
df_test["predicted_label"] = preds
df_test["predicted_label_name"] = df_test["predicted_label"].map(reverse_label_map)
df_test["confidence"] = confidences
df_test.to_csv("data/test_with_predictions.csv", index=False)
print("💾 Predictions saved to: data/test_with_predictions.csv")

# Retrieve training epoch info
try:
    with open("models/bloom_classifier/trainer_state.json", "r") as f:
        trainer_state = json.load(f)
    num_epochs_used = trainer_state.get("log_history", [{}])[-1].get("epoch", "Unknown")
    print(f"\n📌 Number of Epochs Used During Training: {num_epochs_used}")
except Exception as e:
    print(f"⚠️ Could not retrieve number of epochs: {e}")






# (venv) PS C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew - Copy> python train_model.py
# Generating test split: 96 examples [00:00, 4885.09 examples/s]
# Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 96/96 [00:00<00:00, 1719.93 examples/s]
# C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew - Copy\venv\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.40it/s]

# ✅ Test Accuracy: 1.0000
# ✅ Test F1 Score: 1.0000
# C:\Users\Soham\Documents\NLP&GEN_AI(Sem 6)\Course Project\scrutinynew - Copy\venv\Lib\site-packages\torch\utils\data\dataloader.py:665: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
#   warnings.warn(warn_msg)
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  4.91it/s]

# 📊 Classification Report:

#               precision    recall  f1-score   support

#            1       1.00      1.00      1.00        22
#            2       1.00      1.00      1.00        12
#            3       1.00      1.00      1.00        13
#            4       1.00      1.00      1.00        14
#            5       1.00      1.00      1.00        16
#            6       1.00      1.00      1.00        19

#     accuracy                           1.00        96
#    macro avg       1.00      1.00      1.00        96
# weighted avg       1.00      1.00      1.00        96

