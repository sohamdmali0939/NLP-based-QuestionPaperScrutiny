from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

label_map = {
    1: "Remember (BL1)",
    2: "Understand (BL2)",
    3: "Apply (BL3)",
    4: "Analyze (BL4)",
    5: "Evaluate (BL5)",
    6: "Create (BL6)"
}

tokenizer = AutoTokenizer.from_pretrained("models/bloom_classifier")
model = AutoModelForSequenceClassification.from_pretrained("models/bloom_classifier")

def classify_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_map[predicted_class]
