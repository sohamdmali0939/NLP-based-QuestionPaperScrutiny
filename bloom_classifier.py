from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer once
model_path = "models/bloom_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyze",
    4: "Evaluate",
    5: "Create"
}

explanation_map = {
    "Remember": "This question involves recall of facts or basic concepts, which corresponds to 'Remember' in Bloom’s taxonomy.",
    "Understand": "This question requires explanation or interpretation, indicating the 'Understand' level.",
    "Apply": "The action-oriented verb suggests application of knowledge to solve problems, falling under 'Apply'.",
    "Analyze": "This question breaks down complex concepts, hinting at analysis, which maps to 'Analyze'.",
    "Evaluate": "The question expects a judgment or decision-making based on criteria, aligning with 'Evaluate'.",
    "Create": "This involves assembling parts to form a new whole or proposing novel ideas, matching 'Create'."
}

def classify_bloom(text, return_explanation=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][predicted].item()
    
    label = label_map[predicted]
    explanation = explanation_map.get(label, "No explanation available.")

    return (label, confidence, explanation) if return_explanation else (label, confidence)
