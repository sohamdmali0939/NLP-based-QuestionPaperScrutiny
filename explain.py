from lime.lime_text import LimeTextExplainer
import numpy as np
import torch

def get_prediction_fn(model, tokenizer):
    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.detach().cpu().numpy()
    return predict_proba

def explain_question(text, model, tokenizer, label_map):
    class_names = list(label_map.keys())
    explainer = LimeTextExplainer(class_names=class_names)
    predict_fn = get_prediction_fn(model, tokenizer)

    explanation = explainer.explain_instance(text, predict_fn, num_features=8)
    explanation.show_in_notebook()
