from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn
from functools import partial

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

def pipes(s: str, topk = 3):
  result = dict()
  inputs = tokenizer(s, return_tensors="pt", padding=True)
  with torch.no_grad():
    logits = model(**inputs).logits
    print(logits)
    probabilities = nn.functional.softmax(logits, dim=-1)
  emotions = []
  probs = []
  top3 = probabilities[0].topk(topk)
  emotions.append([model.config.id2label[ind.item()] for ind in top3.indices])
  probs.append([round(prob.item(), 2) for prob in top3.values])
  result["emotions"] = emotions
  result["probabilities"] = probs
  return result
