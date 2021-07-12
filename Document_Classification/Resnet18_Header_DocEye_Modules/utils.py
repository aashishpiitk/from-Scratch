import torch
import torch.nn as nn

def accuracy(x, targets):
  probs = nn.functional.softmax(x)
  labels = torch.argmax(probs, dim=1)

  count=0
  for label, target in zip(labels, targets):
    if(label.item() == target.item()):
      count+=1
  
  return count/labels.shape[0]



