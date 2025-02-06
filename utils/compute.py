import torch
import torch.nn.functional as F
def cosine_similarity(logits1, logits2):
    ret = F.cosine_similarity(logits1, logits2, dim=1).unsqueeze(1)
    return ret
