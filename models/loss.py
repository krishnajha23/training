import torch
import torch.nn.functional as F

def gpt2_loss(outputs):
    """Language modeling loss — returned directly by HuggingFace GPT-2."""
    return outputs.loss

def in_batch_contrastive_loss(user_emb, item_emb, temperature):
    """
    InfoNCE in-batch contrastive loss for two-tower training.
    Diagonal of similarity matrix = positive pairs.
    Off-diagonal = in-batch negatives.
    """
    batch_size = user_emb.size(0)
    similarity = torch.matmul(user_emb, item_emb.T) / temperature
    labels     = torch.arange(batch_size, device=user_emb.device)
    loss_u     = F.cross_entropy(similarity,   labels)
    loss_i     = F.cross_entropy(similarity.T, labels)
    return (loss_u + loss_i) / 2.0