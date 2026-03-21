import jax
import jax.numpy as jnp
import optax


def gpt2_loss_jax(logits, input_ids):
    """
    Language modeling loss for JAX GPT-2.
    Shifts logits and labels by 1 — predict token i+1 from token i.
    """
    logits = logits[:, :-1, :]   # (B, T-1, vocab)
    labels = input_ids[:, 1:]    # (B, T-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()


def in_batch_contrastive_loss_jax(user_emb, item_emb, temperature=0.07):
    """
    InfoNCE in-batch contrastive loss for two-tower training.
    Diagonal of similarity matrix = positive pairs.
    Off-diagonal = in-batch negatives.

    Args:
        user_emb: (batch, embed_dim) — L2 normalized
        item_emb: (batch, embed_dim) — L2 normalized
        temperature: scalar float
    """
    similarity = jnp.matmul(user_emb, item_emb.T) / temperature  # (B, B)
    labels = jnp.arange(user_emb.shape[0])

    loss_u = optax.softmax_cross_entropy_with_integer_labels(similarity, labels)
    loss_i = optax.softmax_cross_entropy_with_integer_labels(similarity.T, labels)

    return ((loss_u + loss_i) / 2.0).mean()