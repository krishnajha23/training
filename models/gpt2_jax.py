import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    n_positions: int

    @nn.compact
    def __call__(self, x, training=False):
        B, T, C = x.shape
        head_dim = C // self.n_head

        qkv = nn.Dense(3 * C, use_bias=True)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # reshape to (B, n_head, T, head_dim)
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # causal mask
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        attn = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, attn)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(0.1)(attn, deterministic=not training)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=True)(out)


class MLP(nn.Module):
    n_embd: int

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(4 * self.n_embd)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_embd)(x)
        x = nn.Dropout(0.1)(x, deterministic=not training)
        return x


class Block(nn.Module):
    n_head: int
    n_embd: int
    n_positions: int

    @nn.compact
    def __call__(self, x, training=False):
        x = x + CausalSelfAttention(self.n_head, self.n_embd, self.n_positions)(
            nn.LayerNorm()(x), training=training
        )
        x = x + MLP(self.n_embd)(nn.LayerNorm()(x), training=training)
        return x


class GPT2JAX(nn.Module):
    vocab_size: int = 50257
    n_positions: int = 256
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

    @nn.compact
    def __call__(self, input_ids, training=False):
        B, T = input_ids.shape

        tok_emb = nn.Embed(self.vocab_size, self.n_embd)(input_ids)
        pos_emb = nn.Embed(self.n_positions, self.n_embd)(jnp.arange(T))
        x = nn.Dropout(0.1)(tok_emb + pos_emb, deterministic=not training)

        for _ in range(self.n_layer):
            x = Block(self.n_head, self.n_embd, self.n_positions)(x, training=training)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)
        return logits