import jax
import jax.numpy as jnp
import flax.linen as nn


class TowerEncoder(nn.Module):
    hidden_dims: tuple
    embed_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=False):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.embed_dim)(x)
        # L2 normalize
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        return x


class TwoTowerModelJAX(nn.Module):
    user_feature_dim: int
    item_feature_dim: int
    hidden_dims: tuple = (256, 128)
    embed_dim: int = 64
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, user_features, item_features, training=False):
        user_emb = TowerEncoder(self.hidden_dims, self.embed_dim, self.dropout_rate)(
            user_features, training=training
        )
        item_emb = TowerEncoder(self.hidden_dims, self.embed_dim, self.dropout_rate)(
            item_features, training=training
        )
        return user_emb, item_emb