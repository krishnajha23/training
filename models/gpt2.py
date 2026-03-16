from transformers import GPT2LMHeadModel, GPT2Config

def build_gpt2(n_positions=512, n_embd=768, n_layer=12, n_head=12):
    """
    GPT-2 small (124M params).
    Fits on single T4 GPU with batch_size=4 at sequence_length=512.
    """
    config = GPT2Config(
        vocab_size=50257,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT2LMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 parameters: {total_params/1e6:.1f}M")
    return model