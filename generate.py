"""
Standalone text generation script for GPT-2.

Usage:
  python generate.py --checkpoint log_model_19072.pt --prompt "Hello, I'm a language model,"
  python generate.py --checkpoint log_model_19072.pt --prompt "AI will" --temperature 1.0 --top_k 40
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, GPTConfig


def load_model(checkpoint_path, device):
    """Load a trained GPT model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = GPTConfig(**checkpoint["config"])
    model = GPT(config)

    # Handle compiled-model key prefix
    state_dict = {}
    for k, v in checkpoint["model"].items():
        key = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        state_dict[key] = v

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, enc, prompt, max_length=100, num_samples=3, temperature=0.8, top_k=50, device="cpu"):
    """Generate text continuations from a prompt."""
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    tokens = tokens.repeat(num_samples, 1)

    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    while tokens.size(1) < max_length:
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        ):
            logits, _ = model(tokens)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=rng)
        next_token = torch.gather(topk_indices, -1, ix)

        tokens = torch.cat((tokens, next_token), dim=1)

    return [enc.decode(tokens[i].tolist()) for i in range(num_samples)]


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained GPT-2 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt",     type=str, default="Hello, I'm a language model,")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--num_samples",type=int, default=3)
    parser.add_argument("--temperature",type=float, default=0.8)
    parser.add_argument("--top_k",      type=int, default=50)
    parser.add_argument("--device",     type=str, default=None, help="Device (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")

    print(f"\n🧾 Prompt: {args.prompt}")
    print("-" * 60)

    outputs = generate(
        model, enc, args.prompt,
        max_length=args.max_length,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    for i, text in enumerate(outputs):
        print(f"\nSample {i + 1}:\n{text}")


if __name__ == "__main__":
    main()
