"""
HellaSwag evaluation + text generation for a trained GPT-2 checkpoint.

Usage:
  python hellaswag_eval.py --checkpoint log_model_19072.pt
  python hellaswag_eval.py --checkpoint log_model_19072.pt --device cuda
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from hellaswag import render_example, iterate_examples
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


def get_most_likely_row(tokens, mask, logits):
    """Return the index of the completion with the lowest average loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)

    losses = F.cross_entropy(flat_logits, flat_tokens, reduction="none")
    losses = losses.view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked_losses = losses * shift_mask

    avg_loss = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()


@torch.no_grad()
def evaluate_hellaswag(model, device):
    """Run HellaSwag evaluation and return accuracy."""
    device_type = "cuda" if "cuda" in device else "cpu"
    num_correct = 0
    num_total = 0

    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)

        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
        ):
            logits, _ = model(tokens)

        pred = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred == label)

        if num_total % 100 == 0:
            print(f"  Processed {num_total} examples...")

    accuracy = num_correct / num_total
    return accuracy, num_correct, num_total


def generate_samples(model, enc, prompts, device, max_length=50, num_samples=3, temperature=0.8, top_k=50):
    """Generate text samples for a list of prompts."""
    device_type = "cuda" if "cuda" in device else "cpu"
    rng = torch.Generator(device=device)
    rng.manual_seed(42)

    for prompt in prompts:
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        tokens = tokens.repeat(num_samples, 1)

        xgen = tokens
        while xgen.size(1) < max_length:
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
            ):
                logits, _ = model(xgen)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

        print(f"\n Prompt: {prompt}\n" + "-" * 50)
        for i in range(num_samples):
            print(f"\nSample {i + 1}:\n{enc.decode(xgen[i].tolist())}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on HellaSwag + generate samples")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    enc = tiktoken.get_encoding("gpt2")

    # 1) HellaSwag Evaluation
    print("\nRunning HellaSwag Evaluation...\n")
    accuracy, correct, total = evaluate_hellaswag(model, device)
    print(f"\n Final HellaSwag Accuracy: {accuracy:.4f} ({correct}/{total})")

    # 2) Text Generation
    print("\nGenerating Sample Outputs...\n")
    prompts = [
        "Hello, I'm a language model,",
        "Artificial Intelligence is",
        "In the future, humans will",
    ]
    generate_samples(model, enc, prompts, device)


if __name__ == "__main__":
    main()