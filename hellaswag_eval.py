import torch
import torch.nn.functional as F
import tiktoken
from hellaswag import render_example, iterate_examples
from model import GPT, GPTConfig   # make sure this matches your filename

# CONFIG
checkpoint_path = "/teamspace/studios/this_studio/log_model_19072.pt"   # change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

config_dict = checkpoint['config']
config = GPTConfig(**config_dict)

model = GPT(config)
state_dict = checkpoint['model']

# remove "_orig_mod." prefix if present
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        new_key = k[len("_orig_mod."):]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

# tokenizer
enc = tiktoken.get_encoding("gpt2")

print(f"Using device: {device}")

# HELLA SWAG HELPER
def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)

    losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    losses = losses.view(tokens.size(0), -1)

    shift_mask = mask[..., 1:].contiguous()
    masked_losses = losses * shift_mask

    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    return avg_loss.argmin().item()

# HELLA SWAG EVALUATION
print("\nRunning HellaSwag Evaluation...\n")

num_correct = 0
num_total = 0

for example in iterate_examples("val"):
    _, tokens, mask, label = render_example(example)

    tokens = tokens.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
            logits, _ = model(tokens)

        pred = get_most_likely_row(tokens, mask, logits)

    num_total += 1
    num_correct += int(pred == label)

    if num_total % 100 == 0:
        print(f"Processed {num_total} examples...")

accuracy = num_correct / num_total
print(f"\n Final HellaSwag Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")

# TEXT GENERATION
print("\nGenerating Sample Outputs...\n")

def generate(prompt, max_length=50, num_return_sequences=3, temperature=0.8, top_k=50):
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    tokens = tokens.repeat(num_return_sequences, 1).to(device)

    xgen = tokens
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                logits, _ = model(xgen)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)

            xgen = torch.cat((xgen, xcol), dim=1)

    outputs = []
    for i in range(num_return_sequences):
        decoded = enc.decode(xgen[i].tolist())
        outputs.append(decoded)

    return outputs

# TEST PROMPTS
prompts = [
    "Hello, I'm a language model,",
    "Artificial Intelligence is",
    "In the future, humans will",
]

for prompt in prompts:
    print(f"\n Prompt: {prompt}\n" + "-"*50)
    outputs = generate(prompt)

    for i, out in enumerate(outputs):
        print(f"\nSample {i+1}:\n{out}")