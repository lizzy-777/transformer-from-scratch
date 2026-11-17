"""Train the existing Transformer on a synthetic, variable-length copy task.

PAD=0, BOS=1, EOS=2; data tokens start at 3. No downloads required.
"""
import argparse
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformer import Transformer

PAD, BOS, EOS = 0, 1, 2


def make_data(count, max_len, vocab_size, rng, excluded=None):
    rows, seen = [], set(excluded or ())
    while len(rows) < count:
        row = tuple(rng.randrange(3, vocab_size)
                    for _ in range(rng.randint(2, max_len)))
        if row not in seen:
            rows.append(row)
            seen.add(row)
    return rows


def collate(rows, device):
    src = torch.full((len(rows), max(map(len, rows)) + 1), PAD, dtype=torch.long)
    target = torch.full((len(rows), src.size(1) + 1), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        src[i, :len(row) + 1] = torch.tensor([*row, EOS])
        target[i, :len(row) + 2] = torch.tensor([BOS, *row, EOS])
    # Teacher forcing: predict the next token, never the current input token.
    return src.to(device), target[:, :-1].to(device), target[:, 1:].to(device)


@torch.no_grad()
def greedy_decode(model, src, max_steps):
    model.eval()
    generated = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_steps):
        logits = model(src, generated)[:, -1].clone()
        logits[:, [PAD, BOS]] = -torch.inf
        token = logits.argmax(-1)
        token = torch.where(finished, PAD, token)
        generated = torch.cat((generated, token[:, None]), dim=1)
        finished |= token.eq(EOS)
        if finished.all():
            break
    return generated[:, 1:]


@torch.no_grad()
def validate(model, rows, batch_size, device):
    model.eval()
    loss_sum = correct = tokens = exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        src, decoder_in, labels = collate(batch, device)
        logits = model(src, decoder_in)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                                    ignore_index=PAD, reduction='sum').item()
        active = labels.ne(PAD)
        tokens += active.sum().item()
        correct += ((logits.argmax(-1) == labels) & active).sum().item()
        predictions = greedy_decode(model, src, labels.size(1)).tolist()
        for row, prediction in zip(batch, predictions):
            expected = [*row, EOS]
            # Require EOS at the correct position, not just a matching prefix.
            exact += prediction[:len(expected)] == expected
    return loss_sum / tokens, correct / tokens, exact / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--train-size', type=int, default=2048)
    parser.add_argument('--val-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.train_size, args.val_size, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('sizes, epochs, threads and learning rate must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    vocab_size, max_len = 16, 6
    training = make_data(args.train_size, max_len, vocab_size, rng)
    validation = make_data(args.val_size, max_len, vocab_size, rng, training)
    model = Transformer(vocab_size, vocab_size, d_model=64, num_heads=4,
                        d_ff=128, num_layers=2, max_len=max_len + 3,
                        padding_idx=PAD).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial, _, _ = validate(model, validation, args.batch_size, device)
    print(f'Device: {device} | Initial validation loss: {initial:.4f}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(training)
        total_loss = total_tokens = 0
        for start in range(0, len(training), args.batch_size):
            src, decoder_in, labels = collate(training[start:start + args.batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, decoder_in)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = labels.ne(PAD).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_loss, accuracy, exact = validate(model, validation, args.batch_size, device)
            print(f'Epoch {epoch:3d} | Train loss: {total_loss / total_tokens:.4f} | '
                  f'Val loss: {val_loss:.4f} | Token acc: {accuracy:.1%} | '
                  f'Greedy exact: {exact:.1%}', flush=True)
    print('Held-out autoregressive examples (EOS=2):')
        # Save model weights and the configuration needed to load them.
    save_path = Path("checkpoints/step9_copy_task.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "src_vocab_size": vocab_size,
            "tgt_vocab_size": vocab_size,
            "d_model": 64,
            "num_heads": 4,
            "d_ff": 128,
            "num_layers": 2,
            "max_len": max_len + 3,
            "padding_idx": PAD,
        },
        "epochs": args.epochs,
    }, save_path)

    print(f"Model saved to: {save_path.resolve()}")
    for row in validation[:5]:
        src, _, _ = collate([row], device)
        prediction = greedy_decode(model, src, max_len + 1)[0].tolist()
        print(f'Input: {list(row)} | Expected: {[*row, EOS]} | Predicted: {prediction}')


if __name__ == '__main__':
    main()
