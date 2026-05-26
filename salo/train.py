"""SALO probe training utilities and runtime LLM-side prediction helper."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from . import model as M


# --------------------------------------------------------------------------- #
# Dataset / collation
# --------------------------------------------------------------------------- #


class ActivationDataset(Dataset):
    """Lazy-load `.pt` activation files written by process_data.py.

    Each `.pt` is a tuple of `end_layer - start_layer` tensors of shape
    (1, seq_len, hidden_dim), fp16. __getitem__ returns:
        {'activation': (L, T, D) tensor, 'label': int}
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        info = self.data_list[idx]
        raw = torch.load(info['path'], map_location='cpu')
        squeezed = [t.squeeze(0) for t in raw]
        return {'activation': torch.stack(squeezed, dim=0), 'label': info['label']}

    def split(self, split_idx):
        return (ActivationDataset(self.data_list[:split_idx]),
                ActivationDataset(self.data_list[split_idx:]))


def padding_collate_fn(batch):
    """Right-pad variable-length (L, T, D) tensors into a single (B, L, Tmax, D) batch.

    Returns: (padded_batch, mask, labels)
    """
    raw = [item['activation'].squeeze(0) if item['activation'].dim() == 4
           else item['activation'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)

    lengths = [t.size(1) for t in raw]
    B = len(raw)
    L = raw[0].size(0)
    D = raw[0].size(2)
    Tmax = max(lengths)

    padded = torch.zeros(B, L, Tmax, D)
    mask = torch.zeros(B, Tmax)
    for i, (seq, length) in enumerate(zip(raw, lengths)):
        padded[i, :, :length, :] = seq
        mask[i, :length] = 1

    return padded, mask, labels


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_model(input_dim=2048, num_filters=64, layer_kernel_size=3,
               dropout=0.5, pooling='max'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return M.hiddenDetector(
        input_dim=input_dim,
        num_filters=num_filters,
        layer_kernel_size=layer_kernel_size,
        dropout=dropout,
        pooling=pooling,
    ).to(device)


def load_demo_dataset(dataset_path):
    """Build an ActivationDataset from `<dataset_path>/{safe,unsafe}/*.pt`."""
    print(f'dataset path: {dataset_path}')

    safe_files = sorted(Path(f"{dataset_path}/safe").glob('*.pt'))
    unsafe_files = sorted(Path(f"{dataset_path}/unsafe").glob('*.pt'))

    all_list = (
        [{'path': f, 'label': 0} for f in safe_files] +
        [{'path': f, 'label': 1} for f in unsafe_files]
    )
    random.shuffle(all_list)

    print(f'Safe samples: {len(safe_files)}, Unsafe samples: {len(unsafe_files)}')
    return ActivationDataset(all_list)


# --------------------------------------------------------------------------- #
# Runtime: score a single (prompt, optional injection) on the live LLM
# --------------------------------------------------------------------------- #


def predict(model, prompt, llm, tokenizer, device, start_layer=10, end_layer=15):
    """Score a single sample at inference time.

    prompt: dict with keys
        'prompt'    : str       — user instruction (placed inside the user role)
        'injection' : str|None  — text appended AFTER the chat template,
                                  i.e. right after the `<assistant>` generation
                                  header. Use this for prefilling-style attacks.

    For an attack where the adversarial suffix should live INSIDE the user
    block (e.g. GCG), concatenate it into prompt['prompt'] and set
    prompt['injection']=None.
    """
    model.eval()
    with torch.no_grad():
        chat = [{"role": "user", "content": prompt['prompt']}]
        chat_str = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        if prompt['injection'] is not None:
            chat_str += prompt['injection']

        tokens = tokenizer.encode(
            chat_str, return_tensors="pt", add_special_tokens=False,
        ).to(device)

        outputs = llm(tokens, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states[start_layer:end_layer])
        hidden_states = hidden_states.permute(1, 0, 2, 3)  # (B, L, T, D)

        B, _, T, _ = hidden_states.shape
        mask = torch.ones((B, T), dtype=torch.float32, device=device)

        logits = model(hidden_states, mask)
        return torch.sigmoid(logits).squeeze().item()


# --------------------------------------------------------------------------- #
# Evaluation + training
# --------------------------------------------------------------------------- #


def eval(model, eval_dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, masks, labels in eval_dataloader:
            inputs = inputs.to(model.device)
            masks = masks.to(model.device)
            labels = labels.to(model.device)

            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            total_acc += (preds == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples, total_acc / total_samples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(num_epochs=50, input_dim=2048, lr=1e-3, batch_size=16, seed=999,
        pooling='max', dataset_path=None, num_workers=8):
    """Train a `hiddenDetector` CNN probe on serialized activations."""
    set_seed(seed)
    dataset = load_demo_dataset(dataset_path=dataset_path)
    train_dataset, eval_dataset = dataset.split(int(0.8 * len(dataset)))

    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=padding_collate_fn,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=padding_collate_fn,
        num_workers=max(1, num_workers // 2), pin_memory=True, persistent_workers=True,
    )

    model = load_model(input_dim=input_dim, pooling=pooling)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, masks, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{batch_idx+1}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}")
        scheduler.step()

        eval_loss, eval_acc = eval(model, eval_dataloader, criterion)
        max_acc = max(max_acc, eval_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Evaluation Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")

    print(f"Training complete. Max Accuracy: {max_acc:.4f}")
    return model, max_acc


# --------------------------------------------------------------------------- #
# Optional baseline probes (Linear / MLP / RePE)
# --------------------------------------------------------------------------- #


def train_probe(num_epochs=10, input_dim=4096, seed=42, batch_size=16,
                dataset_path=None, model_type='linear'):
    """Train a non-CNN baseline probe (linear or MLP) on the same activations."""
    set_seed(seed)
    dataset = load_demo_dataset(dataset_path=dataset_path)
    train_dataset, _ = dataset.split(int(0.8 * len(dataset)))
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=padding_collate_fn,
    )

    if model_type == 'linear':
        model = M.linearProbe(input_dim=input_dim)
    elif model_type == 'mlp':
        model = M.MLP(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(model.device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, masks, labels) in enumerate(data_loader):
            inputs = inputs.to(model.device)
            masks = masks.to(model.device)
            labels = labels.to(model.device)

            if model_type == 'linear':
                # Take the last valid token at the last selected layer.
                last_layer = inputs[:, -1, :, :]                  # (B, T, D)
                seq_lengths = masks.sum(dim=1).long()
                last_idx = seq_lengths - 1
                inputs = last_layer[torch.arange(inputs.size(0)), last_idx]
            else:  # mlp
                # Mean-pool over tokens, then flatten layers.
                inputs = inputs.mean(dim=2).view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{batch_idx+1}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}")
    return model


def train_repe(input_dim=4096, seed=42, batch_size=16, dataset_path=None,
               mode='roi_last_token'):
    """Fit a RePE direction (no gradient step — just class-conditional means)."""
    set_seed(seed)
    dataset = load_demo_dataset(dataset_path=dataset_path)
    train_dataset, _ = dataset.split(int(0.8 * len(dataset)))

    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=padding_collate_fn,
    )

    model = M.RePE(input_dim=input_dim, mode=mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pooled_chunks, label_chunks = [], []
    print("Extracting features for RePE...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, masks, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            # Reuse RePE's pooling — handles left/right padding correctly.
            pooled = model._pool(inputs, mask=masks)
            pooled_chunks.append(pooled.cpu())
            label_chunks.append(labels.cpu())
            if batch_idx % 10 == 0:
                print(f"Step [{batch_idx+1}/{len(data_loader)}] processed.")

    full_x = torch.cat(pooled_chunks, dim=0).to(device)
    full_y = torch.cat(label_chunks, dim=0).to(device)

    print("Fitting RePE direction...")
    model.fit(x=full_x, y=full_y, mask=None, positive_label=1)
    print("RePE fit complete.")
    return model
