import torch
import torch.nn as nn
import torch.nn.functional as F


class hiddenDetector(nn.Module):
    """Multi-kernel 2D-CNN probe over (layer, seq_len, hidden_dim) activations.

    Input  : (batch, layer, seq_len, hidden_dim)
    Output : (batch,) — unnormalized logits (apply sigmoid for probability).
    """

    def __init__(self, input_dim=2048, num_filters=64, layer_kernel_size=3,
                 dropout=0.6, pooling='max'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_sizes = [(layer_kernel_size, 2),
                             (layer_kernel_size, 3),
                             (layer_kernel_size, 5)]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=input_dim, out_channels=num_filters,
                      kernel_size=k, padding='same')
            for k in self.kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in self.kernel_sizes])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), 1)

        if pooling == 'max':
            self.max_pooling, self.mean_pooling = True, False
        elif pooling == 'mean':
            self.max_pooling, self.mean_pooling = False, True
        else:
            raise ValueError("pooling must be 'max' or 'mean'.")

    def forward(self, x, mask=None):
        # x:    (B, L, T, D)
        # mask: (B, T)
        x = x.permute(0, 3, 1, 2)  # -> (B, D, L, T) so input_dim feeds Conv2d in_channels
        pooled_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            features = conv(x)
            features = bn(features)
            features = self.relu(features)

            if mask is not None:
                mask_expanded = mask.view(mask.size(0), 1, 1, mask.size(1))
                if self.max_pooling:
                    features = features.masked_fill(mask_expanded == 0, float('-inf'))
                if self.mean_pooling:
                    features = features.masked_fill(mask_expanded == 0, 0.0)

            if self.max_pooling:
                pooled_w, _ = torch.max(features, dim=3)
                pooled_h, _ = torch.max(pooled_w, dim=2)
            else:
                pooled_w = torch.mean(features, dim=3)
                pooled_h = torch.mean(pooled_w, dim=2)

            pooled_outputs.append(pooled_h)

        pooled = torch.cat(pooled_outputs, dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)
        return logits


class linearProbe(nn.Module):
    """Linear probe on the last-layer last-token activation."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.name = 'LinearProbe'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        x = self.bn(x)
        return self.fc(x).squeeze(1)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


class MLP(nn.Module):
    """Small MLP on mean-pooled multi-layer activations (flattened)."""

    def __init__(self, input_dim=2048):
        super().__init__()
        self.name = 'MLPProbe'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, mask=None):
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.sigmoid(self.forward(x))


class RePE(nn.Module):
    """Representation Engineering probe: scores along a learned harmful direction.

    Direction is fit from class-conditional means: d = mean(positive) - mean(negative).
    Score for a sample is (pooled - mu_negative) . d.
    """

    def __init__(self, input_dim=2048, mode="roi_mean", eps=1e-8):
        super().__init__()
        if mode not in ("roi_mean", "roi_last_token"):
            raise ValueError("mode must be 'roi_mean' or 'roi_last_token'")
        self.name = f"RePE-{mode}"
        self.input_dim = input_dim
        self.mode = mode
        self.eps = eps

        self.register_buffer("mu_h", torch.zeros(input_dim))
        self.register_buffer("mu_b", torch.zeros(input_dim))
        self.register_buffer("direction", torch.zeros(input_dim))
        self.is_fitted = False

    def _pool_roi_mean(self, x, mask=None):
        # x: (B, L, T, D); mask: (B, T)
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(-1).to(x.dtype)
            x = (x * m).sum(dim=2) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype).unsqueeze(-1)
        else:
            x = x.mean(dim=2)
        return x.mean(dim=1)

    def _pool_roi_last_token(self, x, mask=None):
        # Robust to both left- and right-padding: weight token positions by mask
        # and take the argmax to find the last valid token.
        B, L, T, _ = x.shape
        if mask is not None:
            positions = torch.arange(T, device=mask.device)
            last_idx = (mask * positions).argmax(dim=1)
        else:
            last_idx = torch.full((B,), T - 1, device=x.device, dtype=torch.long)

        b_idx = torch.arange(B, device=x.device).unsqueeze(1)
        l_idx = torch.arange(L, device=x.device).unsqueeze(0)
        t_idx = last_idx.unsqueeze(1)
        x = x[b_idx, l_idx, t_idx, :]
        return x.mean(dim=1)

    def _pool(self, x, mask=None):
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        if x.dim() == 2:
            return x
        if x.dim() != 4:
            raise ValueError(f"Expected (B, L, T, D) or (B, D), got {tuple(x.shape)}")
        if self.mode == "roi_mean":
            return self._pool_roi_mean(x, mask=mask)
        return self._pool_roi_last_token(x, mask=mask)

    @torch.no_grad()
    def fit(self, x, y, mask=None, positive_label=1):
        """Fit the harmful direction from labeled training data.

        x : (B, L, T, D) or (B, D)
        y : (B,)
        """
        self.eval()
        pooled = self._pool(x, mask=mask)
        y = y.to(pooled.device)
        pos = (y == positive_label)
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            raise ValueError("Need both positive and negative samples to fit RePE.")

        mu_h = pooled[pos].mean(dim=0)
        mu_b = pooled[neg].mean(dim=0)
        direction = F.normalize(mu_h - mu_b, dim=0, eps=self.eps)

        self.mu_h.copy_(mu_h)
        self.mu_b.copy_(mu_b)
        self.direction.copy_(direction)
        self.is_fitted = True

    def forward(self, x, mask=None):
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before forward().")
        pooled = self._pool(x, mask=mask)
        centered = pooled - self.mu_b.unsqueeze(0)
        return torch.sum(centered * self.direction.unsqueeze(0), dim=-1)

    @torch.no_grad()
    def predict(self, x, mask=None):
        self.eval()
        return self.forward(x, mask=mask)
