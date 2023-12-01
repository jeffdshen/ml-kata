import torch
import numpy

from torch.utils.data import DataLoader
import torch.optim as optim

from kata2.dataset import NumpyDataset
from kata2.mlp import Mlp

def train(train_inputs, train_targets, valid_inputs, valid_targets, epochs):
    train_set = NumpyDataset(train_inputs, train_targets)
    valid_set = NumpyDataset(valid_inputs, valid_targets)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)

    model = Mlp(train_inputs.shape[-1], 256, 10)
    model.train()
    optimizer = optim.AdamW(model.parameters())

    for _ in range(epochs):
        total_loss = 0.0
        total_success = 0.0
        total_count = 0.0
        for x, y in train_loader:
            z = model(x)
            loss = model.loss(z, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            success = (torch.argmax(z.detach(), dim=-1) == y).sum()
            count = y.size(0)
            total_success += success
            total_loss += loss.item() * count
            total_count += count
        train_loss = total_loss / total_count
        train_acc = total_success / total_count
        
        with torch.no_grad():
            model.eval()
            total_loss = 0.0
            total_success = 0.0
            total_count = 0.0
            for x, y in valid_loader:
                z = model(x)
                loss = model.loss(z, y)
                success = (torch.argmax(z.detach(), dim=-1) == y).sum()
                count = y.size(0)

                total_success += success
                total_loss += loss.item() * count
                total_count += count
            valid_loss = total_loss / total_count
            valid_acc = total_success / total_count
            print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}. "
                  f"valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}")
            model.train()
    
    preds = []
    with torch.no_grad():
        model.eval()
        for x, y in valid_loader:
            z = model(x)
            preds.append(torch.argmax(z.detach().clone(), dim=-1))
    preds = torch.concat(preds)
    return preds.numpy()