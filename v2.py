import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import timm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. Data Processor
class HSIDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
    def load_data(self):
        mat = scipy.io.loadmat(self.data_path, squeeze_me=True)
        hsi = mat.get('hsi')
        lidar = mat.get('lidar')
        train = mat.get('train')
        if hsi is None or lidar is None or train is None:
            raise ValueError("Missing keys in .mat file")
        hsi = (np.squeeze(hsi).astype(np.float32) - np.min(hsi)) / (np.max(hsi) - np.min(hsi) + 1e-6)
        lidar = (np.squeeze(lidar).astype(np.float32) - np.min(lidar)) / (np.max(lidar) - np.min(lidar) + 1e-6)
        lidar = np.expand_dims(lidar, -1)
        return hsi, lidar, np.squeeze(train).astype(np.int64), np.squeeze(mat.get('test')).astype(np.int64)

# 2. Patch Extractor
class PatchExtractor:
    def __init__(self, patch_size=16, stride=8):
        self.patch_size = patch_size
        self.stride = stride
    def extract(self, hsi, lidar, gt):
        H, W, _ = hsi.shape
        ph, pl, lab = [], [], []
        for i in range(0, H - self.patch_size + 1, self.stride):
            for j in range(0, W - self.patch_size + 1, self.stride):
                h = hsi[i:i+self.patch_size, j:j+self.patch_size, :]
                l = lidar[i:i+self.patch_size, j:j+self.patch_size, :]
                g = gt[i:i+self.patch_size, j:j+self.patch_size]
                if (g == 0).all(): continue
                ph.append(torch.tensor(h).permute(2, 0, 1))
                pl.append(torch.tensor(l).permute(2, 0, 1))
                lab.append(torch.tensor(g))
        return torch.stack(ph), torch.stack(pl), torch.stack(lab)

# 3. Dataset
class HSIDataset(Dataset):
    def __init__(self, hsi, lidar, label):
        self.hsi, self.lidar, self.label = hsi, lidar, label
    def __len__(self): return len(self.hsi)
    def __getitem__(self, idx):
        return self.hsi[idx], self.lidar[idx], self.label[idx]

# 4. SwinSegmenter
class SwinSegmenter(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.hsi_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True, in_chans=144)
        self.lidar_backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True, in_chans=1)
        self.fusion = nn.Conv2d(768*2, 768, 1)
        self.decoder = nn.Sequential(nn.Conv2d(768, 256, 3, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(256, num_classes, 1))

    
    def forward(self, xh, xl):
        xh = F.interpolate(xh, size=(224, 224), mode='nearest')
        xl = F.interpolate(xl, size=(224, 224), mode='nearest')
        hf = self.hsi_backbone(xh)[-1].permute(0, 3, 1, 2)
        lf = self.lidar_backbone(xl)[-1].permute(0, 3, 1, 2)
        f = self.fusion(torch.cat([hf, lf], 1))
        return self.decoder(f)

# 5. Trainer
class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for xh, xl, y in loader:
            xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(xh, xl)
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode='nearest')
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        preds, gts = [], []
        with torch.no_grad():
            for xh, xl, y in loader:
                xh, xl, y = xh.to(self.device), xl.to(self.device), y.to(self.device)
                out = self.model(xh, xl)
                if out.shape[-2:] != y.shape[-2:]:
                    out = F.interpolate(out, size=y.shape[-2:], mode='nearest')
                loss = self.criterion(out, y)
                total_loss += loss.item()
                pred = out.argmax(1)
                mask = y != 0
                correct += (pred[mask] == y[mask]).sum().item()
                total += mask.sum().item()
                preds.append(pred.cpu().numpy())
                gts.append(y.cpu().numpy())
        return total_loss / len(loader), 100. * correct / total, np.concatenate(preds), np.concatenate(gts)

# 6. Reconstruction
def reconstruct(preds, patch_size, stride, shape):
    H, W = shape
    out = np.zeros((H, W), dtype=np.int64)
    count = np.zeros((H, W), dtype=np.int64)
    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            if idx >= len(preds): continue
            out[i:i+patch_size, j:j+patch_size] += preds[idx]
            count[i:i+patch_size, j:j+patch_size] += 1
            idx += 1
    mask = count > 0
    out[mask] //= count[mask]
    return out

# 7. Visualization
def visualize(hsi, gt, pred, path=None):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(hsi[:, :, :3])
    plt.title('Original HSI')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='tab20')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='tab20')
    plt.title('Prediction')
    plt.axis('off')
    if path: plt.savefig(path)
    plt.show()

# 8. Main
if __name__ == "__main__":
    cfg = {
        'path': "/kaggle/input/houstn-hsi-ldr/houston_data.mat",
        'patch_size': 16,
        'stride': 8,
        'batch_size': 32,
        'epochs': 30,
        'classes': 16,
        'save_dir': 'output'
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = HSIDataProcessor(cfg['path'])
    hsi, lidar, train, test = processor.load_data()
    patcher = PatchExtractor(cfg['patch_size'], cfg['stride'])
    hsi_p, lidar_p, label_p = patcher.extract(hsi, lidar, train)

    dataset = HSIDataset(hsi_p, lidar_p, label_p)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'])

    model = SwinSegmenter(cfg['classes']).to(device)
    trainer = Trainer(model, device)

    best_acc = 0
    for epoch in range(cfg['epochs']):
        loss = trainer.train_epoch(train_dl)
        val_loss, val_acc, val_preds, val_gts = trainer.evaluate(val_dl)
        print(f"Epoch {epoch+1}: Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(cfg['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], f"best_model.pth"))

    print("Evaluating on test data...")
    hsi_test, lidar_test, test_labels = patcher.extract(hsi, lidar, test)
    test_ds = HSIDataset(hsi_test, lidar_test, test_labels)
    test_dl = DataLoader(test_ds, batch_size=cfg['batch_size'])
    model.load_state_dict(torch.load(os.path.join(cfg['save_dir'], f"best_model.pth")))
    test_loss, test_acc, test_preds, test_gts = trainer.evaluate(test_dl)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.2f}%")

    recon = reconstruct(test_preds, cfg['patch_size'], cfg['stride'], test.shape)
    visualize(hsi, test, recon)
