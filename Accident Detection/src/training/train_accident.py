import argparse
from pathlib import Path
import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.models.cnn_lstm import CNNLSTM

class VideoSequenceDataset(Dataset):
    """Samples a fixed number of frames from each labeled video clip."""
    def __init__(self, csv_path, sequence_length=16, image_size=224):
        self.df = pd.read_csv(csv_path, comment="#").reset_index(drop=True)
        required = {"clip_path", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"{csv_path} needs columns: {sorted(required)}")
        self.sequence_length = sequence_length
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = str(row["clip_path"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            cap.release()
            raise RuntimeError(f"No frames found in video: {path}")

        positions = torch.linspace(0, frame_count - 1, self.sequence_length).long().tolist()
        frames = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed reading frame {position} from {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
        cap.release()
        return torch.stack(frames), torch.tensor(int(row["label"]), dtype=torch.long)

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for clips, labels in tqdm(loader, leave=False):
            clips, labels = clips.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)

def main():
    parser = argparse.ArgumentParser(description="Train a CNN-LSTM on labeled video clips.")
    parser.add_argument("--train-csv", default="dataset/splits/train.csv")
    parser.add_argument("--val-csv", default="dataset/splits/val.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    train_ds = VideoSequenceDataset(args.train_csv, args.sequence_length, args.image_size)
    val_ds = VideoSequenceDataset(args.val_csv, args.sequence_length, args.image_size)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation CSVs must not be empty.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = CNNLSTM(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    output_dir = Path("models/accident_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f}, acc={train_acc:.3f}; "
              f"val loss={val_loss:.4f}, acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "sequence_length": args.sequence_length,
                "image_size": args.image_size,
                "class_names": ["normal", "accident"],
                "val_accuracy": best_val_acc,
            }, output_dir / "best.pt")
    print(f"Best model saved to {output_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
