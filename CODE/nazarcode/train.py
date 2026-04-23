import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import EncoderCNN, DecoderRNN


# -----------------------------
# Vocabulary + simple tokenizer
# -----------------------------
class SimpleVocab:
    def __init__(self, class_names):
        # tokens: <pad>=0, <start>=1, <end>=2
        self.pad = 0
        self.start = 1
        self.end = 2

        # map each class name to a token
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2}
        idx = 3
        for name in class_names:
            self.word2idx[name] = idx
            idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode_caption(self, class_name):
        # caption format: <start> class_name <end>
        return torch.tensor([
            self.start,
            self.word2idx[class_name],
            self.end
        ])


# -----------------------------
# CIFAR‑10 Caption Dataset
# -----------------------------
class CIFARCaptionDataset(Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        self.class_names = self.data.classes
        self.vocab = SimpleVocab(self.class_names)

    def __len__(self):
        return 200  # tiny subset for quick training

    def __getitem__(self, idx):
        image, label = self.data[idx]
        class_name = self.class_names[label]
        caption = self.vocab.encode_caption(class_name)
        return image, caption


# -----------------------------
# Collate function (pad captions)
# -----------------------------
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    max_len = max(len(c) for c in captions)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)

    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded


# -----------------------------
# Training loop
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CIFARCaptionDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    embed_size = 256
    hidden_size = 256
    vocab_size = dataset.vocab.vocab_size

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.pad)
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for epoch in range(2):
        for images, captions in loader:
            images, captions = images.to(device), captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])

            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "vocab": dataset.vocab.word2idx
    }, "ckpt_A.pth")

    print("Checkpoint saved as ckpt_A.pth")


if __name__ == "__main__":
    train()
