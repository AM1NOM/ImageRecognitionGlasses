import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import EncoderCNN, DecoderRNN


# -----------------------------
# Tiny dummy dataset
# -----------------------------
class DummyCaptionDataset(Dataset):
    def __init__(self, vocab_size=20, length=8, n=10):
        self.vocab_size = vocab_size
        self.length = length
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)  # fake image
        caption = torch.randint(1, self.vocab_size, (self.length,))
        return image, caption


# -----------------------------
# Training loop
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_size = 256
    hidden_size = 256
    vocab_size = 20

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    dataset = DummyCaptionDataset(vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    criterion = nn.CrossEntropyLoss()
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
    }, "ckpt_A.pth")

    print("Checkpoint saved as ckpt_A.pth")


if __name__ == "__main__":
    train()
