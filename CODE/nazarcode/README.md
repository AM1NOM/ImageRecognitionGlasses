## Image Captioning Mini Model

This PR adds a minimal image-captioning model using:
- Encoder: pretrained ResNet50 (final layer removed)
- Decoder: simple LSTM

### How to run training

```bash
pip install torch torchvision
python train.py
