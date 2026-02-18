import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm

# 1. Updated Custom Transform for CLAHE
class ApplyCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_array = np.array(img)
        clahe_img = self.clahe.apply(img_array)
        return clahe_img

# 2. Updated Data Transforms (Adding Resize to 224)
data_transform = transforms.Compose([
    ApplyCLAHE(),
    transforms.ToPILImage(),          # Convert back to PIL to use Resize
    transforms.Resize((224, 224)),    # <--- Critical Step for ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 3. Load Datasets
train_dataset = PneumoniaMNIST(split='train', transform=data_transform, download=True)
test_dataset = PneumoniaMNIST(split='test', transform=data_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # Smaller batch for ViT

# 4. Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2, in_chans=1)
model.to(device)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 6. Training Function (Same as before)
def train_model(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device).squeeze().long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

# Start Training
train_model(model, train_loader, criterion, optimizer, epochs=5)
