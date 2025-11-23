import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# --- Configuration ---
CONFIG = {
    'img_size': 128,          
    'batch_size': 16,         # Lower batch size slightly for RGB
    'learning_rate': 1e-3,
    'epochs': 100,            # More epochs needed for color
    'channels': 3,            # RGB Images
    'data_dir': './data/normal_legs', 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- 1. RGB Model Architecture ---
class RGBAutoencoder(nn.Module):
    def __init__(self):
        super(RGBAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (3, 128, 128)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 16, 16)
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Output: 3 Channels (RGB)
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Forces pixel values to 0-1 range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 2. Data Loading with Augmentation ---
def get_dataloader():
    # Training transforms: Add noise/rotation to make model robust
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),    # Flip left/right
        transforms.RandomRotation(degrees=5), # Rotate slightly (+- 5 degrees)
        transforms.ToTensor(),
    ])

    try:
        # Ensure your folder structure is: ./data/normal_legs/images/img1.jpg
        dataset = datasets.ImageFolder(root=CONFIG['data_dir'], transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        print(f"Loaded {len(dataset)} training images.")
        return dataloader
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- 3. Training Loop ---
def train_model(model, dataloader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model.to(CONFIG['device'])
    
    history = []
    print(f"Starting training on {CONFIG['device']}...")

    for epoch in range(CONFIG['epochs']):
        running_loss = 0.0
        for data in dataloader:
            img, _ = data
            img = img.to(CONFIG['device'])
            
            output = model(img)
            loss = criterion(output, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {avg_loss:.6f}")
            
    return model

# --- 4. Prediction Function ---
def predict_single_image(model, image_path):
    model.eval()
    # No augmentation for prediction, just resize
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(CONFIG['device'])
    
    with torch.no_grad():
        reconstruction = model(input_tensor)
        loss = nn.MSELoss()(reconstruction, input_tensor).item()
        
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original
    ax[0].imshow(input_tensor.cpu().squeeze().permute(1, 2, 0)) # Permute for RGB plotting
    ax[0].set_title(f"Original\nError Score: {loss:.5f}")
    ax[0].axis('off')
    
    # Reconstruction
    ax[1].imshow(reconstruction.cpu().squeeze().permute(1, 2, 0))
    ax[1].set_title("Reconstructed by AI")
    ax[1].axis('off')
    
    plt.show()
    return loss

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize
    train_loader = get_dataloader()
    
    if train_loader:
        model = RGBAutoencoder()
        
        # 2. Train
        model = train_model(model, train_loader)
        
        # 3. Save
        torch.save(model.state_dict(), "thermal_leg_model.pth")
        print("Model saved!")

        # 4. Test (Optional - select a file to test)
        # test_img_path = "./data/normal_legs/images/some_image.jpg"
        # predict_single_image(model, test_img_path)
