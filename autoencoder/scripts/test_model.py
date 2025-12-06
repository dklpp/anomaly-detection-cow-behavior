import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CONFIG = {
    'img_size': 128,
    'batch_size': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'thermal_leg_model.pth',
    
    # POINT THESE TO THE PARENT FOLDERS
    # The script expects: ./data/normal_legs/images/*.jpg
    'normal_test_path': './data/test/normal_legs', 
    'ill_test_path': './data/test/ill_legs' 
}

# --- Model Architecture ---
class RGBAutoencoder(nn.Module):
    def __init__(self):
        super(RGBAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_error_scores(model, folder_path):
    print(f"Loading images from: {folder_path}...")
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
    ])
    
    try:
        # ImageFolder looks for subfolders (e.g. 'images') inside folder_path
        dataset = datasets.ImageFolder(folder_path, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"Error loading {folder_path}: {e}")
        return []

    errors = []
    criterion = nn.MSELoss(reduction='none') 

    with torch.no_grad():
        for img, _ in loader:
            img = img.to(CONFIG['device'])
            reconstruction = model(img)
            
            # Calculate Mean Squared Error per image
            loss = criterion(reconstruction, img)
            loss_val = loss.mean().item()
            errors.append(loss_val)
            
    print(f" -> Processed {len(errors)} images.")
    return errors

if __name__ == "__main__":
    # 1. Load Model
    if not torch.cuda.is_available():
        print("Warning: CUDA not found, running on CPU (slower).")
        
    model = RGBAutoencoder().to(CONFIG['device'])
    try:
        model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'thermal_leg_model.pth' not found. Please train the model first.")
        exit()

    # 2. Get Scores for both sets
    normal_scores = get_error_scores(model, CONFIG['normal_test_path'])
    ill_scores = get_error_scores(model, CONFIG['ill_test_path'])

    if not normal_scores or not ill_scores:
        print("Failed to load one of the datasets. Exiting.")
        exit()

    # 3. Calculate Threshold (Dynamic, based on this specific Normal set)
    # We set threshold at Mean + 2 Standard Deviations of the Normal data
    threshold = np.mean(normal_scores) + 2 * np.std(normal_scores)
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Histograms
    plt.hist(normal_scores, bins=30, alpha=0.6, color='green', label='Normal Legs (Low Error)')
    plt.hist(ill_scores, bins=30, alpha=0.6, color='red', label='Ill Legs (High Error)')
    
    # Plot Threshold Line
    plt.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.5f}')
    
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count of Images')
    plt.title('Separation of Normal vs Ill Cows')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

        # 5. Accuracy Statistics
    tp = sum(i > threshold for i in ill_scores)     # Ill correctly flagged as Ill
    fn = sum(i <= threshold for i in ill_scores)    # Ill mistakenly flagged as Normal
    fp = sum(i > threshold for i in normal_scores)  # Normal mistakenly flagged as Ill
    tn = sum(i <= threshold for i in normal_scores) # Normal correctly flagged as Normal

    accuracy = (tp + tn) / (len(ill_scores) + len(normal_scores))

    # ---- F1 METRICS ----
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    print(f"Threshold set to: {threshold:.6f}")
    print(f"Accuracy:         {accuracy*100:.2f}%")
    print("-" * 30)
    print(f"True Positives (Ill caught):     {tp} / {len(ill_scores)}")
    print(f"False Negatives (Ill missed):    {fn} / {len(ill_scores)}")
    print("-" * 30)
    print(f"True Negatives (Normal clear):   {tn} / {len(normal_scores)}")
    print(f"False Positives (False alarms):  {fp} / {len(normal_scores)}")
    print("="*30)

    # Print F1 block
    print("\n" + "="*30)
    print("  PRECISION / RECALL / F1  ")
    print("="*30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    print("="*30)
