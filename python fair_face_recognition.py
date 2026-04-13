"""
Low-Bias Face Recognition - Fixed Working Version
Optimized for CPU execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# SIMPLIFIED DATASET FOR TESTING
# ============================================================================

class SimpleFaceDataset(Dataset):
    """Simple synthetic dataset for testing"""
    def __init__(self, num_samples=500, num_classes=50, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transform
        
        # Generate synthetic data (random tensors for testing)
        self.images = torch.randn(num_samples, 3, 112, 112)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.genders = torch.randint(0, 2, (num_samples,))
        self.races = torch.randint(0, 4, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': self.labels[idx],
            'gender': self.genders[idx],
            'race': self.races[idx]
        }

# ============================================================================
# ARCFACE MODEL (Simplified)
# ============================================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
    
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=128, num_classes=50):
        super(FaceRecognitionModel, self).__init__()
        
        # Simplified backbone for faster training
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        self.arcface = ArcFaceLoss(embedding_size, num_classes)
    
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.feature_extractor(features)
        
        if labels is not None:
            output = self.arcface(features, labels)
            return output, features
        else:
            return F.normalize(features)

# ============================================================================
# MMD FAIRNESS REGULARIZATION
# ============================================================================

def compute_mmd(x, y, sigma=1.0):
    """Maximum Mean Discrepancy"""
    def rbf_kernel(u, v):
        return torch.exp(-torch.cdist(u, v, p=2).pow(2) / (2 * sigma**2))
    
    n = x.size(0)
    m = y.size(0)
    
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)
    
    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    
    mmd = k_xx.sum() / (n * n)
    mmd += k_yy.sum() / (m * m)
    mmd -= 2 * k_xy.sum() / (n * m)
    
    return torch.clamp(mmd, min=0)

# ============================================================================
# TRAINING SYSTEM
# ============================================================================

class FairFaceSystem:
    def __init__(self, embedding_size=128, num_classes=50, lambda_fair=0.01):
        self.device = device
        self.model = FaceRecognitionModel(embedding_size, num_classes).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_fair = lambda_fair
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        arcface_losses = []
        fair_losses = []
        
        for batch in tqdm(dataloader, desc="Training"):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            genders = batch['gender'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, features = self.model(images, labels)
            
            # ArcFace loss
            arcface_loss = self.criterion(output, labels)
            
            # Fairness loss (MMD between gender groups)
            male_mask = genders == 0
            female_mask = genders == 1
            
            if male_mask.sum() > 0 and female_mask.sum() > 0:
                male_features = features[male_mask]
                female_features = features[female_mask]
                fair_loss = compute_mmd(male_features, female_features)
            else:
                fair_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = arcface_loss + self.lambda_fair * fair_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            arcface_losses.append(arcface_loss.item())
            fair_losses.append(fair_loss.item() if fair_loss.item() > 0 else 0)
        
        return {
            'total_loss': total_loss / len(dataloader),
            'arcface_loss': np.mean(arcface_losses),
            'fair_loss': np.mean(fair_losses)
        }
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_features = []
        all_labels = []
        all_genders = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                features = self.model(images)
                all_features.append(features.cpu())
                all_labels.append(batch['label'])
                all_genders.append(batch['gender'])
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_genders = torch.cat(all_genders, dim=0)
        
        # Calculate accuracy per group
        male_mask = all_genders == 0
        female_mask = all_genders == 1
        
        male_acc = self._calculate_accuracy(all_features[male_mask], all_labels[male_mask])
        female_acc = self._calculate_accuracy(all_features[female_mask], all_labels[female_mask])
        
        return {
            'male_accuracy': male_acc,
            'female_accuracy': female_acc,
            'accuracy_gap': abs(male_acc - female_acc),
            'overall_accuracy': (male_acc + female_acc) / 2
        }
    
    def _calculate_accuracy(self, features, labels):
        if len(features) == 0:
            return 0.0
        
        correct = 0
        n = min(len(features), 100)  # Limit for speed
        
        for i in range(n):
            distances = torch.cdist(features[i:i+1], features[:n])
            nearest = distances.argsort()[0][1]
            if nearest < len(labels[:n]) and labels[nearest] == labels[i]:
                correct += 1
        
        return (correct / n) * 100 if n > 0 else 0
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(loss_history, metrics_history):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(loss_history['total_loss'], label='Total Loss', color='blue')
    axes[0, 0].plot(loss_history['arcface_loss'], label='ArcFace Loss', color='green')
    axes[0, 0].plot(loss_history['fair_loss'], label='Fairness Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    epochs = range(1, len(metrics_history['overall_accuracy']) + 1)
    axes[0, 1].plot(epochs, metrics_history['male_accuracy'], label='Male', marker='o')
    axes[0, 1].plot(epochs, metrics_history['female_accuracy'], label='Female', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy per Group')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy gap
    axes[1, 0].plot(epochs, metrics_history['accuracy_gap'], color='red', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy Gap (%)')
    axes[1, 0].set_title('Fairness Metric - Accuracy Gap')
    axes[1, 0].grid(True)
    
    # Final comparison
    final_male = metrics_history['male_accuracy'][-1] if metrics_history['male_accuracy'] else 0
    final_female = metrics_history['female_accuracy'][-1] if metrics_history['female_accuracy'] else 0
    x = ['Male', 'Female']
    y = [final_male, final_female]
    bars = axes[1, 1].bar(x, y, color=['steelblue', 'coral'])
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Final Accuracy Comparison')
    axes[1, 1].set_ylim(0, 100)
    
    for bar, val in zip(bars, y):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    print("Results plot saved as 'training_results.png'")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("="*60)
    print("FAIR FACE RECOGNITION SYSTEM")
    print("Low-Bias Face Recognition via Synthetic Data Augmentation")
    print("="*60)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets with REDUCED BATCH SIZE
    print("\n📂 Creating datasets...")
    train_dataset = SimpleFaceDataset(num_samples=300, num_classes=30, transform=transform)  # Reduced
    val_dataset = SimpleFaceDataset(num_samples=100, num_classes=30, transform=transform)    # Reduced
    
    # FIXED: Reduced batch size from 32 to 8 for CPU memory
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Batch size: 8 (optimized for CPU)")
    
    # Initialize system
    print("\n🚀 Initializing Fair Face Recognition System...")
    fr_system = FairFaceSystem(embedding_size=64, num_classes=30, lambda_fair=0.01)  # Reduced size
    print(f"✅ Model initialized on {device}")
    
    # Training loop
    print("\n🏋️ Starting Training...")
    num_epochs = 15  # Reduced for faster testing
    loss_history = {'total_loss': [], 'arcface_loss': [], 'fair_loss': []}
    metrics_history = {'male_accuracy': [], 'female_accuracy': [], 'accuracy_gap': [], 'overall_accuracy': []}
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        losses = fr_system.train_epoch(train_loader)
        loss_history['total_loss'].append(losses['total_loss'])
        loss_history['arcface_loss'].append(losses['arcface_loss'])
        loss_history['fair_loss'].append(losses['fair_loss'])
        
        print(f"\n📊 Losses:")
        print(f"  Total Loss: {losses['total_loss']:.4f}")
        print(f"  ArcFace Loss: {losses['arcface_loss']:.4f}")
        print(f"  Fairness Loss: {losses['fair_loss']:.4f}")
        
        # Evaluate every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"\n📈 Evaluating...")
            metrics = fr_system.evaluate(val_loader)
            metrics_history['male_accuracy'].append(metrics['male_accuracy'])
            metrics_history['female_accuracy'].append(metrics['female_accuracy'])
            metrics_history['accuracy_gap'].append(metrics['accuracy_gap'])
            metrics_history['overall_accuracy'].append(metrics['overall_accuracy'])
            
            print(f"\n📈 Performance:")
            print(f"  Male Accuracy: {metrics['male_accuracy']:.2f}%")
            print(f"  Female Accuracy: {metrics['female_accuracy']:.2f}%")
            print(f"  Accuracy Gap: {metrics['accuracy_gap']:.2f}%")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    
    # Save model
    print("\n💾 Saving model...")
    fr_system.save_model('models/fair_face_model.pth')
    
    # Plot results
    print("\n📊 Generating visualizations...")
    if metrics_history['overall_accuracy']:  # Check if we have metrics
        plot_results(loss_history, metrics_history)
    else:
        print("Not enough data for visualization")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved to:")
    print(f"  - Model: models/fair_face_model.pth")
    print(f"  - Plot: training_results.png")
    
    if metrics_history['overall_accuracy']:
        print(f"\n📊 Final Performance:")
        print(f"  • Male Accuracy: {metrics_history['male_accuracy'][-1]:.2f}%")
        print(f"  • Female Accuracy: {metrics_history['female_accuracy'][-1]:.2f}%")
        print(f"  • Accuracy Gap: {metrics_history['accuracy_gap'][-1]:.2f}%")
        
        if metrics_history['accuracy_gap'][-1] < 5.0:
            print(f"\n🎉 Success! Model achieved fair performance with <5% accuracy gap")
        else:
            print(f"\n⚠️ Consider training longer or adjusting lambda_fair parameter")
    
    return fr_system, loss_history, metrics_history

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n🔧 System Information:")
    print(f"  Python Version: {os.sys.version}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Device: {device}")
    
    if device == 'cpu':
        print("  ⚠️ Running on CPU - Training will be slower")
        print("  💡 Tip: For faster training, install CUDA version of PyTorch")
    
    # Run training
    try:
        model, losses, metrics = main()
        print("\n🎉 All done! Check the generated files.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed:")
        print("   pip install torch torchvision numpy matplotlib tqdm scikit-learn")
        print("2. Check if you have enough RAM (close other applications)")
        print("3. Try reducing num_samples further in SimpleFaceDataset")