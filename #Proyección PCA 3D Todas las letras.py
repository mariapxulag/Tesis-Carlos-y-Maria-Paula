#Proyección PCA 3D Todas las letras, Primer dataset
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Configuración de transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 usa 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset personalizado
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {subdir: idx for idx, subdir in enumerate(sorted(os.listdir(root_dir)))}

        for subdir, label in self.class_to_idx.items():
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(subdir_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Cargar dataset
data_dir = '/content/drive/MyDrive/Espectogramas de Mel'
full_dataset = SpectrogramDataset(root_dir=data_dir, transform=transform)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_data, test_data = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Cargar ResNet50 preentrenado
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(full_dataset.class_to_idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Configurar entrenamiento
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenar modelo
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    print(f"Época {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {correct_predictions / total_samples * 100:.2f}%")

# Extracción de características
model.eval()
features, labels = [], []
with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        outputs = model(images)
        features.append(outputs.cpu().numpy())
        labels.extend(label.numpy())

features = np.concatenate(features, axis=0)
features_scaled = StandardScaler().fit_transform(features)
features_pca = PCA(n_components=3).fit_transform(features_scaled)

# Visualización
df_pca = pd.DataFrame({'Dim1': features_pca[:, 0], 'Dim2': features_pca[:, 1], 'Dim3': features_pca[:, 2],
                        'Letra': [list(full_dataset.class_to_idx.keys())[list(full_dataset.class_to_idx.values()).index(label)] for label in labels]})
fig = px.scatter_3d(df_pca, x='Dim1', y='Dim2', z='Dim3', color='Letra', title="Proyección PCA 3D")
fig.show()