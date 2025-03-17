#Trabajando con las 20 componentes de PCA en KNN
#Curva de pérdida de entrenamiento
#Matrices de Confusión

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
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

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
num_epochs = 10  # Número de épocas

# Entrenar modelo
train_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Época {epoch + 1}/{num_epochs}, Pérdida: {epoch_loss:.4f}")

# Graficar la pérdida
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Curva de Pérdida del Entrenamiento')
plt.grid()
plt.show()

# Evaluación de la CNN y extracción de características
model.eval()
features, labels, y_true, y_pred = [], [], [], []
with torch.no_grad():
    for images, label in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        features.append(outputs.cpu().numpy())
        labels.extend(label.numpy())
        y_true.extend(label.numpy())
        y_pred.extend(predicted.cpu().numpy())

features = np.concatenate(features, axis=0)

# Mostrar matriz de confusión de la CNN
cm_cnn = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta real')
plt.title('Matriz de Confusión - CNN')
plt.show()

# Normalización y reducción de dimensiones
features_scaled = StandardScaler().fit_transform(features)
features_pca = PCA(n_components=3).fit_transform(features_scaled)  # Reducimos a 3D para graficar

# Evaluación de métricas de separación de clases
silhouette = silhouette_score(features_scaled, labels)
db_index = davies_bouldin_score(features_scaled, labels)
print(f"Coeficiente de Silueta: {silhouette:.4f}")
print(f"Índice de Davies-Bouldin: {db_index:.4f}")

# Visualización en 3D con PCA
df_pca = pd.DataFrame(features_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['label'] = labels
fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color=df_pca['label'].astype(str),
                    title='Proyección PCA 3D de Características', labels={'label': 'Clases'})
fig.show()

# Evaluación con KNN usando PCA
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Precisión del clasificador KNN (con PCA): {knn_accuracy * 100:.2f}%")

# Mostrar matriz de confusión de KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta real')
plt.title('Matriz de Confusión - KNN con PCA')
plt.show()

# Función para mostrar la matriz de confusión con porcentajes
def plot_confusion_matrix(cm, title):
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convertir a porcentaje
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues')  # Mostrar valores en formato de porcentaje
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta real')
    plt.title(title)
    plt.show()

# Matriz de confusión de la CNN en porcentaje
plot_confusion_matrix(cm_cnn, 'Matriz de Confusión - CNN (Porcentajes)')

# Matriz de confusión de KNN en porcentaje
plot_confusion_matrix(cm_knn, 'Matriz de Confusión - KNN con PCA (Porcentajes)')
