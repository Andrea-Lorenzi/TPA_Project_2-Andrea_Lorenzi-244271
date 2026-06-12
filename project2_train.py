"""
File di Training: allena ResNet18 ed EfficientNet-B0 tramite Transfer Learning.
Progetto 2 - Tecniche di Programmazione Avanzata ed AI
"""
from pathlib import Path
from time import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Importiamo i modelli aggiornati
from project2_model import ResNetCatsDogs, EfficientNetCatsDogs

### 1. Caricamento e Preprocessing dei Dati
DATASET_PATH = Path("training_set/training_set")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = ImageFolder(root=DATASET_PATH, transform=train_transforms)

# --- GESTIONE DATASET ---
# Mantengo 0.5 (50%) per non sovraccaricare la GPU 
PERCENTUALE_DATI = 0.5 

num_samples = int(len(full_dataset) * PERCENTUALE_DATI)
indices = list(range(len(full_dataset)))
random.shuffle(indices)
subset_indices = indices[:num_samples]
reduced_dataset = Subset(full_dataset, subset_indices)
# -------------------------

train_size = int(0.8 * len(reduced_dataset))
val_size = len(reduced_dataset) - train_size
train_dataset, val_dataset = random_split(reduced_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Dataset pronto: {len(train_dataset)} immagini di train.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = Path(__file__).resolve().parent

# Configurazione sessioni di allenamento 
modelli_da_allenare = {
    "ResNet18": {
        "model": ResNetCatsDogs(pretrained=True).to(device),
        "filename": "resnet_cats_dogs.pth",
        "epochs": 5
    },
    "EfficientNet": {
        "model": EfficientNetCatsDogs(pretrained=True).to(device),
        "filename": "efficientnet_cats_dogs.pth",
        "epochs": 5
    }
}

### 2. Training Loop
for nome_modello, config in modelli_da_allenare.items():
    model = config["model"]
    filename = config["filename"]
    epochs = config["epochs"]
    
    print(f"\n--- Inizio Training di: {nome_modello} su {device} ---")
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time()
    
    for epoch in range(epochs):
        epoch_start_time = time()
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_time = time() - epoch_start_time
        print(f"[{nome_modello}] Epoca [{epoch+1}/{epochs}], Loss Media: {epoch_loss/len(train_loader):.4f} | Tempo: {epoch_time:.2f}s")
        
    print(f"Training di {nome_modello} completato in {time() - start_time:.2f} secondi.")
    torch.save(model.state_dict(), base_path / filename)
    print(f"Pesi salvati correttamente in: {filename}")