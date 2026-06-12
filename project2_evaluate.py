"""
File di Evaluation: confronta le performance di ResNet18 ed EfficientNet-B0 sul Test Set.
Progetto 2 - Tecniche di Programmazione Avanzata ed AI
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from project2_model import ResNetCatsDogs, EfficientNetCatsDogs

### 1. Caricamento del Test Set
TEST_DATASET_PATH = Path("test_set/test_set")

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=TEST_DATASET_PATH, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = Path(__file__).resolve().parent

# Configurazione modelli per la valutazione
modelli_da_valutare = {
    "ResNet18": (ResNetCatsDogs(pretrained=False).to(device), "resnet_cats_dogs.pth"),
    "EfficientNet": (EfficientNetCatsDogs(pretrained=False).to(device), "efficientnet_cats_dogs.pth")
}

risultati_accuracy = {}

### 2. Loop di Valutazione e Grafici
for nome_modello, (model, filename) in modelli_da_valutare.items():
    print(f"\nValutazione in corso per {nome_modello}...")
    
    # Caricamento dei pesi specifici generati dal train
    model.load_state_dict(torch.load(base_path / filename, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    risultati_accuracy[nome_modello] = accuracy
    print(f'Accuracy di {nome_modello} sul Test Set: {accuracy:.2f}%')
    
    # Visualizzazione Matrice di Confusione
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix: {nome_modello}\nAccuracy: {accuracy:.2f}%')
    plt.show()

### 3. Tabella Riassuntiva Finale per la relazione
print("\n" + "="*40)
print("TABELLA COMPARATIVA PERFORMANCE")
print("="*40)
for nome, acc in risultati_accuracy.items():
    print(f"- {nome}: {acc:.2f}% di Accuracy")
print("="*40)