import cv2
import time
import torch
import copy
import random
import os

import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.optim as optim
import albumentations as A
from torchvision import models
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
torch.cuda.empty_cache()
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

class_dict = {'detergentes': 0,
 'carton': 1,
 'latas': 2,
 'frascos': 3,
 'botella de plastico': 4,
 'vidrio': 5}

def path_n_dict(root,train_ratio = None):
    path_im = []
    label_im = [] 
    for k in os.listdir(root):
        big_class = os.path.join(root,k)
        for i in os.listdir(big_class):
            per_class_path = os.path.join(big_class,i)
            for j in os.listdir(per_class_path):
                path_im.append(os.path.join(per_class_path,j)) 
                label_im.append(class_dict[i])
    if train_ratio :
        return train_test_split(path_im,label_im,train_size=train_ratio,random_state=42)
    return (path_im,label_im)

class MyDataset(Dataset):
    def __init__(self,path,label,transforms = None):
        self.path_image = path
        self.labels = label
        self.transforms = transforms
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        image = cv2.imread(self.path_image[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image = image)
        return image,label
    
    # Ajustes de Data Augmentation 
data_transforms = {
    'train': A.Compose([
        A.LongestMaxSize(max_size=512, interpolation=2),  # Ajusta el tamaño máximo a 512
        A.PadIfNeeded(min_height=512, min_width=512),  # Ajusta a 512x512
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.8),  # Reduce rotate_limit
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.CoarseDropout(max_holes=20, max_height=16, max_width=16, p=0.5),  # Reduce CoarseDropout
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]),
    'val': A.Compose([
        A.LongestMaxSize(max_size=512, interpolation=2),
        A.PadIfNeeded(min_height=512, min_width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
}

# Cargar datasets y DataLoader
train_path, val_path, train_label, val_label = path_n_dict('train_crops', train_ratio=0.8)
test_path, test_label = path_n_dict('test_crops')

train_dataset = MyDataset(train_path, train_label, transforms=data_transforms["train"])
val_dataset = MyDataset(val_path, val_label, transforms=data_transforms["val"])
test_dataset = MyDataset(test_path, test_label, transforms=data_transforms["val"])

# Ajusta batch_size y num_workers según tu hardware
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(16, 6))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

key_list = list(class_dict.keys())
inputs,classes = next(iter(train_loader))
out = torchvision.utils.make_grid(inputs['image'])

imshow(out, title=[key_list[x.item()] for x in classes])


# Definir el modelo tal como lo entrenaste originalmente
base_model = models.efficientnet_b5(weights='IMAGENET1K_V1')
base_model.classifier = nn.Sequential(
    nn.BatchNorm1d(2048),
    nn.Linear(2048, 512),
    nn.LeakyReLU(0.1),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.1),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.1),
    nn.Linear(128, 6)  # Salida final con 6 clases
)
base_model = nn.DataParallel(base_model)
base_model.to(device)

# Ajuste de optimizador y scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(base_model.parameters(), lr=0.001, weight_decay=0.01)  # Ajustar weight_decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)  # Ajuste más agresivo

# Implementación de Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0001):  # Incrementar paciencia y ajustar delta
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

impact_matrix = {
    # Detergentes mal clasificados
    (0, 1): 12,  # Detergentes clasificados como cartón: posible contaminación del papel reciclado con químicos.
    (0, 2): 28,  # Detergentes clasificados como latas: riesgo de contaminación química en metales reciclados.
    (0, 3): 18,  # Detergentes clasificados como frascos: residuos químicos pueden mezclarse con vidrio reciclado.
    (0, 4): 35,  # Detergentes clasificados como botella de plástico: alto riesgo de contaminación química.
    (0, 5): 30,  # Detergentes clasificados como vidrio: problemas en el reciclaje del vidrio con residuos peligrosos.

    # Cartón mal clasificado
    (1, 0): 8,   # Cartón clasificado como detergentes: menores costos pero puede dañar procesos de reciclaje.
    (1, 2): 15,  # Cartón clasificado como latas: puede causar contaminación de procesos de reciclaje de metal.
    (1, 3): 10,  # Cartón clasificado como frascos: afecta la calidad del reciclado de vidrio.
    (1, 4): 20,  # Cartón clasificado como botella de plástico: problemas en el reciclaje de plásticos.
    (1, 5): 22,  # Cartón clasificado como vidrio: contaminación cruzada con residuos orgánicos.

    # Latas mal clasificadas
    (2, 0): 35,  # Latas clasificadas como detergentes: contaminación severa por residuos metálicos y químicos.
    (2, 1): 20,  # Latas clasificadas como cartón: puede causar daños al proceso de reciclaje de papel.
    (2, 3): 25,  # Latas clasificadas como frascos: mezclas indeseables en vidrio reciclado.
    (2, 4): 30,  # Latas clasificadas como botella de plástico: contamina plásticos reciclados con residuos metálicos.
    (2, 5): 27,  # Latas clasificadas como vidrio: contamina la cadena de reciclaje del vidrio.

    # Frascos mal clasificados
    (3, 0): 22,  # Frascos clasificados como detergentes: contaminación química que afecta procesos de reciclaje.
    (3, 1): 18,  # Frascos clasificados como cartón: contaminación y menor calidad del papel reciclado.
    (3, 2): 20,  # Frascos clasificados como latas: mezclas indeseables en metales reciclados.
    (3, 4): 24,  # Frascos clasificados como botella de plástico: problemas en procesos de reciclaje de plástico.
    (3, 5): 12,  # Frascos clasificados como vidrio: impacto menor pero aún presente.

    # Botella de plástico mal clasificada
    (4, 0): 40,  # Botella de plástico clasificada como detergentes: alta contaminación química.
    (4, 1): 25,  # Botella de plástico clasificada como cartón: contamina el papel reciclado.
    (4, 2): 32,  # Botella de plástico clasificada como latas: residuos plásticos afectan la calidad de los metales.
    (4, 3): 38,  # Botella de plástico clasificada como frascos: contaminantes plásticos en vidrio.
    (4, 5): 35,  # Botella de plástico clasificada como vidrio: afecta el reciclaje del vidrio.

    # Vidrio mal clasificado
    (5, 0): 30,  # Vidrio clasificado como detergentes: puede causar daños en el reciclaje químico.
    (5, 1): 20,  # Vidrio clasificado como cartón: afecta la calidad del reciclado de papel.
    (5, 2): 25,  # Vidrio clasificado como latas: puede dañar procesos de reciclaje de metales.
    (5, 3): 18,  # Vidrio clasificado como frascos: menor impacto pero puede mezclar residuos.
    (5, 4): 22,  # Vidrio clasificado como botella de plástico: afecta la calidad del reciclado de plásticos.
}
# Ajustar Early Stopping para evitar sobreajuste
early_stopping = EarlyStopping(patience=10, delta=0.0005)  # Ajuste de patience y delta

# Modificación del ciclo de entrenamiento y validación para evaluar el impacto ambiental
def train(model, criterion, optimizer, scheduler, epochs, resume_train=False, PATH=None):
    if resume_train:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = None
    best_acc = 0.0
    best_loss = float('inf')
    t_list_loss = []
    t_list_acc = []
    v_list_loss = []
    v_list_acc = []

    # Listas para guardar las métricas por época
    epoch_precision = []
    epoch_recall = []
    epoch_f1 = []
    epoch_accuracy = []
    epoch_impact = []  # Nueva lista para almacenar el impacto ambiental total por época

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"Epoch: {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        for inputs, labels in train_loader:
            inputs = inputs['image'].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(preds == labels)

        # Almacenar etiquetas y predicciones de validación
        val_true_labels = []
        val_preds = []

        total_impact = 0  # Inicializar el impacto total para la época

        with torch.no_grad():
            model.eval()
            for inputs, labels in val_loader:
                inputs = inputs['image'].to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = outputs.argmax(dim=1)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_acc += torch.sum(preds == labels)

                val_true_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                # Calcular el impacto ambiental de los errores
                for true_label, pred_label in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                    if true_label != pred_label:
                        total_impact += impact_matrix.get((true_label, pred_label), 0)

        # Convertir listas a arrays de NumPy
        val_true_labels = np.array(val_true_labels)
        val_preds = np.array(val_preds)

        # Calcular métricas por época
        val_precision = precision_score(val_true_labels, val_preds, average='weighted')
        val_recall = recall_score(val_true_labels, val_preds, average='weighted')
        val_f1 = f1_score(val_true_labels, val_preds, average='weighted')
        val_accuracy = accuracy_score(val_true_labels, val_preds)

        # Guardar métricas por época
        epoch_precision.append(val_precision)
        epoch_recall.append(val_recall)
        epoch_f1.append(val_f1)
        epoch_accuracy.append(val_accuracy)
        epoch_impact.append(total_impact)  # Guardar el impacto total de la época

        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc.double() / len(train_dataset)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_acc.double() / len(val_dataset)

        t_list_loss.append(train_loss)
        t_list_acc.append(train_acc.item())
        v_list_loss.append(val_loss)
        v_list_acc.append(val_acc.item())

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Precision: {val_precision:.4f} Recall: {val_recall:.4f} F1 Score: {val_f1:.4f} Accuracy: {val_accuracy:.4f}')
        print(f'Impacto Ambiental Total: {total_impact}')  # Mostrar el impacto total de la época

        # Ajuste de scheduler
        scheduler.step(val_loss)

        # Implementación de Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping activated")
            break

        # Guardar mejor modelo
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = val_loss

    total_time = time.time() - start
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'Best val Epoch: {best_epoch}')
    print(f'Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    print("Model:", model)


    # Retornar las métricas junto con las listas de pérdidas, accuracies y el impacto ambiental
    return model, t_list_loss, t_list_acc, v_list_loss, v_list_acc, epoch_precision, epoch_recall, epoch_f1, epoch_accuracy, epoch_impact

# Cargar el modelo y las métricas desde el archivo guardado
checkpoint = torch.load('model_b5_1.pt')

# Cargar los pesos del modelo y el estado del optimizador
base_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Cargar las métricas y otros valores
t_list_loss = checkpoint['t_list_loss']
t_list_acc = checkpoint['t_list_acc']
v_list_loss = checkpoint['v_list_loss']
v_list_acc = checkpoint['v_list_acc']
epoch_precision = checkpoint['epoch_precision']
epoch_recall = checkpoint['epoch_recall']
epoch_f1 = checkpoint['epoch_f1']
epoch_accuracy = checkpoint['epoch_accuracy']
epoch_impact = checkpoint['epoch_impact']

def evaluate_impact(impact_value):
    """Evalúa si el impacto ambiental es bueno, moderado o malo."""
    if impact_value < 2000:
        return "Bajo (Bueno)"
    elif 2000 <= impact_value <= 6000:
        return "Moderado (Aceptable)"
    else:
        return "Alto (Necesita Mejorar)"

def test(model):
    start = time.time()
    y_ = []
    y = []
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        for inputs, labels in test_loader:
            inputs = inputs['image']
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y = np.concatenate((y, labels.cpu().detach().numpy()))
            y_ = np.concatenate((y_, preds.cpu().detach().numpy()))
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_acc += torch.sum(preds == labels)

        test_loss = test_loss / len(test_dataset)
        test_acc = test_acc.double() / len(test_dataset)

    # Calcular métricas de rendimiento
    accuracy = accuracy_score(y, y_)
    precision = precision_score(y, y_, average='weighted', zero_division=0)
    recall = recall_score(y, y_, average='weighted', zero_division=0)
    f1 = f1_score(y, y_, average='weighted', zero_division=0)

    # Calcular el Impacto Ambiental Total
    total_impact = 0
    for true, pred in zip(y, y_):
        if true != pred:  # Solo calcular el impacto si hay un error de clasificación
            total_impact += impact_matrix.get((true, pred), 0)

    impact_evaluation = evaluate_impact(total_impact)  # Evaluar el impacto ambiental total


    total_time = time.time() - start
    print(f'Testing complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    return y, y_


true_label, predict_label = test(base_model)  # Usar `base_model` en lugar de `model_trained`

# Calcular la matriz de confusión
confusion_matrix = confusion_matrix(true_label, predict_label)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

# Verificar que el modelo y las métricas se han cargado correctamente
print(f"Modelo y métricas cargadas correctamente. Última precisión: {epoch_precision[-1]:.4f}")

# Visualización de las métricas incluyendo el impacto ambiental por época
plt.figure(figsize=(14, 12))

# Gráfico de Precisión
plt.subplot(3, 2, 1)
plt.plot(epoch_precision, label='Precision', marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Precision per Epoch')
plt.grid(True)
plt.legend()

# Gráfico de Recall
plt.subplot(3, 2, 2)
plt.plot(epoch_recall, label='Recall', marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Recall per Epoch')
plt.grid(True)
plt.legend()

# Gráfico de F1 Score
plt.subplot(3, 2, 3)
plt.plot(epoch_f1, label='F1 Score', marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score per Epoch')
plt.grid(True)
plt.legend()

# Gráfico de Accuracy
plt.subplot(3, 2, 4)
plt.plot(epoch_accuracy, label='Accuracy', marker='o', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.grid(True)
plt.legend()

# Gráfico de Impacto Ambiental Total
plt.subplot(3, 2, 6)
plt.plot(epoch_impact, label='Impacto Ambiental Total', marker='o', color='purple')
plt.xlabel('Epochs')
plt.ylabel('Impacto Ambiental Total')
plt.title('Impacto Ambiental Total por Época')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()