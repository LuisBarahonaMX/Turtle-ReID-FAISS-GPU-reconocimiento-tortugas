#!/usr/bin/env python
# coding: utf-8

# # Proyecto: Reconocimiento de Tortugas 🐢
# 
# Pega aquí el flujo de código que te pasé.

# # 🐢 Proyecto: Reconocimiento de Tortugas Marinas  
# 
# Esta notebook tiene como objetivo **probar y documentar** un flujo de trabajo para el reconocimiento de tortugas marinas mediante **técnicas de visión por computadora e inteligencia artificial**.  
# 
# En ella se incluyen pasos básicos para:  
# - Cargar y explorar datasets (ej. *SeaTurtleIDHeads*, *SeaTurtleID2022*).  
# - Preprocesar imágenes de tortugas (cabeza y aletas).  
# - Entrenar un modelo de clasificación con **transfer learning**.  
# - Probar detección automática usando el modelo **Shelly** de Hugging Face.  
# - Visualizar resultados y documentar hallazgos.  
# 
# El propósito es generar una **base de experimentación clara y reproducible**, combinando código, visualizaciones y explicaciones en un mismo entorno.  
# 

# In[ ]:


# =====================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =====================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

# Para usar un modelo de Hugging Face (ej. Shelly)
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForObjectDetection


# In[ ]:


# =====================================
# 2. CARGA DEL DATASET
# =====================================
# Opción A: Dataset local (ejemplo con Kaggle)
data_dir = "./SeaTurtleIDHeads"  # Carpeta donde descomprimiste el dataset
classes = os.listdir(data_dir)
print("Número de clases (tortugas únicas):", len(classes))

# Mostrar algunas imágenes de prueba
import random
sample_class = random.choice(classes)
sample_path = os.path.join(data_dir, sample_class)
sample_img = random.choice(os.listdir(sample_path))
img = Image.open(os.path.join(sample_path, sample_img))
plt.imshow(img)
plt.title(f"Ejemplo de la clase: {sample_class}")
plt.axis("off")
plt.show()


# In[ ]:


# =====================================
# 3. TRANSFORMACIONES Y DATASET
# =====================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Total de imágenes:", len(dataset))
print("Clases:", dataset.classes[:10], "...")


# In[ ]:


# =====================================
# 4. MODELO BASE (Transfer Learning)
# =====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Usamos un modelo preentrenado (ejemplo: ResNet50)
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(dataset.classes))
model = model.to(device)

print(model)


# In[ ]:


# =====================================
# 5. ENTRENAMIENTO BÁSICO
# =====================================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 2  # puedes aumentar a 10-20 para resultados reales
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Época {epoch+1}, Pérdida: {running_loss/len(dataloader):.4f}")


# In[ ]:


# =====================================
# 6. PRUEBA DE PREDICCIÓN
# =====================================
img_path = os.path.join(sample_path, sample_img)
img = Image.open(img_path)
img_t = transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    outputs = model(img_t)
    _, pred = torch.max(outputs, 1)

print("Predicción:", dataset.classes[pred.item()])

plt.imshow(img)
plt.title(f"Predicción: {dataset.classes[pred.item()]}")
plt.axis("off")
plt.show()


# In[ ]:


# =====================================
# 7. OPCIÓN: USAR EL MODELO SHELLY (Hugging Face)
# =====================================
model_id = "anda1835/Shelly"

processor = AutoImageProcessor.from_pretrained(model_id)
shelly = AutoModelForObjectDetection.from_pretrained(model_id).to(device)

# Ejemplo: detectar tortuga en una imagen
encoding = processor(images=img, return_tensors="pt").to(device)
outputs = shelly(**encoding)

print(outputs)  # coordenadas y probabilidad de detección


# In[ ]:


# =====================================
# 8. VISUALIZACIÓN DE RESULTADOS
# =====================================
# (ejemplo simplificado de bounding box con Shelly)
import torch.nn.functional as F

logits = outputs.logits
boxes = outputs.pred_boxes

probs = F.softmax(logits, dim=-1)
scores, labels = probs.max(-1)

for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score > 0.8:  # umbral de confianza
        x0, y0, x1, y1 = box.tolist()
        plt.imshow(img)
        plt.gca().add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0,
                                          fill=False, color='red', linewidth=2))
        plt.title(f"Detección: {score:.2f}")
        plt.axis("off")
        plt.show()

