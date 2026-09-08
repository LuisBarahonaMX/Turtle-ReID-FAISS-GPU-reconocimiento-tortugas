# Turtle Re-Identification with FAISS

Proyecto experimental de inteligencia artificial y visión por computadora para la reidentificación individual de tortugas marinas mediante características visuales de la cabeza.

## Objetivo

Desarrollar y evaluar un pipeline capaz de comparar fotografías de tortugas y determinar cuáles individuos presentan mayor similitud visual.

## Tecnologías

- Python
- PyTorch
- Torchvision
- Jupyter Notebook
- FAISS
- ResNet50
- CUDA / GPU
- Deep Learning
- Computer Vision
- Dataset público de Kaggle

## Flujo general

Imagen de tortuga
→ Preprocesamiento
→ Red neuronal
→ Extracción de características
→ Embedding
→ FAISS
→ Búsqueda por similitud
→ Top-K candidatos

## Etapa experimental

Este repositorio corresponde a una de las primeras fases del proyecto de reidentificación de tortugas. Se utiliza un dataset público como prueba de concepto para experimentar con redes neuronales, extracción de embeddings y búsqueda vectorial mediante FAISS.

El objetivo de esta etapa es validar técnicamente el pipeline antes de utilizar datasets propios y modelos especializados de reidentificación.

## Procesamiento

El proyecto puede ejecutarse mediante Jupyter Notebook y está diseñado para aprovechar aceleración por GPU cuando PyTorch y CUDA se encuentran correctamente configurados.

## Aplicación

La investigación busca contribuir al desarrollo de métodos no invasivos de identificación individual de tortugas marinas mediante visión por computadora, con posibles aplicaciones en monitoreo, seguimiento longitudinal e investigación para conservación.

## Estado

Proyecto experimental y en desarrollo.
