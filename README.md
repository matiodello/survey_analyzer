# Text Analyzer - Sistema de Análisis de Encuestas con BERT

Este proyecto permite analizar respuestas de encuestas abiertas utilizando modelos BERT y variantes, facilitando la clasificación automática de textos en categorías temáticas relevantes para estudios sociales y de opinión pública.

## Tabla de Contenidos
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requerimientos](#requerimientos)
- [Uso del Sistema](#uso-del-sistema)
    - [1. Analizar encuestas](#1-analizar-encuestas)
    - [2. Entrenar (fine-tuning) modelos BERT](#2-entrenar-fine-tuning-modelos-bert)
    - [3. Evaluar modelos](#3-evaluar-modelos)
- [Descripción de Archivos Principales](#descripción-de-archivos-principales)
- [Notas y Créditos](#notas-y-créditos)

---

## Estructura del Proyecto

```
text-analyzer-ordenado/
├── data/                # Datasets de encuestas y entrenamiento
├── docs/                # Documentación adicional
├── main.py              # Script principal de entrada
├── models/              # Modelos BERT entrenados y checkpoints
├── outputs/             # Resultados y logs de ejecuciones
├── requirements.txt     # Dependencias del proyecto
├── scripts/             # (Vacío/reservado para scripts adicionales)
└── src/                 # Código fuente principal
```

## Requerimientos

Instala las dependencias con:
```bash
pip install -r requirements.txt
```
Principales librerías utilizadas:
- `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `nltk`, `matplotlib`, `pyreadstat`

## Uso del Sistema

El punto de entrada es el archivo `main.py` y se ejecuta por consola con diferentes modos:

### 1. Analizar encuestas
Analiza un archivo de respuestas abiertas usando un modelo BERT entrenado.
```bash
python main.py --mode analyze --file data/encuestas/archivo.sav --model models/bert-expanded
```

### 2. Entrenar (fine-tuning) modelos BERT
Realiza el fine-tuning de BERT usando los datos preparados previamente.
```bash
python main.py --mode train
```

### 3. Evaluar modelos
Evalúa el desempeño del modelo entrenado sobre un conjunto de prueba.
```bash
python main.py --mode evaluate
```

## Descripción de Archivos Principales

- **main.py**: Script principal. Permite analizar, entrenar y evaluar modelos desde la línea de comandos.
- **src/analyze_survey_bert_expanded.py**: Analiza encuestas usando el modelo BERT expandido.
- **src/finetune_bert_expanded.py**: Realiza el fine-tuning de BERT sobre datos expandidos.
- **src/evaluate_bert.py**: Evalúa el modelo BERT entrenado.
- **src/create_training_data_from_survey.py**: Genera datasets de entrenamiento a partir de encuestas originales.
- **src/create_expanded_training_data.py**: Expande y reclasifica el dataset de entrenamiento.
- **src/create_refined_training_data.py**: Genera datasets con categorías refinadas.
- **src/finetune_bert_refined.py**: Fine-tuning de BERT usando datos refinados.
- **models/**: Carpeta donde se guardan los modelos entrenados.
- **data/**: Carpeta con los archivos de encuestas y datasets.
- **outputs/**: Resultados de análisis, métricas y logs.

## Notas y Créditos
- Proyecto desarrollado para experimentos con BERT y análisis de texto en español.
- Inspirado en prácticas de NLP y clasificación de texto con modelos preentrenados.
- Contacto: [Tu Nombre o contacto]
