#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning de BERT con dataset expandido (incluye subcategorías de OTROS)
Autor: Sistema de Análisis BERT
"""

import os
import pandas as pd
import numpy as np
import torch
import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bert_expanded_finetuning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BERTExpandedFineTuner:
    """
    Fine-tuner de BERT para dataset expandido con subcategorías
    """
    
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-uncased"):
        """Inicializar el fine-tuner"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Verificar disponibilidad de GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Configurar directorio de salida
        self.output_dir = "models/bert-expanded"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_datasets(self):
        """Cargar los datasets de entrenamiento expandido"""
        try:
            logger.info("Cargando datasets expandidos...")
            
            # Cargar splits
            train_df = pd.read_csv("data/expanded_training/train.csv")
            val_df = pd.read_csv("data/expanded_training/validation.csv")
            test_df = pd.read_csv("data/expanded_training/test.csv")
            
            logger.info(f"Train: {len(train_df)} muestras")
            logger.info(f"Validación: {len(val_df)} muestras")
            logger.info(f"Test: {len(test_df)} muestras")
            
            # Cargar label encoder
            label_encoder_path = "data/expanded_training/label_encoder.pkl"
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                # Crear label encoder si no existe
                all_categories = sorted(list(set(
                    list(train_df['category'].unique()) +
                    list(val_df['category'].unique()) +
                    list(test_df['category'].unique())
                )))
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(all_categories)
            
            logger.info(f"Categorías: {list(self.label_encoder.classes_)}")
            
            # Codificar etiquetas
            train_df['labels'] = self.label_encoder.transform(train_df['category'])
            val_df['labels'] = self.label_encoder.transform(val_df['category'])
            test_df['labels'] = self.label_encoder.transform(test_df['category'])
            
            # Crear datasets de Hugging Face
            self.train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
            self.val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
            self.test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando datasets: {str(e)}")
            return False
    
    def initialize_model(self):
        """Inicializar tokenizer y modelo"""
        try:
            logger.info("Inicializando modelo y tokenizer...")
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Cargar modelo con configuración para clasificación
            num_labels = len(self.label_encoder.classes_)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
            
            logger.info(f"Modelo inicializado con {num_labels} etiquetas")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {str(e)}")
            return False
    
    def tokenize_dataset(self, examples):
        """Tokenizar los ejemplos"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
    
    def prepare_datasets(self):
        """Preparar datasets tokenizados"""
        try:
            logger.info("Tokenizando datasets...")
            
            # Tokenizar datasets
            self.train_dataset = self.train_dataset.map(
                self.tokenize_dataset,
                batched=True,
                remove_columns=['text']
            )
            
            self.val_dataset = self.val_dataset.map(
                self.tokenize_dataset,
                batched=True,
                remove_columns=['text']
            )
            
            self.test_dataset = self.test_dataset.map(
                self.tokenize_dataset,
                batched=True,
                remove_columns=['text']
            )
            
            # Configurar formato para PyTorch
            self.train_dataset.set_format("torch")
            self.val_dataset.set_format("torch")
            self.test_dataset.set_format("torch")
            
            logger.info("Datasets tokenizados correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparando datasets: {str(e)}")
            return False
    
    def compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Métricas básicas
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune(self):
        """Ejecutar fine-tuning del modelo"""
        try:
            logger.info("Iniciando fine-tuning...")
            
            # Configurar argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=500,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),  # Usar mixed precision si hay GPU
                dataloader_num_workers=2,
                remove_unused_columns=False,
                report_to=None  # Desactivar wandb
            )
            
            # Configurar data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Crear trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
            
            # Ejecutar entrenamiento
            logger.info("Comenzando entrenamiento...")
            trainer.train()
            
            # Guardar modelo final
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            logger.info(f"Modelo guardado en: {self.output_dir}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Error durante fine-tuning: {str(e)}")
            return None
    
    def evaluate_model(self, trainer):
        """Evaluar el modelo en el conjunto de test"""
        try:
            logger.info("Evaluando modelo en conjunto de test...")
            
            # Predecir en test set
            predictions = trainer.predict(self.test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Calcular métricas
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Reporte detallado por clase
            target_names = self.label_encoder.classes_
            class_report = classification_report(
                y_true, y_pred, 
                target_names=target_names,
                output_dict=True
            )
            
            # Matriz de confusión
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            
            # Guardar resultados
            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'categories': target_names.tolist()
            }
            
            # Guardar métricas
            with open(os.path.join(self.output_dir, "evaluation_results.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Guardar label encoder
            with open(os.path.join(self.output_dir, "label_encoder.pkl"), 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Guardar label encoder como JSON también
            label_dict = {str(i): category for i, category in enumerate(target_names)}
            with open(os.path.join(self.output_dir, "label_encoder.json"), 'w') as f:
                json.dump(label_dict, f, indent=2)
            
            # Generar reporte de texto
            self.generate_evaluation_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {str(e)}")
            return None
    
    def generate_evaluation_report(self, results):
        """Generar reporte detallado de evaluación"""
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== REPORTE DE EVALUACIÓN - BERT EXPANDIDO ===\n\n")
            f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo: {self.model_name}\n")
            f.write(f"Número de categorías: {len(results['categories'])}\n\n")
            
            # Métricas generales
            f.write("=== MÉTRICAS GENERALES ===\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n\n")
            
            # Categorías incluidas
            f.write("=== CATEGORÍAS INCLUIDAS ===\n")
            for i, category in enumerate(results['categories']):
                f.write(f"{i}: {category}\n")
            f.write("\n")
            
            # Métricas por clase
            f.write("=== MÉTRICAS POR CATEGORÍA ===\n")
            class_report = results['classification_report']
            for category in results['categories']:
                if category in class_report:
                    metrics = class_report[category]
                    f.write(f"{category}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
        
        logger.info(f"Reporte de evaluación guardado en: {report_path}")

def main():
    """Función principal"""
    try:
        # Crear fine-tuner
        fine_tuner = BERTExpandedFineTuner()
        
        # Cargar datasets
        if not fine_tuner.load_datasets():
            return 1
        
        # Inicializar modelo
        if not fine_tuner.initialize_model():
            return 1
        
        # Preparar datasets
        if not fine_tuner.prepare_datasets():
            return 1
        
        # Ejecutar fine-tuning
        trainer = fine_tuner.fine_tune()
        if trainer is None:
            return 1
        
        # Evaluar modelo
        results = fine_tuner.evaluate_model(trainer)
        if results is None:
            return 1
        
        logger.info("Fine-tuning de BERT expandido completado exitosamente")
        logger.info(f"Resultados finales - Accuracy: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en fine-tuning expandido: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
