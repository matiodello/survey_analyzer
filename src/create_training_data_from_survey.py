#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para crear datos de entrenamiento a partir de los clusters identificados
en la encuesta PROBLEMA_AB_C para fine-tuning de modelos BERT.

Este script implementa las recomendaciones del análisis de mejoras de fine-tuning.
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import logging
import sys
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("create_training_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SurveyDataProcessor:
    """Procesador de datos de encuesta para crear datasets de entrenamiento"""
    
    def __init__(self, survey_file_path):
        self.survey_file_path = survey_file_path
        self.cluster_labels = {
            0: "INSEGURIDAD",
            1: "TRABAJO", 
            2: "ECONOMIA",
            3: "GENERAL_MIXTO",
            4: "CORRUPCION",
            5: "INFLACION"
        }
        
        # Palabras clave por cluster (basado en el análisis previo)
        self.cluster_keywords = {
            0: ["inseguridad", "delincuencia", "crimen", "robo", "violencia"],
            1: ["trabajo", "empleo", "desempleo", "falta de trabajo", "laboral"],
            2: ["economia", "economico", "crisis economica", "situacion economica"],
            3: ["los", "la", "el", "pobreza", "educacion", "salud"],
            4: ["corrupcion", "politicos", "gobierno", "corruptos"],
            5: ["inflacion", "precios", "carestia", "costo de vida"]
        }
    
    def load_survey_data(self):
        """Cargar datos de la encuesta"""
        logger.info(f"Cargando datos de encuesta desde: {self.survey_file_path}")
        
        try:
            df, meta = pyreadstat.read_sav(self.survey_file_path)
            logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocesar texto para análisis"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convertir a string y limpiar
        text = str(text).lower().strip()
        
        # Normalizar caracteres especiales
        text = re.sub(r'[áàäâ]', 'a', text)
        text = re.sub(r'[éèëê]', 'e', text)
        text = re.sub(r'[íìïî]', 'i', text)
        text = re.sub(r'[óòöô]', 'o', text)
        text = re.sub(r'[úùüû]', 'u', text)
        text = re.sub(r'ñ', 'n', text)
        
        # Limpiar caracteres especiales pero mantener espacios
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def assign_labels_by_keywords(self, responses):
        """Asignar etiquetas basadas en palabras clave"""
        labels = []
        
        for response in responses:
            processed_response = self.preprocess_text(response)
            scores = {}
            
            # Calcular score para cada cluster
            for cluster_id, keywords in self.cluster_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in processed_response:
                        # Dar más peso a coincidencias exactas
                        if processed_response == keyword:
                            score += 3
                        elif processed_response.startswith(keyword) or processed_response.endswith(keyword):
                            score += 2
                        else:
                            score += 1
                scores[cluster_id] = score
            
            # Asignar al cluster con mayor score
            if max(scores.values()) > 0:
                best_cluster = max(scores, key=scores.get)
                labels.append(best_cluster)
            else:
                # Si no hay coincidencias claras, asignar a GENERAL_MIXTO
                labels.append(3)
        
        return labels
    
    def create_training_dataset(self, min_samples_per_class=50, max_samples_per_class=2000):
        """Crear dataset de entrenamiento balanceado"""
        logger.info("Creando dataset de entrenamiento...")
        
        # Cargar datos
        df = self.load_survey_data()
        
        # Extraer respuestas de PROBLEMA_AB_C
        responses = df['PROBLEMA_AB_C'].dropna()
        logger.info(f"Total de respuestas válidas: {len(responses)}")
        
        # Preprocesar respuestas
        processed_responses = [self.preprocess_text(resp) for resp in responses]
        
        # Filtrar respuestas muy cortas o vacías
        valid_responses = []
        for resp in processed_responses:
            if len(resp) >= 3 and resp.strip():  # Al menos 3 caracteres
                valid_responses.append(resp)
        
        logger.info(f"Respuestas válidas después del filtrado: {len(valid_responses)}")
        
        # Asignar etiquetas
        labels = self.assign_labels_by_keywords(valid_responses)
        
        # Crear DataFrame
        training_df = pd.DataFrame({
            'text': valid_responses,
            'label': labels,
            'category': [self.cluster_labels[label] for label in labels]
        })
        
        # Balancear dataset
        balanced_df = self.balance_dataset(training_df, min_samples_per_class, max_samples_per_class)
        
        return balanced_df
    
    def balance_dataset(self, df, min_samples, max_samples):
        """Balancear dataset por categorías"""
        logger.info("Balanceando dataset...")
        
        # Contar muestras por categoría
        category_counts = df['category'].value_counts()
        logger.info("Distribución original:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} muestras")
        
        balanced_dfs = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            if len(category_df) < min_samples:
                logger.warning(f"Categoría {category} tiene pocas muestras ({len(category_df)}), usando todas")
                balanced_dfs.append(category_df)
            elif len(category_df) > max_samples:
                # Submuestrear
                sampled_df = category_df.sample(n=max_samples, random_state=42)
                balanced_dfs.append(sampled_df)
                logger.info(f"Categoría {category}: submuestreada de {len(category_df)} a {max_samples}")
            else:
                balanced_dfs.append(category_df)
                logger.info(f"Categoría {category}: mantenida con {len(category_df)} muestras")
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Mezclar el dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Dataset balanceado final: {len(balanced_df)} muestras")
        final_counts = balanced_df['category'].value_counts()
        for category, count in final_counts.items():
            logger.info(f"  {category}: {count} muestras")
        
        return balanced_df
    
    def create_train_val_split(self, df, test_size=0.2, val_size=0.1):
        """Crear splits de entrenamiento, validación y test"""
        logger.info("Creando splits de datos...")
        
        # Primero separar test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'],
            random_state=42
        )
        
        # Luego separar train y validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size/(1-test_size),  # Ajustar proporción
            stratify=train_val_df['label'],
            random_state=42
        )
        
        logger.info(f"Split creado:")
        logger.info(f"  Entrenamiento: {len(train_df)} muestras")
        logger.info(f"  Validación: {len(val_df)} muestras") 
        logger.info(f"  Test: {len(test_df)} muestras")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df, val_df, test_df, output_dir="data/training"):
        """Guardar datasets en archivos CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar datasets
        train_path = os.path.join(output_dir, "train_survey_data.csv")
        val_path = os.path.join(output_dir, "val_survey_data.csv")
        test_path = os.path.join(output_dir, "test_survey_data.csv")
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        # Guardar mapeo de etiquetas
        label_mapping = pd.DataFrame([
            {'label': k, 'category': v} for k, v in self.cluster_labels.items()
        ])
        label_path = os.path.join(output_dir, "label_mapping.csv")
        label_mapping.to_csv(label_path, index=False, encoding='utf-8')
        
        logger.info(f"Datasets guardados en: {output_dir}")
        logger.info(f"  Entrenamiento: {train_path}")
        logger.info(f"  Validación: {val_path}")
        logger.info(f"  Test: {test_path}")
        logger.info(f"  Mapeo de etiquetas: {label_path}")
        
        return train_path, val_path, test_path, label_path

def main():
    """Función principal"""
    logger.info("Iniciando creación de datos de entrenamiento desde encuesta...")
    
    # Configurar rutas
    survey_file = "data/encuestas/acum_LLM.sav"
    
    if not os.path.exists(survey_file):
        logger.error(f"Archivo de encuesta no encontrado: {survey_file}")
        return 1
    
    try:
        # Crear procesador
        processor = SurveyDataProcessor(survey_file)
        
        # Crear dataset de entrenamiento
        training_df = processor.create_training_dataset(
            min_samples_per_class=100,
            max_samples_per_class=3000
        )
        
        # Crear splits
        train_df, val_df, test_df = processor.create_train_val_split(training_df)
        
        # Guardar datasets
        processor.save_datasets(train_df, val_df, test_df)
        
        # Mostrar estadísticas finales
        logger.info("\n" + "="*50)
        logger.info("RESUMEN FINAL")
        logger.info("="*50)
        logger.info(f"Total de muestras procesadas: {len(training_df)}")
        logger.info(f"Categorías identificadas: {len(processor.cluster_labels)}")
        
        logger.info("\nDistribución final por categoría:")
        for category, count in training_df['category'].value_counts().items():
            percentage = (count / len(training_df)) * 100
            logger.info(f"  {category}: {count} muestras ({percentage:.1f}%)")
        
        logger.info("\nDatos listos para fine-tuning de BERT!")
        logger.info("Próximo paso: Ejecutar finetune_bert.py con estos datos")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
