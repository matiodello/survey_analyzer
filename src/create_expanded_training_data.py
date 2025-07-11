#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crear dataset expandido con subcategorías de "OTROS"
Autor: Sistema de Análisis BERT
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import logging
import sys
import warnings
warnings.filterwarnings("ignore")
import re

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("expanded_training_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExpandedDatasetCreator:
    """
    Creador de dataset expandido con subcategorías de OTROS
    """
    
    def __init__(self):
        """Inicializar el creador de dataset"""
        self.original_dataset = None
        self.otros_classification = None
        self.expanded_dataset = None
        
        # Definir reglas de palabras clave para reclasificación
        self.keyword_rules = {
            'CORRUPCION': [
                'corrupción', 'corrupto', 'corruptos', 'chorros', 'ladrones',
                'justicia', 'impunidad', 'impuestos', 'gobernantes', 'malos políticos',
                'políticos chorros', 'roban', 'robaron', 'vagos'
            ],
            'CULTURA_VALORES': [
                'educación', 'falta educación', 'cultura', 'valores', 'educativo',
                'enseñanza', 'formación', 'ignorancia', 'falta cultura'
            ],
            'PODER_ADQUISITIVO': [
                'sueldos', 'bajos', 'plata', 'alcanza', 'salario', 'salarios',
                'económico', 'no alcanza', 'plata alcanza', 'falta plata',
                'poder adquisitivo', 'sueldo bajo', 'dinero', 'platita'
            ]
        }
        
    def load_datasets(self):
        """Cargar el dataset original y las clasificaciones de OTROS"""
        try:
            # Cargar dataset original refinado
            logger.info("Cargando dataset original refinado...")
            train_path = "data/training_refined/train_refined_data.csv"
            if os.path.exists(train_path):
                self.original_dataset = pd.read_csv(train_path)
                logger.info(f"Dataset original cargado: {len(self.original_dataset)} muestras")
            else:
                logger.error(f"Dataset original no encontrado: {train_path}")
                return False
            
            # Cargar clasificaciones de OTROS con clusters
            logger.info("Cargando clasificaciones de OTROS...")
            otros_path = "outputs/bert_survey_analysis/classified_PROBLEMA_AB_C.csv"
            if os.path.exists(otros_path):
                # Cargar y filtrar solo OTROS
                full_classification = pd.read_csv(otros_path)
                self.otros_classification = full_classification[
                    full_classification['category'] == 'OTROS'
                ].copy()
                logger.info(f"Respuestas OTROS cargadas: {len(self.otros_classification)}")
                
                # Verificar si tiene columna cluster (del análisis anterior)
                if 'cluster' not in self.otros_classification.columns:
                    logger.info("No se encontró columna 'cluster'. Se usarán reglas de palabras clave.")
                
            else:
                logger.error(f"Archivo de clasificaciones no encontrado: {otros_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error cargando datasets: {str(e)}")
            return False
    
    def classify_by_keywords(self, text):
        """Clasificar texto usando reglas de palabras clave"""
        if pd.isna(text):
            return 'OTROS'
        
        text_lower = str(text).lower()
        
        # Revisar cada categoría y sus palabras clave
        for category, keywords in self.keyword_rules.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return category
        
        # Si no coincide con ninguna regla, mantener como OTROS
        return 'OTROS'
    
    def reclassify_otros_samples(self):
        """Reclasificar las muestras de OTROS según palabras clave"""
        logger.info("Reclasificando muestras de OTROS usando palabras clave...")
        
        # Aplicar clasificación por palabras clave
        self.otros_classification['new_category'] = self.otros_classification['response'].apply(
            self.classify_by_keywords
        )
        
        # Contar reclasificaciones
        reclassification_counts = self.otros_classification['new_category'].value_counts()
        logger.info("Distribución de reclasificaciones:")
        for category, count in reclassification_counts.items():
            percentage = (count / len(self.otros_classification)) * 100
            logger.info(f"  {category}: {count} muestras ({percentage:.1f}%)")
        
        # Crear dataset de muestras reclasificadas
        reclassified_samples = []
        
        for _, row in self.otros_classification.iterrows():
            # Usar la nueva categoría
            new_category = row['new_category']
            response = row['response']
            
            # Solo agregar si no es NaN
            if pd.notna(new_category) and pd.notna(response):
                reclassified_samples.append({
                    'text': response,
                    'category': new_category,
                    'source': 'reclassified_otros'
                })
        
        logger.info(f"Total de muestras reclasificadas: {len(reclassified_samples)}")
        return pd.DataFrame(reclassified_samples)
    
    def create_expanded_dataset(self):
        """Crear el dataset expandido combinando original + reclasificado"""
        logger.info("Creando dataset expandido...")
        
        # Reclasificar muestras OTROS
        reclassified_df = self.reclassify_otros_samples()
        
        # Preparar dataset original
        original_prepared = self.original_dataset[['text', 'category']].copy()
        original_prepared['source'] = 'original_training'
        
        # Combinar datasets
        self.expanded_dataset = pd.concat([
            original_prepared,
            reclassified_df
        ], ignore_index=True)
        
        # Estadísticas del dataset expandido
        logger.info(f"Dataset expandido creado: {len(self.expanded_dataset)} muestras totales")
        
        category_counts = self.expanded_dataset['category'].value_counts()
        logger.info("Distribución por categoría:")
        for category, count in category_counts.items():
            percentage = (count / len(self.expanded_dataset)) * 100
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        return True
    
    def balance_dataset(self, max_samples_per_category=2000):
        """Balancear el dataset para evitar clases dominantes"""
        logger.info(f"Balanceando dataset (máximo {max_samples_per_category} por categoría)...")
        
        balanced_samples = []
        
        for category in self.expanded_dataset['category'].unique():
            category_samples = self.expanded_dataset[
                self.expanded_dataset['category'] == category
            ]
            
            # Si tiene más muestras que el máximo, hacer sampling
            if len(category_samples) > max_samples_per_category:
                category_samples = category_samples.sample(
                    n=max_samples_per_category, 
                    random_state=42
                )
                logger.info(f"  {category}: reducido de {len(self.expanded_dataset[self.expanded_dataset['category'] == category])} a {max_samples_per_category}")
            else:
                logger.info(f"  {category}: mantenido {len(category_samples)} muestras")
            
            balanced_samples.append(category_samples)
        
        self.expanded_dataset = pd.concat(balanced_samples, ignore_index=True)
        logger.info(f"Dataset balanceado: {len(self.expanded_dataset)} muestras totales")
        
        return True
    
    def create_train_test_splits(self):
        """Crear splits de entrenamiento, validación y prueba"""
        logger.info("Creando splits del dataset...")
        
        # Primero split train/temp (80/20)
        train_df, temp_df = train_test_split(
            self.expanded_dataset,
            test_size=0.2,
            random_state=42,
            stratify=self.expanded_dataset['category']
        )
        
        # Luego split temp en validation/test (10/10)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['category']
        )
        
        logger.info(f"Split creados:")
        logger.info(f"  Entrenamiento: {len(train_df)} muestras")
        logger.info(f"  Validación: {len(val_df)} muestras")
        logger.info(f"  Prueba: {len(test_df)} muestras")
        
        return train_df, val_df, test_df
    
    def save_datasets(self, train_df, val_df, test_df):
        """Guardar los datasets y metadatos"""
        logger.info("Guardando datasets...")
        
        # Crear directorio
        output_dir = "data/expanded_training"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar splits
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        
        # Guardar dataset completo
        self.expanded_dataset.to_csv(os.path.join(output_dir, "expanded_training_data.csv"), index=False)
        
        # Crear y guardar label encoder
        all_categories = sorted(self.expanded_dataset['category'].unique())
        label_encoder = LabelEncoder()
        label_encoder.fit(all_categories)
        
        # Guardar como pickle
        with open(os.path.join(output_dir, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Guardar como JSON también
        label_dict = {str(i): category for i, category in enumerate(all_categories)}
        with open(os.path.join(output_dir, "label_encoder.json"), 'w') as f:
            json.dump(label_dict, f, indent=2)
        
        # Guardar metadatos
        metadata = {
            'total_samples': len(self.expanded_dataset),
            'num_categories': len(all_categories),
            'categories': all_categories,
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'category_distribution': self.expanded_dataset['category'].value_counts().to_dict(),
            'keyword_rules_used': self.keyword_rules
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Datasets guardados en: {output_dir}")
        logger.info(f"Categorías finales: {all_categories}")
        
        return output_dir
    
    def generate_report(self):
        """Generar reporte del proceso"""
        logger.info("Generando reporte...")
        
        output_dir = "outputs/expanded_dataset_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "expanded_dataset_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== REPORTE DEL DATASET EXPANDIDO ===\n\n")
            f.write(f"Fecha de creación: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Información general
            f.write("=== INFORMACIÓN GENERAL ===\n")
            f.write(f"Total de muestras: {len(self.expanded_dataset)}\n")
            f.write(f"Número de categorías: {len(self.expanded_dataset['category'].unique())}\n\n")
            
            # Reglas de palabras clave aplicadas
            f.write("=== REGLAS DE PALABRAS CLAVE APLICADAS ===\n")
            for category, keywords in self.keyword_rules.items():
                f.write(f"{category}: {', '.join(keywords[:10])}{'...' if len(keywords) > 10 else ''}\n")
            f.write("\n")
            
            # Distribución por categoría
            f.write("=== DISTRIBUCIÓN POR CATEGORÍA ===\n")
            category_counts = self.expanded_dataset['category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(self.expanded_dataset)) * 100
                f.write(f"{category}: {count} muestras ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Comparación con dataset original
            f.write("=== COMPARACIÓN CON DATASET ORIGINAL ===\n")
            original_categories = set(self.original_dataset['category'].unique())
            expanded_categories = set(self.expanded_dataset['category'].unique())
            new_categories = expanded_categories - original_categories
            
            f.write(f"Categorías originales: {len(original_categories)}\n")
            f.write(f"Categorías expandidas: {len(expanded_categories)}\n")
            f.write(f"Nuevas categorías añadidas: {list(new_categories)}\n")
            
        logger.info(f"Reporte guardado en: {report_path}")
        return report_path

def main():
    """Función principal"""
    try:
        # Crear instancia del creador
        creator = ExpandedDatasetCreator()
        
        # Cargar datasets
        if not creator.load_datasets():
            return 1
        
        # Crear dataset expandido
        if not creator.create_expanded_dataset():
            return 1
        
        # Balancear dataset
        creator.balance_dataset()
        
        # Crear splits
        train_df, val_df, test_df = creator.create_train_test_splits()
        
        # Guardar datasets
        output_dir = creator.save_datasets(train_df, val_df, test_df)
        
        # Generar reporte
        creator.generate_report()
        
        logger.info("Dataset expandido creado exitosamente")
        logger.info(f"Archivos disponibles en: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en la creación del dataset expandido: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
