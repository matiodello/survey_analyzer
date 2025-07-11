#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script mejorado para crear datos de entrenamiento con categorías refinadas,
eliminando la categoría "GENERAL_MIXTO" y creando subcategorías más específicas.
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
        logging.FileHandler("create_refined_training_data.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RefinedSurveyDataProcessor:
    """Procesador mejorado con categorías refinadas"""
    
    def __init__(self, survey_file_path):
        self.survey_file_path = survey_file_path
        
        # Categorías refinadas (eliminamos GENERAL_MIXTO)
        self.cluster_labels = {
            0: "INSEGURIDAD",
            1: "TRABAJO", 
            2: "ECONOMIA",
            3: "POLITICA",      # Nueva: gobierno, políticos, presidente
            4: "CORRUPCION",
            5: "INFLACION",
            6: "EDUCACION",     # Nueva: educación, escuelas
            7: "POBREZA",       # Nueva: pobreza, necesidades
            8: "SALUD",         # Nueva: salud, hospitales
            9: "SERVICIOS",     # Nueva: servicios públicos
            10: "OTROS"         # Para casos que no encajen en ninguna categoría específica
        }
        
        # Palabras clave refinadas y expandidas
        self.cluster_keywords = {
            0: ["inseguridad", "delincuencia", "crimen", "robo", "violencia", "seguridad", "ladrones"],
            1: ["trabajo", "empleo", "desempleo", "falta de trabajo", "laboral", "desocupacion", "empleos"],
            2: ["economia", "economico", "crisis economica", "situacion economica", "recesion"],
            3: ["politica", "gobierno", "presidente", "politicos", "ministros", "congreso", "elecciones", "partidos", "milei", "gestion", "nacional"],
            4: ["corrupcion", "corruptos", "corrupcion politica", "corrupcion de los politicos"],
            5: ["inflacion", "precios", "carestia", "costo de vida", "aumento de precios"],
            6: ["educacion", "escuela", "universidad", "estudio", "maestros", "profesores", "enseñanza", "falta de educacion"],
            7: ["pobreza", "pobres", "hambre", "necesidades", "carencias", "falta de recursos", "indigencia"],
            8: ["salud", "hospital", "medico", "medicina", "enfermedad", "atencion medica", "obras sociales", "hospitales"],
            9: ["servicios", "luz", "agua", "gas", "transporte", "colectivos", "trenes", "servicios publicos"],
            10: []  # OTROS no tiene palabras clave específicas
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
    
    def assign_refined_labels(self, responses):
        """Asignar etiquetas refinadas basadas en palabras clave mejoradas"""
        labels = []
        assignment_details = []
        
        for response in responses:
            processed_response = self.preprocess_text(response)
            scores = {}
            matched_keywords = {}
            
            # Calcular score para cada cluster
            for cluster_id, keywords in self.cluster_keywords.items():
                if cluster_id == 10:  # Skip OTROS por ahora
                    continue
                    
                score = 0
                matched = []
                
                for keyword in keywords:
                    if keyword in processed_response:
                        # Dar más peso a coincidencias exactas
                        if processed_response == keyword:
                            score += 5
                            matched.append(f"{keyword}(exact)")
                        elif processed_response.startswith(keyword) or processed_response.endswith(keyword):
                            score += 3
                            matched.append(f"{keyword}(start/end)")
                        else:
                            score += 1
                            matched.append(f"{keyword}(contains)")
                
                scores[cluster_id] = score
                matched_keywords[cluster_id] = matched
            
            # Asignar al cluster con mayor score
            if max(scores.values()) > 0:
                best_cluster = max(scores, key=scores.get)
                labels.append(best_cluster)
                assignment_details.append({
                    'text': response,
                    'processed': processed_response,
                    'assigned_to': self.cluster_labels[best_cluster],
                    'score': scores[best_cluster],
                    'matched_keywords': matched_keywords[best_cluster]
                })
            else:
                # Si no hay coincidencias claras, asignar a OTROS
                labels.append(10)
                assignment_details.append({
                    'text': response,
                    'processed': processed_response,
                    'assigned_to': 'OTROS',
                    'score': 0,
                    'matched_keywords': []
                })
        
        return labels, assignment_details
    
    def create_refined_training_dataset(self, min_samples_per_class=50, max_samples_per_class=2500):
        """Crear dataset de entrenamiento con categorías refinadas"""
        logger.info("Creando dataset de entrenamiento refinado...")
        
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
        
        # Asignar etiquetas refinadas
        labels, assignment_details = self.assign_refined_labels(valid_responses)
        
        # Crear DataFrame
        training_df = pd.DataFrame({
            'text': valid_responses,
            'label': labels,
            'category': [self.cluster_labels[label] for label in labels]
        })
        
        # Mostrar distribución inicial
        logger.info("\nDistribución inicial:")
        initial_counts = training_df['category'].value_counts()
        for category, count in initial_counts.items():
            percentage = (count / len(training_df)) * 100
            logger.info(f"  {category}: {count} muestras ({percentage:.1f}%)")
        
        # Balancear dataset
        balanced_df = self.balance_refined_dataset(training_df, min_samples_per_class, max_samples_per_class)
        
        # Guardar detalles de asignación para revisión
        details_df = pd.DataFrame(assignment_details)
        details_df.to_csv('outputs/assignment_details.csv', index=False, encoding='utf-8')
        logger.info("Detalles de asignación guardados en: outputs/assignment_details.csv")
        
        return balanced_df
    
    def balance_refined_dataset(self, df, min_samples, max_samples):
        """Balancear dataset refinado"""
        logger.info("Balanceando dataset refinado...")
        
        # Contar muestras por categoría
        category_counts = df['category'].value_counts()
        logger.info("Distribución antes del balanceo:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} muestras")
        
        balanced_dfs = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            if len(category_df) < min_samples:
                if category == 'OTROS':
                    # Para OTROS, podemos ser más permisivos
                    logger.info(f"Categoría {category} tiene pocas muestras ({len(category_df)}), usando todas")
                    balanced_dfs.append(category_df)
                else:
                    logger.warning(f"Categoría {category} tiene muy pocas muestras ({len(category_df)}), considerando eliminar o combinar")
                    if len(category_df) > 20:  # Umbral mínimo absoluto
                        balanced_dfs.append(category_df)
            elif len(category_df) > max_samples:
                # Submuestrear
                sampled_df = category_df.sample(n=max_samples, random_state=42)
                balanced_dfs.append(sampled_df)
                logger.info(f"Categoría {category}: submuestreada de {len(category_df)} a {max_samples}")
            else:
                balanced_dfs.append(category_df)
                logger.info(f"Categoría {category}: mantenida con {len(category_df)} muestras")
        
        if not balanced_dfs:
            logger.error("No hay categorías válidas después del balanceo")
            return pd.DataFrame()
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Mezclar el dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Dataset balanceado final: {len(balanced_df)} muestras")
        final_counts = balanced_df['category'].value_counts()
        for category, count in final_counts.items():
            percentage = (count / len(balanced_df)) * 100
            logger.info(f"  {category}: {count} muestras ({percentage:.1f}%)")
        
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
    
    def save_refined_datasets(self, train_df, val_df, test_df, output_dir="data/training_refined"):
        """Guardar datasets refinados"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar datasets
        train_path = os.path.join(output_dir, "train_refined_data.csv")
        val_path = os.path.join(output_dir, "val_refined_data.csv")
        test_path = os.path.join(output_dir, "test_refined_data.csv")
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        # Guardar mapeo de etiquetas refinado
        label_mapping = pd.DataFrame([
            {'label': k, 'category': v} for k, v in self.cluster_labels.items()
        ])
        label_path = os.path.join(output_dir, "refined_label_mapping.csv")
        label_mapping.to_csv(label_path, index=False, encoding='utf-8')
        
        logger.info(f"Datasets refinados guardados en: {output_dir}")
        logger.info(f"  Entrenamiento: {train_path}")
        logger.info(f"  Validación: {val_path}")
        logger.info(f"  Test: {test_path}")
        logger.info(f"  Mapeo de etiquetas: {label_path}")
        
        return train_path, val_path, test_path, label_path

def main():
    """Función principal"""
    logger.info("Iniciando creación de datos de entrenamiento refinados...")
    
    # Configurar rutas
    survey_file = "data/encuestas/acum_LLM.sav"
    
    if not os.path.exists(survey_file):
        logger.error(f"Archivo de encuesta no encontrado: {survey_file}")
        return 1
    
    try:
        # Crear procesador refinado
        processor = RefinedSurveyDataProcessor(survey_file)
        
        # Crear dataset de entrenamiento refinado
        training_df = processor.create_refined_training_dataset(
            min_samples_per_class=50,
            max_samples_per_class=2500
        )
        
        if len(training_df) == 0:
            logger.error("No se pudo crear el dataset refinado")
            return 1
        
        # Crear splits
        train_df, val_df, test_df = processor.create_train_val_split(training_df)
        
        # Guardar datasets
        processor.save_refined_datasets(train_df, val_df, test_df)
        
        # Mostrar estadísticas finales
        logger.info("\n" + "="*60)
        logger.info("RESUMEN FINAL - DATASET REFINADO")
        logger.info("="*60)
        logger.info(f"Total de muestras procesadas: {len(training_df)}")
        logger.info(f"Categorías refinadas: {len([k for k, v in processor.cluster_labels.items() if k != 10 or len(training_df[training_df['label'] == k]) > 0])}")
        
        logger.info("\nDistribución final por categoría:")
        for category, count in training_df['category'].value_counts().items():
            percentage = (count / len(training_df)) * 100
            logger.info(f"  {category}: {count} muestras ({percentage:.1f}%)")
        
        logger.info("\nMejoras implementadas:")
        logger.info("  ✓ Eliminada categoría 'GENERAL_MIXTO' demasiado amplia")
        logger.info("  ✓ Agregadas categorías específicas: POLÍTICA, EDUCACIÓN, POBREZA, SALUD, SERVICIOS")
        logger.info("  ✓ Palabras clave expandidas y refinadas")
        logger.info("  ✓ Sistema de scoring mejorado para asignación de etiquetas")
        
        logger.info("\nDatos refinados listos para fine-tuning de BERT!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
