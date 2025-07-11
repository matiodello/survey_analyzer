#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis de respuestas de preguntas abiertas en encuestas usando BERT expandido.
Este script analiza un archivo SPSS (.sav) y clasifica las respuestas usando 
el modelo BERT fine-tuned con categorías expandidas (incluyendo subcategorías).
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyze_survey_bert_expanded.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleLabelEncoder:
    """
    Codificador de etiquetas simple basado en diccionario JSON
    """
    def __init__(self, label_dict):
        self.label_dict = label_dict
        self.classes_ = list(label_dict.values())
        self.reverse_dict = {v: int(k) for k, v in label_dict.items()}
    
    def inverse_transform(self, indices):
        """Convierte índices a etiquetas"""
        if isinstance(indices, (list, tuple)):
            return [self.label_dict[str(idx)] for idx in indices]
        else:
            return self.label_dict[str(indices)]
    
    def transform(self, labels):
        """Convierte etiquetas a índices"""
        if isinstance(labels, (list, tuple)):
            return [self.reverse_dict[label] for label in labels]
        else:
            return self.reverse_dict[labels]

class BERTExpandedSurveyAnalyzer:
    """
    Analizador de encuestas usando modelo BERT expandido con subcategorías
    """
    
    def __init__(self, model_path="models/bert-expanded"):
        """
        Inicializa el analizador con el modelo BERT expandido
        
        Args:
            model_path: Ruta al modelo BERT entrenado expandido
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.confidence_threshold = 0.7
        self.load_model()
    
    def load_model(self):
        """
        Carga el modelo BERT expandido entrenado y el tokenizer
        """
        try:
            logger.info(f"Cargando modelo BERT expandido desde {self.model_path}")
            
            # Cargar modelo y tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Cargar label encoder
            label_encoder_path = os.path.join(self.model_path, "label_encoder.json")
            with open(label_encoder_path, 'r') as f:
                label_dict = json.load(f)
                self.label_encoder = SimpleLabelEncoder(label_dict)
            
            # Configurar modelo para evaluación
            self.model.eval()
            
            logger.info("Modelo BERT expandido cargado exitosamente")
            logger.info(f"Categorías disponibles: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo BERT expandido: {str(e)}")
            raise
    
    def predict_response(self, text):
        """
        Clasifica una respuesta de texto usando BERT expandido
        
        Args:
            text: Texto a clasificar
            
        Returns:
            tuple: (categoria_predicha, confianza, todas_las_probabilidades)
        """
        try:
            # Tokenizar texto
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Obtener predicción y confianza
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Convertir índice a etiqueta
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Obtener todas las probabilidades por categoría
            all_probs = {}
            for i, prob in enumerate(probabilities[0]):
                category = self.label_encoder.inverse_transform([i])[0]
                all_probs[category] = prob.item()
            
            return predicted_label, confidence, all_probs
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return "OTROS", 0.0, {}
    
    def classify_responses(self, responses, question_name):
        """
        Clasifica múltiples respuestas
        
        Args:
            responses: Lista de respuestas a clasificar
            question_name: Nombre de la pregunta
            
        Returns:
            DataFrame con resultados de clasificación
        """
        logger.info(f"Clasificando {len(responses)} respuestas para {question_name}")
        
        results = []
        low_confidence_count = 0
        
        for i, response in enumerate(responses):
            if pd.isna(response) or str(response).strip() == "":
                continue
                
            # Limpiar y procesar texto
            clean_response = str(response).strip()
            
            # Clasificar respuesta
            category, confidence, all_probs = self.predict_response(clean_response)
            
            # Marcar respuestas con baja confianza
            is_low_confidence = confidence < self.confidence_threshold
            if is_low_confidence:
                low_confidence_count += 1
            
            results.append({
                'response': clean_response,
                'category': category,
                'confidence': confidence,
                'low_confidence': is_low_confidence,
                'all_probabilities': all_probs
            })
        
        # Crear DataFrame con resultados
        result_df = pd.DataFrame(results)
        
        logger.info(f"Clasificación completada. {low_confidence_count} respuestas con baja confianza (<{self.confidence_threshold})")
        
        return result_df
    
    def generate_classification_report(self, result_df, question_name, output_dir):
        """
        Genera reporte detallado de clasificación
        
        Args:
            result_df: DataFrame con resultados
            question_name: Nombre de la pregunta
            output_dir: Directorio de salida
        """
        try:
            # Estadísticas básicas
            total_responses = len(result_df)
            category_counts = result_df['category'].value_counts()
            confidence_stats = result_df['confidence'].describe()
            low_confidence_count = result_df['low_confidence'].sum()
            
            # Crear reporte
            report_path = os.path.join(output_dir, f'report_{question_name.replace(" ", "_")}.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"REPORTE DE ANÁLISIS - {question_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total de respuestas analizadas: {total_responses}\n")
                f.write(f"Respuestas con baja confianza: {low_confidence_count} ({low_confidence_count/total_responses*100:.1f}%)\n\n")
                
                f.write("DISTRIBUCIÓN POR CATEGORÍA:\n")
                f.write("-" * 30 + "\n")
                for category, count in category_counts.items():
                    percentage = (count / total_responses) * 100
                    f.write(f"{category}: {count} ({percentage:.1f}%)\n")
                
                f.write("\nESTADÍSTICAS DE CONFIANZA:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Promedio: {confidence_stats['mean']:.3f}\n")
                f.write(f"Mediana: {confidence_stats['50%']:.3f}\n")
                f.write(f"Mínimo: {confidence_stats['min']:.3f}\n")
                f.write(f"Máximo: {confidence_stats['max']:.3f}\n")
                
                f.write("\n=== EJEMPLOS POR CATEGORÍA ===\n\n")
                
                for category in category_counts.index:
                    category_examples = result_df[result_df['category'] == category].head(5)
                    f.write(f"--- {category} ---\n")
                    for _, row in category_examples.iterrows():
                        f.write(f"• {row['response']}... (Conf: {row['confidence']:.3f})\n")
                    f.write("\n")
            
            logger.info(f"Reporte guardado en: {report_path}")
            
            # Crear visualizaciones
            self.create_visualizations(result_df, question_name, output_dir)
            
            return {
                'total_responses': total_responses,
                'category_distribution': category_counts.to_dict(),
                'confidence_stats': confidence_stats.to_dict(),
                'low_confidence_count': low_confidence_count
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            return None
    
    def create_visualizations(self, result_df, question_name, output_dir):
        """
        Crea visualizaciones de los resultados
        
        Args:
            result_df: DataFrame con resultados
            question_name: Nombre de la pregunta
            output_dir: Directorio de salida
        """
        try:
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Crear figura con subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Análisis de Clasificación - {question_name}', fontsize=16, fontweight='bold')
            
            # 1. Distribución por categoría
            category_counts = result_df['category'].value_counts()
            ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribución por Categoría')
            
            # 2. Histograma de confianza
            ax2.hist(result_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(self.confidence_threshold, color='red', linestyle='--', label=f'Umbral ({self.confidence_threshold})')
            ax2.set_xlabel('Confianza')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Distribución de Confianza')
            ax2.legend()
            
            # 3. Barras por categoría
            category_counts.plot(kind='bar', ax=ax3, color='lightcoral')
            ax3.set_title('Conteo por Categoría')
            ax3.set_xlabel('Categoría')
            ax3.set_ylabel('Número de Respuestas')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Boxplot de confianza por categoría
            result_df.boxplot(column='confidence', by='category', ax=ax4)
            ax4.set_title('Confianza por Categoría')
            ax4.set_xlabel('Categoría')
            ax4.set_ylabel('Confianza')
            
            plt.tight_layout()
            
            # Guardar visualización
            viz_path = os.path.join(output_dir, f'visualization_{question_name.replace(" ", "_")}.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualización guardada en: {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creando visualizaciones: {str(e)}")


def load_sav_file(file_path):
    """
    Carga un archivo SPSS (.sav) y devuelve un DataFrame
    
    Args:
        file_path: Ruta al archivo .sav
        
    Returns:
        DataFrame con los datos y diccionario de metadatos
    """
    try:
        logger.info(f"Cargando archivo: {file_path}")
        df, meta = pyreadstat.read_sav(file_path)
        logger.info(f"Archivo cargado exitosamente. Dimensiones: {df.shape}")
        return df, meta
    except Exception as e:
        logger.error(f"Error cargando archivo: {str(e)}")
        return None, None

def identify_open_questions(df, meta):
    """
    Identifica las preguntas abiertas en el conjunto de datos
    
    Args:
        df: DataFrame con los datos
        meta: Metadatos del archivo SPSS
        
    Returns:
        Lista de nombres de columnas con preguntas abiertas
    """
    open_questions = []
    
    for column in df.columns:
        # Verificar si la columna contiene texto (respuestas abiertas)
        if df[column].dtype == 'object':
            # Verificar que no sean todas respuestas vacías
            non_null_responses = df[column].dropna()
            if len(non_null_responses) > 0:
                # Verificar que las respuestas tengan longitud variable (indicativo de texto libre)
                response_lengths = non_null_responses.astype(str).str.len()
                if response_lengths.std() > 5:  # Variabilidad en longitud
                    open_questions.append(column)
                    logger.info(f"Pregunta abierta identificada: {column} ({len(non_null_responses)} respuestas)")
    
    return open_questions

def analyze_survey_with_bert_expanded(file_path, model_path="models/bert-expanded"):
    """
    Función principal para analizar respuestas de encuestas con BERT expandido
    
    Args:
        file_path: Ruta al archivo .sav
        model_path: Ruta al modelo BERT expandido entrenado
    """
    # Inicializar analizador BERT expandido
    analyzer = BERTExpandedSurveyAnalyzer(model_path)
    
    # Cargar archivo
    df, meta = load_sav_file(file_path)
    if df is None:
        return
    
    # Identificar preguntas abiertas
    open_questions = identify_open_questions(df, meta)
    
    if not open_questions:
        logger.warning("No se identificaron preguntas abiertas en el archivo.")
        return
    
    # Crear directorio para resultados
    output_dir = os.path.join('outputs', 'bert_expanded_survey_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analizar cada pregunta abierta
    all_results = {}
    all_stats = {}
    
    for question in open_questions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Analizando pregunta: {question}")
        logger.info(f"{'='*50}")
        
        # Obtener respuestas no nulas
        responses = df[question].dropna().tolist()
        
        if len(responses) < 10:
            logger.warning(f"Muy pocas respuestas para {question}. Omitiendo análisis.")
            continue
        
        # Clasificar respuestas con BERT expandido
        result_df = analyzer.classify_responses(responses, question)
        
        if result_df is not None and len(result_df) > 0:
            all_results[question] = result_df
            
            # Generar reporte detallado
            stats = analyzer.generate_classification_report(result_df, question, output_dir)
            all_stats[question] = stats
            
            # Guardar resultados CSV
            csv_path = os.path.join(output_dir, f'classified_{question.replace(" ", "_")}.csv')
            result_df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Resultados guardados en: {csv_path}")
    
    # Generar reporte consolidado
    generate_consolidated_report(all_results, all_stats, output_dir)
    
    logger.info(f"\n{'='*50}")
    logger.info("ANÁLISIS COMPLETADO")
    logger.info(f"Resultados disponibles en: {output_dir}")

def generate_consolidated_report(all_results, all_stats, output_dir):
    """
    Genera un reporte consolidado de todas las preguntas analizadas
    
    Args:
        all_results: Diccionario con resultados por pregunta
        all_stats: Diccionario con estadísticas por pregunta
        output_dir: Directorio de salida
    """
    try:
        report_path = os.path.join(output_dir, 'consolidated_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE CONSOLIDADO - ANÁLISIS DE ENCUESTAS CON BERT EXPANDIDO\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Preguntas analizadas: {len(all_results)}\n\n")
            
            for question, stats in all_stats.items():
                f.write(f"PREGUNTA: {question}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total respuestas: {stats['total_responses']}\n")
                f.write(f"Respuestas baja confianza: {stats['low_confidence_count']}\n")
                f.write("Distribución por categoría:\n")
                
                for category, count in stats['category_distribution'].items():
                    percentage = (count / stats['total_responses']) * 100
                    f.write(f"  - {category}: {count} ({percentage:.1f}%)\n")
                
                f.write("\n")
        
        logger.info(f"Reporte consolidado guardado en: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generando reporte consolidado: {str(e)}")

def main():
    """Función principal"""
    # Ruta al archivo de encuesta
    file_path = "data/encuestas/acum_LLM.sav"
    
    if not os.path.exists(file_path):
        logger.error(f"Archivo no encontrado: {file_path}")
        return 1
    
    # Ejecutar análisis con modelo BERT expandido
    analyze_survey_with_bert_expanded(file_path, "models/bert-expanded")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
