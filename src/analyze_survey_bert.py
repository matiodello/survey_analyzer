#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis de respuestas de preguntas abiertas en encuestas usando BERT refinado.
Este script analiza un archivo SPSS (.sav) y clasifica las respuestas usando 
el modelo BERT fine-tuned con categorías específicas.
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
        logging.FileHandler("analyze_survey_bert.log"),
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

class BERTSurveyAnalyzer:
    """
    Analizador de encuestas usando modelo BERT refinado
    """
    
    def __init__(self, model_path="models/bert-refined"):
        """
        Inicializa el analizador con el modelo BERT
        
        Args:
            model_path: Ruta al modelo BERT entrenado
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.confidence_threshold = 0.7
        self.load_model()
    
    def load_model(self):
        """
        Carga el modelo BERT entrenado y el tokenizer
        """
        try:
            logger.info(f"Cargando modelo BERT desde {self.model_path}")
            
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
            
            logger.info("Modelo BERT cargado exitosamente")
            logger.info(f"Categorías disponibles: {list(self.label_encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo BERT: {str(e)}")
            raise
    
    def predict_response(self, text):
        """
        Clasifica una respuesta de texto usando BERT
        
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
        # Estadísticas por categoría
        category_stats = result_df.groupby('category').agg({
            'response': 'count',
            'confidence': ['mean', 'std', 'min', 'max']
        }).round(4)
        category_stats.columns = ['count', 'conf_mean', 'conf_std', 'conf_min', 'conf_max']
        category_stats = category_stats.sort_values('count', ascending=False)
        
        # Respuestas de baja confianza
        low_conf_df = result_df[result_df['low_confidence'] == True]
        
        # Crear visualizaciones
        self.create_visualizations(result_df, question_name, output_dir)
        
        # Generar reporte de texto
        report_path = os.path.join(output_dir, f'report_{question_name.replace(" ", "_")}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"=== REPORTE DE CLASIFICACIÓN: {question_name} ===\n\n")
            f.write(f"Total de respuestas analizadas: {len(result_df)}\n")
            f.write(f"Respuestas con baja confianza: {len(low_conf_df)} ({len(low_conf_df)/len(result_df)*100:.1f}%)\n")
            f.write(f"Confianza promedio: {result_df['confidence'].mean():.4f}\n\n")
            
            f.write("=== DISTRIBUCIÓN POR CATEGORÍA ===\n")
            f.write(category_stats.to_string())
            f.write("\n\n")
            
            if len(low_conf_df) > 0:
                f.write("=== RESPUESTAS DE BAJA CONFIANZA ===\n")
                for _, row in low_conf_df.head(20).iterrows():
                    f.write(f"Respuesta: {row['response'][:100]}...\n")
                    f.write(f"Categoría: {row['category']} (Confianza: {row['confidence']:.4f})\n")
                    f.write("-" * 50 + "\n")
            
            f.write("\n=== EJEMPLOS POR CATEGORÍA ===\n")
            for category in category_stats.index:
                cat_responses = result_df[result_df['category'] == category]
                high_conf = cat_responses[cat_responses['confidence'] >= self.confidence_threshold]
                if len(high_conf) > 0:
                    f.write(f"\n--- {category} ---\n")
                    for _, row in high_conf.head(5).iterrows():
                        f.write(f"• {row['response'][:80]}... (Conf: {row['confidence']:.3f})\n")
        
        logger.info(f"Reporte guardado en: {report_path}")
        
        return category_stats
    
    def create_visualizations(self, result_df, question_name, output_dir):
        """
        Crea visualizaciones de los resultados
        
        Args:
            result_df: DataFrame con resultados
            question_name: Nombre de la pregunta
            output_dir: Directorio de salida
        """
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribución de categorías
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico de barras de categorías
        category_counts = result_df['category'].value_counts()
        category_counts.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title(f'Distribución de Categorías - {question_name}')
        ax1.set_xlabel('Categoría')
        ax1.set_ylabel('Número de Respuestas')
        
        # Gráfico circular
        category_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Proporción por Categoría')
        ax2.set_ylabel('')
        
        # Distribución de confianza
        result_df['confidence'].hist(bins=20, ax=ax3, alpha=0.7)
        ax3.axvline(self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Umbral ({self.confidence_threshold})')
        ax3.set_title('Distribución de Confianza')
        ax3.set_xlabel('Confianza')
        ax3.set_ylabel('Frecuencia')
        ax3.legend()
        
        # Confianza por categoría
        sns.boxplot(data=result_df, x='category', y='confidence', ax=ax4)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        ax4.set_title('Confianza por Categoría')
        ax4.axhline(self.confidence_threshold, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f'visualization_{question_name.replace(" ", "_")}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizaciones guardadas en: {viz_path}")

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
    
    for col in df.columns:
        # Verificar si es una columna de texto (string)
        if df[col].dtype == 'object':
            # Verificar que tenga suficientes respuestas no nulas
            non_null_count = df[col].count()
            if non_null_count > 10:
                # Verificar que las respuestas sean texto significativo
                sample_responses = df[col].dropna().head(10)
                avg_length = sample_responses.astype(str).str.len().mean()
                
                if avg_length > 10:  # Respuestas con longitud promedio > 10 caracteres
                    open_questions.append(col)
                    logger.info(f"Pregunta abierta identificada: {col} ({non_null_count} respuestas)")
    
    logger.info(f"Total de preguntas abiertas identificadas: {len(open_questions)}")
    return open_questions

def analyze_survey_with_bert(file_path, model_path="models/bert-refined"):
    """
    Función principal para analizar respuestas de encuestas con BERT
    
    Args:
        file_path: Ruta al archivo .sav
        model_path: Ruta al modelo BERT entrenado
    """
    # Inicializar analizador BERT
    analyzer = BERTSurveyAnalyzer(model_path)
    
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
    output_dir = os.path.join('outputs', 'bert_survey_analysis')
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
        
        # Clasificar respuestas con BERT
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
    logger.info(f"{'='*50}")

def generate_consolidated_report(all_results, all_stats, output_dir):
    """
    Genera un reporte consolidado de todas las preguntas analizadas
    
    Args:
        all_results: Diccionario con resultados por pregunta
        all_stats: Diccionario con estadísticas por pregunta
        output_dir: Directorio de salida
    """
    consolidated_path = os.path.join(output_dir, 'consolidated_report.txt')
    
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        f.write("=== REPORTE CONSOLIDADO DE ANÁLISIS BERT ===\n\n")
        f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de preguntas analizadas: {len(all_results)}\n\n")
        
        # Resumen por pregunta
        for question, result_df in all_results.items():
            f.write(f"\n--- {question} ---\n")
            f.write(f"Total respuestas: {len(result_df)}\n")
            f.write(f"Confianza promedio: {result_df['confidence'].mean():.4f}\n")
            f.write(f"Respuestas baja confianza: {result_df['low_confidence'].sum()}\n")
            
            # Top 3 categorías
            top_categories = result_df['category'].value_counts().head(3)
            f.write("Top 3 categorías:\n")
            for cat, count in top_categories.items():
                pct = (count / len(result_df)) * 100
                f.write(f"  • {cat}: {count} ({pct:.1f}%)\n")
        
        # Estadísticas globales por categoría
        f.write("\n=== ESTADÍSTICAS GLOBALES POR CATEGORÍA ===\n")
        all_categories = {}
        for result_df in all_results.values():
            for cat in result_df['category'].value_counts().index:
                if cat not in all_categories:
                    all_categories[cat] = 0
                all_categories[cat] += result_df[result_df['category'] == cat].shape[0]
        
        total_responses = sum(all_categories.values())
        for cat, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_responses) * 100
            f.write(f"{cat}: {count} respuestas ({pct:.1f}%)\n")
    
    logger.info(f"Reporte consolidado guardado en: {consolidated_path}")

def main():
    """Función principal"""
    file_path = "data/encuestas/acum_LLM.sav"
    
    if not os.path.exists(file_path):
        logger.error(f"Archivo no encontrado: {file_path}")
        return 1
    
    try:
        analyze_survey_with_bert(file_path)
        return 0
    except Exception as e:
        logger.error(f"Error en el análisis: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
