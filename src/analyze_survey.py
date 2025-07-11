#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis y agrupación de respuestas de preguntas abiertas en entrevistas.
Este script analiza un archivo SPSS (.sav) y agrupa las respuestas por significado.
"""

import os
import pandas as pd
import numpy as np
import pyreadstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
import sys
import re
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analyze_survey.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK necesarios
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Cargar stopwords en español
try:
    spanish_stopwords = set(stopwords.words('spanish'))
except:
    # Si no se pueden cargar las stopwords en español, usar una lista básica
    spanish_stopwords = {'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened'}

# Añadir stopwords personalizadas
custom_stopwords = {
    'si', 'no', 'porque', 'creo', 'pienso', 'considero', 'opino',
    'bueno', 'malo', 'bien', 'mal', 'etc', 'ejemplo', 'quizás',
    'quizas', 'tal', 'vez', 'talvez', 'asi', 'así', 'osea', 'o sea'
}
spanish_stopwords.update(custom_stopwords)

def load_sav_file(file_path):
    """
    Carga un archivo SPSS (.sav) y devuelve un DataFrame
    
    Args:
        file_path: Ruta al archivo .sav
        
    Returns:
        DataFrame con los datos y diccionario de metadatos
    """
    logger.info(f"Cargando archivo: {file_path}")
    try:
        df, meta = pyreadstat.read_sav(file_path)
        logger.info(f"Archivo cargado exitosamente. Dimensiones: {df.shape}")
        return df, meta
    except Exception as e:
        logger.error(f"Error al cargar el archivo: {e}")
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
        # Verificar si la columna contiene texto
        if df[col].dtype == 'object':
            # Obtener valores no nulos
            non_null_values = df[col].dropna()
            
            if len(non_null_values) > 0:
                # Convertir a string y calcular estadísticas
                text_values = non_null_values.astype(str)
                avg_length = text_values.str.len().mean()
                unique_ratio = df[col].nunique() / len(non_null_values)
                unique_count = df[col].nunique()
                
                # Criterios ajustados para detectar preguntas abiertas
                is_open_question = (
                    avg_length > 10 and  # Longitud promedio mayor a 10 caracteres
                    unique_count > 100  # Más de 100 respuestas diferentes (indica diversidad)
                ) or (
                    avg_length > 15 and  # O longitud promedio mayor a 15
                    unique_ratio > 0.01  # Con al menos 1% de respuestas únicas
                )
                
                # Verificar que no sean códigos numéricos simples
                sample_values = text_values.head(20).tolist()
                numeric_pattern = all(val.replace('.', '').replace('-', '').isdigit() 
                                    for val in sample_values if len(val) < 5)
                
                if is_open_question and not numeric_pattern:
                    open_questions.append(col)
                    logger.info(f"Pregunta abierta identificada: {col}")
                    logger.info(f"  - Longitud promedio: {avg_length:.1f}")
                    logger.info(f"  - Respuestas únicas: {unique_count}")
                    logger.info(f"  - Ratio de unicidad: {unique_ratio:.3f}")
                    
                    # Mostrar ejemplos de respuestas
                    examples = text_values.head(5).tolist()
                    logger.info(f"  - Ejemplos: {examples}")
    
    if not open_questions:
        logger.warning("No se identificaron preguntas abiertas con los criterios actuales.")
        logger.info("Columnas de texto disponibles:")
        for col in df.columns:
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    avg_len = non_null.astype(str).str.len().mean()
                    unique_count = df[col].nunique()
                    logger.info(f"  - {col}: {unique_count} valores únicos, longitud promedio: {avg_len:.1f}")
    
    return open_questions

def preprocess_text(text):
    """
    Preprocesa el texto para análisis
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        Texto preprocesado
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenizar manualmente en lugar de usar word_tokenize
    tokens = text.split()
    
    # Eliminar stopwords
    tokens = [word for word in tokens if word not in spanish_stopwords and len(word) > 2]
    
    return " ".join(tokens)

def cluster_responses(responses, question_name, n_clusters_range=range(2, 11)):
    """
    Agrupa las respuestas utilizando K-means
    
    Args:
        responses: Lista de respuestas a agrupar
        question_name: Nombre de la pregunta
        n_clusters_range: Rango de número de clusters a probar
        
    Returns:
        DataFrame con las respuestas y sus clusters asignados
    """
    logger.info(f"Analizando respuestas para: {question_name}")
    
    # Preprocesar respuestas
    preprocessed_responses = [preprocess_text(r) for r in responses]
    
    # Filtrar respuestas vacías después del preprocesamiento
    valid_indices = [i for i, r in enumerate(preprocessed_responses) if r.strip()]
    valid_responses = [responses[i] for i in valid_indices]
    valid_preprocessed = [preprocessed_responses[i] for i in valid_indices]
    
    if len(valid_preprocessed) < 10:
        logger.warning(f"Muy pocas respuestas válidas para {question_name}. Omitiendo análisis.")
        return None
    
    # Vectorizar respuestas
    logger.info("Vectorizando respuestas...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(valid_preprocessed)
    
    # Encontrar el número óptimo de clusters
    logger.info("Determinando número óptimo de clusters...")
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        if n_clusters >= len(valid_preprocessed):
            break
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calcular silhouette score
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        logger.info(f"Para n_clusters = {n_clusters}, el silhouette score es: {silhouette_avg}")
    
    # Seleccionar el mejor número de clusters
    if silhouette_scores:
        best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        logger.info(f"Mejor número de clusters: {best_n_clusters}")
    else:
        best_n_clusters = 3  # Valor predeterminado
        logger.info(f"Usando número predeterminado de clusters: {best_n_clusters}")
    
    # Aplicar K-means con el mejor número de clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Crear DataFrame con resultados
    result_df = pd.DataFrame({
        'Respuesta': valid_responses,
        'Respuesta_Preprocesada': valid_preprocessed,
        'Cluster': cluster_labels
    })
    
    # Visualizar clusters
    visualize_clusters(X, cluster_labels, question_name)
    
    # Extraer términos clave por cluster
    extract_cluster_keywords(result_df, vectorizer, question_name)
    
    return result_df

def visualize_clusters(X, labels, question_name):
    """
    Visualiza los clusters en 2D
    
    Args:
        X: Matriz de características
        labels: Etiquetas de cluster
        question_name: Nombre de la pregunta
    """
    # Reducir dimensionalidad para visualización
    if X.shape[1] > 2:
        if X.shape[0] > 50:
            # t-SNE para conjuntos de datos más grandes
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
            X_2d = tsne.fit_transform(X.toarray())
        else:
            # PCA para conjuntos de datos más pequeños
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X.toarray())
    else:
        X_2d = X.toarray()
    
    # Crear gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clusters para {question_name}')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    
    # Guardar gráfico
    output_dir = os.path.join('outputs', 'survey_analysis')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'clusters_{question_name.replace(" ", "_")}.png'))
    plt.close()

def extract_cluster_keywords(result_df, vectorizer, question_name):
    """
    Extrae y muestra las palabras clave para cada cluster
    
    Args:
        result_df: DataFrame con respuestas y clusters
        vectorizer: TfidfVectorizer utilizado
        question_name: Nombre de la pregunta
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Crear archivo para guardar resultados
    output_dir = os.path.join('outputs', 'survey_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'cluster_keywords_{question_name.replace(" ", "_")}.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Palabras clave por cluster para: {question_name}\n")
        f.write("="*50 + "\n\n")
        
        # Para cada cluster
        for cluster_id in sorted(result_df['Cluster'].unique()):
            cluster_responses = result_df[result_df['Cluster'] == cluster_id]['Respuesta_Preprocesada'].tolist()
            
            # Contar palabras en este cluster
            all_words = " ".join(cluster_responses).split()
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(10)
            
            # Escribir resultados
            f.write(f"Cluster {cluster_id} ({len(cluster_responses)} respuestas):\n")
            f.write("-"*40 + "\n")
            
            # Palabras clave
            f.write("Palabras clave: " + ", ".join([f"{word} ({count})" for word, count in top_words]) + "\n\n")
            
            # Ejemplos de respuestas
            f.write("Ejemplos de respuestas:\n")
            sample_responses = result_df[result_df['Cluster'] == cluster_id]['Respuesta'].sample(
                min(5, len(cluster_responses))
            ).tolist()
            
            for i, response in enumerate(sample_responses, 1):
                f.write(f"{i}. {response}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    logger.info(f"Palabras clave por cluster guardadas en: {output_file}")

def analyze_survey_responses(file_path):
    """
    Función principal para analizar respuestas de encuestas
    
    Args:
        file_path: Ruta al archivo .sav
    """
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
    output_dir = os.path.join('outputs', 'survey_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Analizar cada pregunta abierta
    all_results = {}
    for question in open_questions:
        # Obtener respuestas no nulas
        responses = df[question].dropna().tolist()
        
        if len(responses) < 10:
            logger.warning(f"Muy pocas respuestas para {question}. Omitiendo análisis.")
            continue
        
        # Agrupar respuestas
        result_df = cluster_responses(responses, question)
        
        if result_df is not None:
            all_results[question] = result_df
            
            # Guardar resultados
            result_df.to_csv(os.path.join(output_dir, f'clustered_{question.replace(" ", "_")}.csv'), index=False)
    
    # Generar informe de resumen
    generate_summary_report(all_results, output_dir)
    
    logger.info(f"Análisis completado. Resultados guardados en: {output_dir}")

def generate_summary_report(all_results, output_dir):
    """
    Genera un informe de resumen del análisis
    
    Args:
        all_results: Diccionario con resultados por pregunta
        output_dir: Directorio de salida
    """
    report_path = os.path.join(output_dir, 'resumen_analisis.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE ANÁLISIS DE RESPUESTAS ABIERTAS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de preguntas analizadas: {len(all_results)}\n\n")
        
        for i, (question, result_df) in enumerate(all_results.items(), 1):
            f.write(f"{i}. Pregunta: {question}\n")
            f.write("-"*40 + "\n")
            
            # Estadísticas básicas
            f.write(f"   - Total de respuestas: {len(result_df)}\n")
            f.write(f"   - Número de clusters: {result_df['Cluster'].nunique()}\n")
            
            # Distribución de clusters
            f.write("   - Distribución de clusters:\n")
            cluster_counts = result_df['Cluster'].value_counts().sort_index()
            for cluster_id, count in cluster_counts.items():
                percentage = 100 * count / len(result_df)
                f.write(f"     * Cluster {cluster_id}: {count} respuestas ({percentage:.1f}%)\n")
            
            f.write("\n")
    
    logger.info(f"Informe de resumen generado: {report_path}")

def main():
    """Función principal"""
    # Ruta al archivo .sav
    file_path = r"C:\Users\matio\Documents\PyProyects\langchain_intro-main\text-analyzer\data\encuestas\acum_LLM.sav"
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        logger.error(f"El archivo no existe: {file_path}")
        return 1
    
    # Analizar respuestas
    analyze_survey_responses(file_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
