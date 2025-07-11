#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the fine-tuned BERT model on comment classification.
This script compares the performance of the fine-tuned model against the traditional approach.
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluate_bert.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    """
    Load the fine-tuned model and tokenizer
    
    Args:
        model_path: Path to the fine-tuned model
        
    Returns:
        model, tokenizer, label_classes
    """
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load label classes
        label_classes_path = os.path.join(model_path, "label_classes.csv")
        if os.path.exists(label_classes_path):
            label_classes = pd.read_csv(label_classes_path)["category"].tolist()
            logger.info(f"Loaded {len(label_classes)} label classes")
        else:
            logger.error(f"Label classes file not found at {label_classes_path}")
            return None, None, None
        
        return model, tokenizer, label_classes
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        return None, None, None

def predict_with_bert(model, tokenizer, texts, label_classes, batch_size=16):
    """
    Predict categories using the fine-tuned BERT model
    
    Args:
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        texts: List of texts to classify
        label_classes: List of label classes
        batch_size: Batch size for processing
        
    Returns:
        List of predicted categories and confidences
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = []
    confidences = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting with BERT"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted class and confidence
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            batch_confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
            
            # Map to category names
            batch_predictions = [label_classes[pred_class] for pred_class in predicted_classes]
            
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
    
    return predictions, confidences

def load_test_data(file_path):
    """
    Load test data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with test data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ["Comment"]
        category_columns = ["Categoría", "Category"]
        
        # Check for comment column
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Required columns {required_columns} not found in {file_path}")
            return None
        
        # Check for category column
        category_col = None
        for col in category_columns:
            if col in df.columns:
                category_col = col
                break
        
        if category_col is None:
            logger.warning(f"No category column found in {file_path}. This file can be used for prediction but not for evaluation.")
        
        return df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, labels, output_file=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        output_file: Path to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Confusion matrix saved to {output_file}")
    else:
        plt.show()

def main():
    """
    Main function to evaluate the fine-tuned BERT model
    """
    # Set paths
    model_path = os.path.join("models", "finetuned-bert")
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "models", "finetuned-bert")
    
    if not os.path.exists(model_path):
        logger.error(f"Fine-tuned model not found at {model_path}")
        return
    
    # Load model, tokenizer, and label classes
    model, tokenizer, label_classes = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None or label_classes is None:
        logger.error("Failed to load model, tokenizer, or label classes")
        return
    
    # Set output paths
    output_path = os.path.join("outputs")
    if not os.path.exists(output_path):
        output_path = os.path.join("..", "outputs")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Try to find test data
    test_files = [
        os.path.join("data", "resultados_etiquetados.csv"),
        os.path.join("..", "data", "resultados_etiquetados.csv"),
        os.path.join("outputs", "enhanced_comment_analysis.csv"),
        os.path.join("..", "outputs", "enhanced_comment_analysis.csv")
    ]
    
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if test_file is None:
        logger.error("No test data file found")
        return
    
    logger.info(f"Loading test data from {test_file}")
    
    # Load test data
    test_df = load_test_data(test_file)
    if test_df is None:
        logger.error("Failed to load test data")
        return
    
    logger.info(f"Loaded {len(test_df)} test examples")
    
    # Get comments
    comments = test_df["Comment"].tolist()
    
    # Check if we have ground truth labels
    has_ground_truth = False
    if "Categoría" in test_df.columns:
        true_categories = test_df["Categoría"].tolist()
        has_ground_truth = True
    elif "Category" in test_df.columns:
        true_categories = test_df["Category"].tolist()
        has_ground_truth = True
    
    # Predict with BERT
    logger.info("Predicting with fine-tuned BERT model...")
    bert_predictions, bert_confidences = predict_with_bert(model, tokenizer, comments, label_classes)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "Comment": comments,
        "BERTCategory": bert_predictions,
        "BERTConfidence": bert_confidences
    })
    
    # Add ground truth if available
    if has_ground_truth:
        results_df["TrueCategory"] = true_categories
    
    # Save results
    results_file = os.path.join(output_path, "bert_evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Evaluate if ground truth is available
    if has_ground_truth:
        logger.info("Evaluating BERT model performance...")
        
        # Calculate metrics
        accuracy = accuracy_score(true_categories, bert_predictions)
        report = classification_report(true_categories, bert_predictions, output_dict=True)
        
        # Print metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(true_categories, bert_predictions))
        
        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_file = os.path.join(output_path, "bert_evaluation_metrics.csv")
        metrics_df.to_csv(metrics_file)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Plot confusion matrix
        cm_file = os.path.join(output_path, "bert_confusion_matrix.png")
        plot_confusion_matrix(true_categories, bert_predictions, label_classes, cm_file)
        
        # Calculate percentage of indeterminate comments
        indeterminate_count = sum(1 for cat in bert_predictions if cat == "Indeterminado")
        indeterminate_percentage = indeterminate_count / len(bert_predictions) * 100
        logger.info(f"Percentage of indeterminate comments: {indeterminate_percentage:.2f}%")
        
        # Compare with traditional approach if available
        if "Category" in test_df.columns and "Category" != "TrueCategory":
            trad_predictions = test_df["Category"].tolist()
            trad_accuracy = accuracy_score(true_categories, trad_predictions)
            logger.info(f"\nTraditional approach accuracy: {trad_accuracy:.4f}")
            logger.info(f"BERT model accuracy: {accuracy:.4f}")
            logger.info(f"Improvement: {(accuracy - trad_accuracy) * 100:.2f}%")
            
            # Calculate percentage of indeterminate comments in traditional approach
            trad_indeterminate = sum(1 for cat in trad_predictions if cat == "Indeterminado")
            trad_indeterminate_percentage = trad_indeterminate / len(trad_predictions) * 100
            logger.info(f"Traditional approach indeterminate comments: {trad_indeterminate_percentage:.2f}%")
            logger.info(f"BERT model indeterminate comments: {indeterminate_percentage:.2f}%")
            logger.info(f"Reduction in indeterminate comments: {trad_indeterminate_percentage - indeterminate_percentage:.2f}%")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
