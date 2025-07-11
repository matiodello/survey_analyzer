#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune BERT model using the refined survey dataset with 11 specific categories.
Adapted for the new dataset structure with categories like POLÍTICA, EDUCACIÓN, POBREZA, etc.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import logging
import sys
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune_bert_refined.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SurveyDataset(Dataset):
    """Dataset for survey text classification"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

class RefinedBERTTrainer:
    """Trainer for BERT with refined survey data"""
    
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_refined_data(self, data_dir="data/training_refined"):
        """Load the refined training data"""
        logger.info(f"Loading refined data from: {data_dir}")
        
        # Load datasets
        train_path = os.path.join(data_dir, "train_refined_data.csv")
        val_path = os.path.join(data_dir, "val_refined_data.csv")
        test_path = os.path.join(data_dir, "test_refined_data.csv")
        label_path = os.path.join(data_dir, "refined_label_mapping.csv")
        
        # Check if files exist
        for path in [train_path, val_path, test_path, label_path]:
            if not os.path.exists(path):
                logger.error(f"Required file not found: {path}")
                return None, None, None, None
        
        # Load data
        train_df = pd.read_csv(train_path, encoding='utf-8')
        val_df = pd.read_csv(val_path, encoding='utf-8')
        test_df = pd.read_csv(test_path, encoding='utf-8')
        label_mapping = pd.read_csv(label_path, encoding='utf-8')
        
        logger.info(f"Training data: {len(train_df)} samples")
        logger.info(f"Validation data: {len(val_df)} samples")
        logger.info(f"Test data: {len(test_df)} samples")
        
        # Show category distribution
        logger.info("\nTraining data distribution:")
        train_dist = train_df['category'].value_counts()
        for category, count in train_dist.items():
            percentage = (count / len(train_df)) * 100
            logger.info(f"  {category}: {count} samples ({percentage:.1f}%)")
        
        return train_df, val_df, test_df, label_mapping
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip().lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def prepare_data_for_training(self, train_df, val_df, test_df):
        """Prepare data for BERT training"""
        logger.info("Preparing data for BERT training...")
        
        # Preprocess texts
        train_texts = [self.preprocess_text(text) for text in train_df['text']]
        val_texts = [self.preprocess_text(text) for text in val_df['text']]
        test_texts = [self.preprocess_text(text) for text in test_df['text']]
        
        # Encode labels
        all_labels = list(train_df['category']) + list(val_df['category']) + list(test_df['category'])
        self.label_encoder.fit(all_labels)
        
        train_labels = self.label_encoder.transform(train_df['category'])
        val_labels = self.label_encoder.transform(val_df['category'])
        test_labels = self.label_encoder.transform(test_df['category'])
        
        logger.info(f"Number of unique categories: {len(self.label_encoder.classes_)}")
        logger.info(f"Categories: {list(self.label_encoder.classes_)}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize texts
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=128)
        
        # Create datasets
        train_dataset = SurveyDataset(train_encodings, train_labels)
        val_dataset = SurveyDataset(val_encodings, val_labels)
        test_dataset = SurveyDataset(test_encodings, test_labels)
        
        return train_dataset, val_dataset, test_dataset
    
    def initialize_model(self, num_labels):
        """Initialize BERT model for classification"""
        logger.info(f"Initializing model: {self.model_name}")
        logger.info(f"Number of labels: {num_labels}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Move model to device
        self.model.to(device)
        
        return self.model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir="models/bert-refined"):
        """Train the BERT model"""
        logger.info("Starting BERT training...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Disable wandb
            dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Training started...")
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        label_encoder_path = os.path.join(output_dir, "label_encoder.json")
        label_mapping = {str(i): label for i, label in enumerate(self.label_encoder.classes_)}
        with open(label_encoder_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Label encoder saved to: {label_encoder_path}")
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset, output_dir):
        """Evaluate the trained model"""
        logger.info("Evaluating model on test set...")
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_dataset)
        
        # Get predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Generate classification report
        target_names = self.label_encoder.classes_
        class_report = classification_report(y_true, y_pred, target_names=target_names)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info("Test Results:")
        logger.info(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
        logger.info(f"  F1-Score: {test_results['eval_f1']:.4f}")
        logger.info(f"  Precision: {test_results['eval_precision']:.4f}")
        logger.info(f"  Recall: {test_results['eval_recall']:.4f}")
        
        logger.info("\nDetailed Classification Report:")
        logger.info(f"\n{class_report}")
        
        # Save detailed results
        results_path = os.path.join(output_dir, "evaluation_results.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"BERT Fine-tuning Results - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: Refined Survey Data (11 categories)\n\n")
            f.write("Test Results:\n")
            f.write(f"  Accuracy: {test_results['eval_accuracy']:.4f}\n")
            f.write(f"  F1-Score: {test_results['eval_f1']:.4f}\n")
            f.write(f"  Precision: {test_results['eval_precision']:.4f}\n")
            f.write(f"  Recall: {test_results['eval_recall']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(class_report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        logger.info(f"Detailed results saved to: {results_path}")
        
        return test_results
    
    def run_full_pipeline(self, data_dir="data/training_refined", output_dir="models/bert-refined"):
        """Run the complete training pipeline"""
        logger.info("Starting BERT fine-tuning pipeline...")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        try:
            # Load data
            train_df, val_df, test_df, label_mapping = self.load_refined_data(data_dir)
            if train_df is None:
                logger.error("Failed to load data")
                return False
            
            # Prepare data
            train_dataset, val_dataset, test_dataset = self.prepare_data_for_training(train_df, val_df, test_df)
            
            # Initialize model
            num_labels = len(self.label_encoder.classes_)
            self.initialize_model(num_labels)
            
            # Train model
            trainer = self.train_model(train_dataset, val_dataset, output_dir)
            
            # Evaluate model
            self.evaluate_model(trainer, test_dataset, output_dir)
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            return False

def main():
    """Main function"""
    logger.info("Starting BERT fine-tuning with refined survey data...")
    
    # Initialize trainer
    trainer = RefinedBERTTrainer()
    
    # Run pipeline
    success = trainer.run_full_pipeline()
    
    if success:
        logger.info("Fine-tuning completed successfully!")
        return 0
    else:
        logger.error("Fine-tuning failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
