#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Analyzer - Análisis de Encuestas con BERT
===============================================

Sistema modular para análisis de texto en español, especializado en:
- Análisis de respuestas abiertas de encuestas
- Clasificación automática con BERT fine-tuneado
- Fine-tuning de modelos BERT personalizados

Autor: Proyecto Text Analyzer
Fecha: 2025
"""

import sys
import os
import argparse

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Función principal del sistema"""
    parser = argparse.ArgumentParser(description='Text Analyzer - Sistema de Análisis de Encuestas')
    parser.add_argument('--mode', choices=['analyze', 'train', 'evaluate'], 
                       required=True, help='Modo de operación')
    parser.add_argument('--file', help='Archivo de encuesta a analizar')
    parser.add_argument('--model', default='models/bert-expanded', 
                       help='Ruta al modelo BERT a utilizar')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        if not args.file:
            print("Error: Se requiere especificar un archivo con --file")
            return 1
        
        from analyze_survey_bert_expanded import analyze_survey_with_bert_expanded
        print(f"Analizando archivo: {args.file}")
        analyze_survey_with_bert_expanded(args.file, args.model)
        
    elif args.mode == 'train':
        from finetune_bert_expanded import main as train_main
        print("Iniciando fine-tuning de BERT...")
        train_main()
        
    elif args.mode == 'evaluate':
        from evaluate_bert import main as eval_main
        print("Evaluando modelo BERT...")
        eval_main()
    
    print("Proceso completado exitosamente!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
