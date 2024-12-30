import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_text_generation_metrics(generated_reports, reference_reports):
    """
    Calculates BLEU-1..4 and ROUGE-1,2,L (precision, recall, f1) across
    all generated vs. reference reports.
    """
    import warnings
    warnings.filterwarnings('ignore')

    metrics = {
        'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
        'rouge_1_p': [], 'rouge_1_r': [], 'rouge_1_f': [],
        'rouge_2_p': [], 'rouge_2_r': [], 'rouge_2_f': [],
        'rouge_l_p': [], 'rouge_l_r': [], 'rouge_l_f': [],
        'avg_length': []
    }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1
    
    for gen_report, ref_report in zip(generated_reports, reference_reports):
        if not gen_report.strip() or not ref_report.strip():
            continue

        metrics['avg_length'].append(len(gen_report.split()))
        
        reference = nltk.word_tokenize(ref_report.lower())
        hypothesis = nltk.word_tokenize(gen_report.lower())
        
        try:
            metrics['bleu_1'].append(sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_2'].append(sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn))
            metrics['bleu_3'].append(sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn))
            metrics['bleu_4'].append(sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn))
        except Exception as e:
            print(f"Error calculating BLEU scores: {str(e)}")

        try:
            scores = scorer.score(ref_report, gen_report)
            
            metrics['rouge_1_p'].append(scores['rouge1'].precision)
            metrics['rouge_1_r'].append(scores['rouge1'].recall)
            metrics['rouge_1_f'].append(scores['rouge1'].fmeasure)
            
            metrics['rouge_2_p'].append(scores['rouge2'].precision)
            metrics['rouge_2_r'].append(scores['rouge2'].recall)
            metrics['rouge_2_f'].append(scores['rouge2'].fmeasure)
            
            metrics['rouge_l_p'].append(scores['rougeL'].precision)
            metrics['rouge_l_r'].append(scores['rougeL'].recall)
            metrics['rouge_l_f'].append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
    
    final_metrics = {}
    for key in metrics:
        if metrics[key]:
            final_metrics[key] = np.mean(metrics[key])
    return final_metrics
