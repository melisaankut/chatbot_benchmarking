import json
import pandas as pd
import matplotlib.pyplot as plt
from measure_correctness import evaluate_dataset, print_average_scores, plot_scores
import os

def load_test_dataset(json_path):
    """Load the test dataset from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_json_and_csv(json_data, csv_path, question_type):
    """Merge test dataset and chatbot responses for a given question type."""
    df = pd.read_csv(csv_path)
    # Soru tipine göre eşleştirme
    lookup = dict(zip(df['sentence'], df['response']))
    merged = []
    for item in json_data:
        question = item[question_type]
        reference = item['answer']
        generated = lookup.get(question)
        if pd.notna(reference) and pd.notna(generated):
            merged.append({
                "reference": str(reference).strip(),
                "generated": str(generated).strip()
            })
    return merged

def evaluate_and_save(json_data, question_type, csv_path, results_dir):
    data = merge_json_and_csv(json_data, csv_path, question_type)
    results_df = evaluate_dataset(data)
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"results_{question_type}.csv")
    results_df.to_csv(out_csv, index=False)
    plot_scores(results_df)
    plt_path = os.path.join(results_dir, f'scores_{question_type}.png')
    plt.savefig(plt_path)
    plt.close()
    print(f"\nResults for {question_type}:")
    print_average_scores(results_df)
    return out_csv, plt_path

def main():
    dataset = load_test_dataset('test_dataset.json')
    question_types = [
        ('question', 'output/question.csv'),
        ('german_question', 'output/german_question.csv'),
        ('novelty_handling', 'output/novelty_handling.csv'),
        ('question_typo', 'output/question_typo.csv'),
    ]
    results_dir = 'results/question_types'
    for qtype, csv_path in question_types:
        print(f"\nEvaluating {qtype}...")
        evaluate_and_save(dataset, qtype, csv_path, results_dir)

if __name__ == "__main__":
    main() 