import pandas as pd
import matplotlib.pyplot as plt
import json

def transform_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_data.append({
            "question": item["question_typo"],
            "answer": item["answer"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

#transform_json("shuffled_100_samples.json", "shuffled_100_samples_just_typo.json")

def evaluate_chatbot_responses(original_csv, typo_csv):
    # Load both CSV files
    df_original = pd.read_csv(original_csv)
    df_typo = pd.read_csv(typo_csv)

    # Normalize the responses (strip spaces and lowercase for fair comparison)
    df_original['response_norm'] = df_original['response'].astype(str).str.strip().str.lower()
    df_typo['response_norm'] = df_typo['response'].astype(str).str.strip().str.lower()

    # Compare corresponding rows by index
    df_result = pd.DataFrame({
        "original_question": df_original['sentence'],
        "typo_question": df_typo['sentence'],
        "original_response": df_original['response'],
        "typo_response": df_typo['response'],
        "match": df_original['response_norm'] == df_typo['response_norm']
    })

    # Calculate accuracy
    total = len(df_result)
    correct = df_result['match'].sum()
    incorrect = total - correct
    accuracy = (correct / total) * 100

    print(f"Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:.2f}%")

    # Plot pie chart of matching results
    plt.figure(figsize=(6,6))
    plt.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', startangle=90)
    plt.title('Typo vs Original Question Response Accuracy')
    plt.axis('equal')
    plt.show()

    # Optional: Save the comparison results to a CSV
    df_result.to_csv("comparison_result.csv", index=False)

    return df_result


evaluate_chatbot_responses("response_texts_normal.csv", "response_texts_typo.csv")