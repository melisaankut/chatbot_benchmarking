import pandas as pd
import random

def generate_questions_from_excel(file_paths, output_csv="generated_QA_pairs.csv"):
    """
    Generates question-answer pairs from multiple Excel files and saves them to a CSV.

    Parameters:
    ----------
    file_paths (list): List of Excel file paths.
    output_csv (str): Output CSV file to save the generated Q&A pairs.

    Returns:
    ----------
    DataFrame containing the generated Q&A pairs.
    """
    dataframes = {}

    # Read all sheets from each file
    for file in file_paths:
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            dataframes[f"{file} - {sheet}"] = xls.parse(sheet)

    # Function to generate Q&A
    def generate_questions(df, table_name):
        questions_answers = []
        
        for column in df.columns:
            sample_values = df[column].dropna().sample(min(5, len(df)), random_state=42)  # Sample up to 5 values
            for value in sample_values:
                q_templates = [
                    f"What is the {column} of a specific record in {table_name}?",
                    f"Find the {column} for a given entry in {table_name}.",
                    f"Tell me the {column} associated with a record from {table_name}.",
                    f"What is the value of {column} when searching in {table_name}?"
                ]
                question = random.choice(q_templates)
                answer = str(value)
                questions_answers.append((question, answer))
        
        return questions_answers

    # Generate Q&A from all tables
    qa_pairs = []
    for table_name, df in dataframes.items():
        qa_pairs.extend(generate_questions(df, table_name))

    # Convert to DataFrame and save
    qa_df = pd.DataFrame(qa_pairs, columns=["Question", "Answer"])
    qa_df.to_csv(output_csv, index=False)

    return qa_df
