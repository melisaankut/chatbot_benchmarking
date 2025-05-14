import json
import pandas as pd
import logging
import random
from create_base_structures.create_tree_structure import create_tree_from_excels
from create_base_structures.prompt_generation import generate_qa_pairs_from_tree, generate_qa_pairs_from_database, generate_balanced_test_dataset

logging.basicConfig(level=logging.INFO)


def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj

def save_json_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)
        

def main():
    # Define file paths
    file_paths = {
        "articles": "data/articles.xlsx",
        "materials": "data/materials.xlsx",
        "production_orders": "data/production_orders.xlsx",
        "work_orders": "data/work_orders.xlsx",
        "logbooks": "data/logbook.xlsx",
        "stock": "data/stock.xlsx",
    }

    output_files = {
        "tree": "tree_structure.json",
        "full_dataset": "test_dataset.json",
        "subset": "shuffled_100_samples.json"
    }

    try:
        # Step 1: Generate tree structure
        logging.info("Generating tree structure...")
        tree = create_tree_from_excels(file_paths)
        tree_serialized = convert_timestamps(tree)
        save_json_to_file(tree_serialized, output_files["tree"])
        logging.info(f"Tree structure successfully saved to '{output_files['tree']}'")

        # Step 2: Generate balanced dataset
        logging.info("Generating balanced test dataset...")
        balanced_dataset = generate_balanced_test_dataset()
        save_json_to_file(balanced_dataset, output_files["full_dataset"])
        logging.info(f"Full test dataset saved to '{output_files['full_dataset']}'")

        # Step 3: Create subset of 100 samples
        logging.info("Creating subset of 100 samples...")
        random.shuffle(balanced_dataset)
        subset = balanced_dataset[:100]
        save_json_to_file(subset, output_files["subset"])
        logging.info(f"Subset of 100 samples saved to '{output_files['subset']}'")

        # Log dataset statistics
        logging.info(f"Total questions in balanced dataset: {len(balanced_dataset)}")
        logging.info(f"Questions in subset: {len(subset)}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
