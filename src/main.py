import json
import pandas as pd
import logging
import random
from create_base_structures.create_tree_structure import create_tree_from_excels
from create_base_structures.prompt_generation import generate_qa_pairs_from_tree, generate_qa_pairs_from_database

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
        json.dump(data, f, indent=4)

def main():
    file_paths = {
        "articles": "data/articles.xlsx",
        "materials": "data/materials.xlsx",
        "production_orders": "data/production_orders.xlsx",
        "work_orders": "data/work_orders.xlsx"
    }

    try:
        tree = create_tree_from_excels(file_paths)
        tree_serialized = convert_timestamps(tree)
        save_json_to_file(tree_serialized, "tree_structure.json")
        logging.info("Tree structure successfully saved to 'tree_structure.json'.")
    except Exception as e:
        logging.error(f"Failed to generate tree structure: {e}")
        return

    try:
        test_data = generate_qa_pairs_from_tree() + generate_qa_pairs_from_database()
        save_json_to_file(test_data, "test_dataset.json")
        logging.info("Test dataset has been generated and saved as 'test_dataset.json'.")
    except Exception as e:
        logging.error(f"Failed to generate test dataset: {e}")


        # Load the dataset
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle
    random.shuffle(data)

    # Select 1000 samples
    subset = data[:100]

    # Save to a new JSON file
    with open('shuffled_100_samples.json', 'w', encoding='utf-8') as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
