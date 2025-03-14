import json
import pandas as pd
from create_tree_structure import create_tree_from_excels
from prompt_generation import generate_prompts

# Custom function to handle timestamp serialization
def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj

def main():
    # Define the file paths for Excel files
    file_paths = {
        "articles": "data/articles.xlsx",
        "materials": "data/materials.xlsx",
        "production_orders": "data/production_orders.xlsx",
        "work_orders": "data/work_orders.xlsx"
    }

    # Generate the tree structure
    tree = create_tree_from_excels(file_paths)

    # Convert and save the JSON tree structure
    tree_serialized = convert_timestamps(tree)

    with open("tree_structure.json", "w") as f:
        json.dump(tree_serialized, f, indent=4)

    print("Tree structure successfully saved to 'tree_structure.json'.")

    test_data = generate_prompts()
    with open("test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    print("Test dataset has been generated and saved as test_dataset.json.")

# Run the script only if executed directly
if __name__ == "__main__":
    main()
