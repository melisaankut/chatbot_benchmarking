import json
import pandas as pd
from create_tree_structure import generate_tree_structure

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
        "articles": "articles.xlsx",
        "materials": "materials.xlsx",
        "production_orders": "production_orders.xlsx",
        "work_orders": "work_orders.xlsx"
    }

    # Generate the tree structure
    tree = generate_tree_structure(file_paths)

    # Convert and save the JSON tree structure
    tree_serialized = convert_timestamps(tree)

    with open("tree_structure.json", "w") as f:
        json.dump(tree_serialized, f, indent=4)

    print("Tree structure successfully saved to 'tree_structure.json'.")

# Run the script only if executed directly
if __name__ == "__main__":
    main()
