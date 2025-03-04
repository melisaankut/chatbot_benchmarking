import pandas as pd
import json

def create_tree_from_excels(files_dict):
    """
    Creates a hierarchical tree structure from multiple Excel files based on common keys.

    Parameters:
    ----------
    files_dict : dict
        A dictionary containing Excel file names as values with corresponding table names as keys.
        Example:
        {
            "articles": "articles.xlsx",
            "materials": "materials.xlsx",
            "production_orders": "production_orders.xlsx",
            "work_orders": "work_orders.xlsx"
        }

    Returns:
    -------
    dict
        A nested dictionary representing the tree structure where:
        - `orderid` is the main key that connects `production_orders`, `work_orders`, and `materials`.
        - `articlenumber` from `production_orders` links to `productid` from `articles`.
    """
    
    # Load the Excel files 
    files = files_dict 

    # Load all sheets into dataframes
    data = {name: pd.ExcelFile(path) for name, path in files.items()}
    dfs = {name: xls.parse(xls.sheet_names[0]) for name, xls in data.items()}  # Assuming first sheet in each file

    # Tree structure initialization
    tree = {}

    # Step 1: Create the Production Orderstree
    for _, row in dfs["production_orders"].iterrows():
        orderid = row["orderid"]
        if orderid not in tree:
            tree[orderid] = {"production_orders": []}
        tree[orderid]["production_orders"].append(row.to_dict())


    # Step 2: Add Work Orders data
    for _, row in dfs["work_orders"].iterrows():
        orderid = row["orderid"]
        if orderid not in tree:
            tree[orderid] = {"work_orders": []}
        if "work_orders" not in tree[orderid]:
            tree[orderid]["work_orders"] = []  # Ensure the key exists
        tree[orderid]["work_orders"].append(row.to_dict())

    # Step 3: Add Materials data
    for _, row in dfs["materials"].iterrows():
        orderid = row["orderid"]
        if orderid not in tree:
            tree[orderid] = {"materials": []}
        if "materials" not in tree[orderid]:
            tree[orderid]["materials"] = []
        tree[orderid]["materials"].append(row.to_dict())

    # Step 4: Add Articles data based on articlenumber-productid
    for _, row in dfs["articles"].iterrows():
        product_id = row["productid"]
        for order in tree.values():
            for prod_order in order.get("production_orders", []):
                if prod_order["articelnumber"] == product_id:
                    if "articles" not in prod_order:
                        prod_order["articles"] = []
                    prod_order["articles"].append(row.to_dict())

    return tree

