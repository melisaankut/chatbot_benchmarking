import pandas as pd
import json
from collections import defaultdict

def generate_prompts():
    """Generates a dataset of questions and answers based on the JSON structure and saves it as CSV and JSON."""

    # Load the JSON data
    with open("tree_structure.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # List of question templates
    question_templates = [
        "At which workstation is order {ordername} currently?",
        "What is the stock level of article {ordername}?",
        "Where is article {ordername} stored?",
        "What is the status of order {ordername}?",
        "Which articles do I need for order {ordername}?",
        "Which materials do I need for order {ordername}?",
        "What is the stock level of material {materialname}?",
        "Where is material {materialname} stored?",
        "Has the material for order {ordername} been ordered?",
        "Has the material for {orderid} already been ordered?",
        "Give me the info about the order {ordername}.",
        "How many orders with the drawing number {drawno} are in the ERP?",
        "Give me the newest order with the drawing number {drawno}.",
        "What is the status of order {orderid}?",
        "When was the first order with this drawing number {drawno} placed?",
        "How far is the order {orderid} in the ERP system?",
        "What are the working steps for order {orderid}?",
        "Where is the article {articelnumber} stored?",
        "What is currently in warehouse position {warehousepos}?",
        "What does the article {articelnumber} cost?",
        "How often was the article {articelnumber} delivered in 2024?",
        "Give me all information about the working step of {drawno} from the ERP.",
        "Which articles contain the description {keyword} in the ERP?",
        "How many confirmations of {articelnumber} are currently open?",
        "How many warehouse orders from {orderid} are still open?",
        "How many times was article {articelnumber} delivered after 01.01.2024?",
        "List all warehouse positions and the number of articles stored there.",
        "Which warehouse position contains the most different articles?",
        "Sort all warehouse positions by the number of different articles stored."
    ]

    # Initialize the test dataset list
    test_data = []
    warehouse_article_counts = defaultdict(set)  # Using set to ensure unique articles per warehouse
    drawno_count = {}  # Dictionary to count orders by drawing number
    latest_orders = {}  # Dictionary to track newest order per drawing number

    # Iterate through orders in the JSON data
    for order_id, order_data in data.items():
        for production_order in order_data.get("production_orders", []):  # Ensure "production_orders" exists
            for article in production_order.get("articles", []):
                warehouse_pos = article.get("warehousepos", "Unknown")
                article_id = article.get("articelnumber", "Unknown")
                warehouse_article_counts[warehouse_pos].add(article_id)
            

            order_name = production_order.get("ordername", "Not Available")
            order_status = production_order.get("orderstatus", "Not Available")
            order_description = production_order.get("orderdesc", "Not Available")
            order_quantity = production_order.get("productionquantity", "Not Available")
            order_price = production_order.get("price", "Not Available")
            order_workstation = production_order.get("currentworkstation", "Not Available")
            order_warehousepos = production_order.get("warehousepos", "Not Available")
            order_deliverydate = production_order.get("deliverydate", "Not Available")
            order_companyid = production_order.get("companyid", "Not Available")
            order_createdate = production_order.get("createdate", "2000-01-01T00:00:00")  # Default old date
            order_drawno = production_order.get("drawno", "Unknown")
            # Extract materials and articles; use an empty list if the key is missing
            materials = [mat for mat in order_data.get("materials", [])]
            articles = [art for art in production_order.get("articles", [])]
            
            # Generate corresponding question-answer pairs
            for key in (order_name, order_id):
                test_data.append({"question": f"Give me the info about the order {{{key}}}.", "answer": order_description})
                test_data.append({"question": f"Has the material for order {{{key}}} been ordered?", "answer": "Yes" if materials else "No"})
                test_data.append({"question": f"Has the material for {{{key}}} already been ordered?", "answer": "Yes" if materials else "No"})
                test_data.append({"question": f"How far is the order {{{key}}} in the ERP system?", "answer": f"From {order_createdate}" if order_workstation else "Not Started"})
                test_data.append({"question": f"At which workstation is order {{{key}}} currently?", "answer": order_workstation})
                test_data.append({"question": f"What is the number of orders for {{{key}}}?", "answer": order_quantity})
                test_data.append({"question": f"What does the order {{{key}}} cost?", "answer": order_price})
                test_data.append({"question": f"What is the drawing no for {{{key}}}?", "answer": order_drawno})
                test_data.append({"question": f"When is the delivery date for order {{{key}}}?", "answer": order_deliverydate})
                test_data.append({"question": f"What is the company ID for order {{{key}}}?", "answer": order_companyid})
                test_data.append({"question": f"Where is order {{{key}}} stored?", "answer": order_warehousepos})
                test_data.append({"question": f"What is the status of order {{{key}}}?", "answer": order_status})
                test_data.append({"question": f"Which articles do I need for order {{{key}}}?", "answer": ','.join(art["productname"] for art in articles) if articles else "No Articles"})
                test_data.append({"question": f"Which materials do I need for order {{{key}}}?", "answer": ','.join(mat["materialname"] for mat in materials) if materials else "No Materials"})
                

        # Iterate through materials to generate additional question-answer pairs
        for material in order_data.get("materials", []):
            materialname = material.get("materialname", "Not Available")
            stocklevel = material.get("stockquantity", "Not Available")
            location = material.get("materialwarehousepos", "Not Available")

            test_data.append({"question": f"What is the stock level of material {materialname}?", "answer": stocklevel})
            test_data.append({"question": f"Where is material {materialname} stored?", "answer": location})

     # Convert set lengths to actual counts
    warehouse_article_counts = {key: len(value) for key, value in warehouse_article_counts.items()}
    # Convert to a sorted list
    sorted_warehouse_positions = sorted(warehouse_article_counts.items(), key=lambda x: x[1], reverse=True)
    # Get warehouse with most different articles
    most_diverse_warehouse = sorted_warehouse_positions[0] if sorted_warehouse_positions else ("None", 0)
    
    lst_warehouse_count = []
    for warehouse, count in sorted_warehouse_positions:
        lst_warehouse_count.append(f"{warehouse}: {count}")
    
    lst_sorted_warehouse = []   
    for warehouse, count in sorted_warehouse_positions:
        lst_sorted_warehouse.append(f"{warehouse}: {count}")
        
    test_data.append({"question": f"List all warehouse positions and the number of articles stored there.","answer": lst_warehouse_count})
    test_data.append({"question": f"Which warehouse position contains the most different articles?", "answer": f"Warehouse: {most_diverse_warehouse[0]} + Unique Articles: {most_diverse_warehouse[1]}"})
    test_data.append({"question": f"Sort all warehouse positions by the number of different articles stored.", "answer": lst_sorted_warehouse})
    
    
    # Add aggregated questions after looping through all orders
    for drawno, count in drawno_count.items():
        test_data.append({"question": f"How many orders with the drawing number {drawno} are in the ERP?", "answer": count})

    for drawno, order in latest_orders.items():
        ordername = order.get("ordername", "Unknown")
        test_data.append({"question": f"Give me the newest order with the drawing number {drawno}.", "answer": ordername})
    
    # Save as JSON
    return test_data
    