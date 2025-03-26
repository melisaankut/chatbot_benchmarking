import pandas as pd
import json
from collections import defaultdict
import random



def generate_qa_pairs_from_tree():
    """Generates a dataset of questions and answers based on the JSON structure and saves it as CSV and JSON."""
    
    test_data = []
    # Load the JSON data
    with open("tree_structure.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    
    # Iterate through orders in the JSON data
    for order_id, order_data in data.items():
        if random.random() > 0.5:
            continue  # Randomly skip some orders to reduce data
        
        for production_order in order_data.get("production_orders", []):  # Ensure "production_orders" exists
            
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
                if random.random() > 0.1:
                    continue
                test_data.append({"question": f"Give me the info about the order {{{key}}}.", "answer": order_description})
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

        
    return test_data
    

def generate_qa_pairs_from_database():

    articles_df = pd.read_excel(r"data\articles.xlsx")
    materials_df = pd.read_excel(r"data\materials.xlsx")
    production_orders_df = pd.read_excel(r"data\production_orders.xlsx")
    
    test_data = []
    """Generates a dataset of questions and answers based on the database tables and saves it as CSV and JSON."""

    # --- MATERIAL QUESTIONS ---
    materials_df['createdate'] = pd.to_datetime(materials_df['createdate'], errors='coerce')
    materials_df['changedate'] = pd.to_datetime(materials_df['changedate'], errors='coerce')
    materials_df['latestdate'] = materials_df[['changedate', 'createdate']].max(axis=1)

    latest_materials = (
        materials_df.sort_values(by='latestdate', ascending=False)
        .dropna(subset=['materialname'])
        .drop_duplicates(subset='materialname', keep='first')
    )
   
    for _, row in latest_materials.iterrows():
        if random.random() > 0.5:
            continue
        mat_name = row['materialname']
        stock = str(row.get('stockquantity', 'Unknown'))
        location = str(row.get('materialwarehousepos', 'Unknown'))

        test_data.extend([
            {"question": f"What is the stock level of material {mat_name}?", "answer": stock},
            {"question": f"Where is material {mat_name} stored?", "answer": location}
        ])

    # --- DRAWING NUMBER QUESTIONS ---
    production_orders_df['createdate'] = pd.to_datetime(production_orders_df['createdate'], errors='coerce')
    production_orders_df['deliverydate'] = pd.to_datetime(production_orders_df['deliverydate'], errors='coerce')

    drawno_count = defaultdict(int)
    latest_orders = {}

    for _, row in production_orders_df.dropna(subset=['drawno']).iterrows():
       
        drawno = row['drawno']
        createdate = row['createdate']
        ordername = row.get('ordername', 'Unknown')

        drawno_count[drawno] += 1
        if drawno not in latest_orders or createdate > latest_orders[drawno]['createdate']:
            latest_orders[drawno] = {'createdate': createdate, 'ordername': ordername}

    for drawno, count in drawno_count.items():
        if random.random() > 0.5:
            continue
        test_data.append({
            "question": f"How many orders with the drawing number {drawno} are in the ERP?",
            "answer": str(count)
        })

    for drawno, order in latest_orders.items():
        if random.random() > 0.5:
            continue
        test_data.append({
            "question": f"Give me the newest order with the drawing number {drawno}.",
            "answer": order['ordername']
        })

    # --- ORDER STATUS QUESTIONS ---
    for _, row in production_orders_df.iterrows():
        if random.random() > 0.5:
            continue
        orderid = row['orderid']
        orderstatus = row.get('orderstatus', 'Unknown')
        drawno = row.get('drawno', None)
        createdate = row.get('createdate', None)

        test_data.append({
            "question": f"How far is the order {orderid} in the ERP system?",
            "answer": orderstatus
        })

        if drawno and pd.notna(createdate):
            test_data.append({
                "question": f"When was the first order with this drawing number {drawno} placed?",
                "answer": createdate.strftime('%Y-%m-%d')
            })

    # --- WAREHOUSE POSITION QUESTIONS ---
    for pos in materials_df['materialwarehousepos'].dropna().unique():
        if random.random() > 0.5:
            continue
        materials = materials_df[materials_df['materialwarehousepos'] == pos]['materialname'].dropna().unique()
        if len(materials) > 0:
            material_list = ", ".join(materials)
            test_data.append({
                "question": f"What is currently in warehouse position {pos}?",
                "answer": f"The following materials are stored in warehouse position {pos}: {material_list}."
            })
    warehouse_counts = (
        articles_df.dropna(subset=['warehousepos', 'articelnumber'])
        .groupby('warehousepos')['articelnumber']
        .nunique()
        .sort_values(ascending=False)
    )

    lst_warehouse_count = [f"{pos}: {count}" for pos, count in warehouse_counts.items()]
    most_diverse_warehouse = warehouse_counts.idxmax() if not warehouse_counts.empty else "None"
    most_diverse_count = warehouse_counts.max() if not warehouse_counts.empty else 0
    lst_sorted_warehouse = lst_warehouse_count.copy()

    test_data.append({
        "question": "List all warehouse positions and the number of articles stored there.",
        "answer": lst_warehouse_count
    })
    test_data.append({
        "question": "Which warehouse position contains the most different articles?",
        "answer": f"Warehouse: {most_diverse_warehouse} + Unique Articles: {most_diverse_count}"
    })
    test_data.append({
        "question": "Sort all warehouse positions by the number of different articles stored.",
        "answer": lst_sorted_warehouse
    })

    # --- ARTICLE DELIVERY COUNT QUESTIONS ---
    date_threshold = pd.to_datetime("2024-01-01", errors='coerce')

    if 'articlelnumber' in production_orders_df.columns:
        for articlenumber in production_orders_df['articlelnumber'].dropna().unique():
            if random.random() > 0.5:
                continue
            filtered = production_orders_df[
                (production_orders_df['articlelnumber'] == articlenumber) &
                (production_orders_df['deliverydate'] > date_threshold)
            ]
            count = len(filtered)
            test_data.append({
                "question": f"How many times was article {articlenumber} delivered after 01.01.2024?",
                "answer": count
            })

    # Ensure datetime columns are parsed
    articles_df['createdate'] = pd.to_datetime(articles_df['createdate'], errors='coerce')
    articles_df['changedate'] = pd.to_datetime(articles_df['changedate'], errors='coerce')
    production_orders_df['deliverydate'] = pd.to_datetime(production_orders_df['deliverydate'], errors='coerce')

    # --- ARTICLE QUESTIONS ---
    for _, row in articles_df.dropna(subset=['articelnumber']).iterrows():
        if random.random() > 0.5:
            continue
        articlenumber = row['articelnumber']
        location = row.get('warehousepos', 'Unknown')
        test_data.append({
            "question": f"Where is the article {articlenumber} stored?",
            "answer": location
        })

    for _, row in articles_df.dropna(subset=['articelnumber']).iterrows():
        if random.random() > 0.5:
            continue
        articlenumber = row['articelnumber']
        price = row.get('price', 'Unknown')
        test_data.append({
            "question": f"What does the article {articlenumber} cost?",
            "answer": price
        })

    delivery_2024 = production_orders_df[
        (production_orders_df['deliverydate'] >= "2024-01-01") &
        (production_orders_df['deliverydate'] < "2025-01-01")
    ]
    if 'articlelnumber' in production_orders_df.columns:
        for articlenumber in production_orders_df['articelnumber'].dropna().unique():
            if random.random() > 0.5:
                continue
            count = len(delivery_2024[delivery_2024['articelnumber'] == articlenumber])
            test_data.append({
                "question": f"How often was the article {articlenumber} delivered in 2024?",
                "answer": count
            })

    return test_data
