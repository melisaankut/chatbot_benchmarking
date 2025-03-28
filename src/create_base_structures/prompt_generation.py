import pandas as pd
import json
from collections import defaultdict
import random
import nlpaug.augmenter.char as nac
import re

# Initialize the typo augmenter
typo_aug = nac.KeyboardAug(
    aug_char_min=1,
    aug_char_max=3,
    aug_char_p=0.2,
    include_special_char=False,
    include_numeric=False,
    include_upper_case=False
)


def generate_typo_question_safe(original_question):
    if not isinstance(original_question, str):
        raise TypeError(f"Expected string, got {type(original_question)}")

    # 1. Find all placeholders: {ID}, {orderid}, etc.
    placeholders = re.findall(r"\{.*?\}", original_question)
    tokens = [f"MASK{i:05d}" for i in range(len(placeholders))]
    
    # 2. Replace placeholders with tokens
    temp_question = original_question
    for token, placeholder in zip(tokens, placeholders):
        temp_question = temp_question.replace(placeholder, token)

    # 3. Apply typo
    typo_temp_question = typo_aug.augment(temp_question)
    if isinstance(typo_temp_question, list):
        typo_temp_question = typo_temp_question[0]

    # 4. Replace tokens back via regex (in order)
    def restore_placeholders(match):
        index = int(match.group(0)[4:])
        return placeholders[index]

    typo_temp_question = re.sub(r"MASK\d{5}", restore_placeholders, typo_temp_question)
    
    return typo_temp_question



def generate_qa_pairs_from_tree():
    test_data = []
    with open("tree_structure.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for order_id, order_data in data.items():
        if random.random() > 0.5:
            continue
        for production_order in order_data.get("production_orders", []):
            order_name = production_order.get("ordername", "Not Available")
            order_status = production_order.get("orderstatus", "Not Available")
            order_description = production_order.get("orderdesc", "Not Available")
            order_quantity = production_order.get("productionquantity", "Not Available")
            order_price = production_order.get("price", "Not Available")
            order_workstation = production_order.get("currentworkstation", "Not Available")
            order_warehousepos = production_order.get("warehousepos", "Not Available")
            order_deliverydate = production_order.get("deliverydate", "Not Available")
            order_companyid = production_order.get("companyid", "Not Available")
            order_createdate = production_order.get("createdate", "2000-01-01T00:00:00")
            order_drawno = production_order.get("drawno", "Unknown")
            materials = [mat for mat in order_data.get("materials", [])]
            articles = [art for art in production_order.get("articles", [])]

            for key in (order_name, order_id):
                questions = [
                    (f"Give me the info about the order {{{key}}}.", order_description),
                    (f"Has the material for {{{key}}} already been ordered?", "Yes" if materials else "No"),
                    (f"How far is the order {{{key}}} in the ERP system?", f"From {order_createdate}" if order_workstation else "Not Started"),
                    (f"At which workstation is order {{{key}}} currently?", order_workstation),
                    (f"What is the number of orders for {{{key}}}?", order_quantity),
                    (f"What does the order {{{key}}} cost?", order_price),
                    (f"What is the drawing no for {{{key}}}?", order_drawno),
                    (f"When is the delivery date for order {{{key}}}?", order_deliverydate),
                    (f"What is the company ID for order {{{key}}}?", order_companyid),
                    (f"Where is order {{{key}}} stored?", order_warehousepos),
                    (f"What is the status of order {{{key}}}?", order_status),
                    (f"Which articles do I need for order {{{key}}}?", ','.join(art["productname"] for art in articles) if articles else "No Articles"),
                    (f"Which materials do I need for order {{{key}}}?", ','.join(mat["materialname"] for mat in materials) if materials else "No Materials")
                ]

                for q, a in questions:
                    if random.random() > 0.1:
                        pass
                    test_data.append({
                        "question": q,
                        "question_typo": generate_typo_question_safe(q),
                        "answer": a
                    })

    return test_data




def generate_qa_pairs_from_database():
    articles_df = pd.read_excel(r"data\articles.xlsx")
    materials_df = pd.read_excel(r"data\materials.xlsx")
    production_orders_df = pd.read_excel(r"data\production_orders.xlsx")

    test_data = []
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

        q1 = f"What is the stock level of material {mat_name}?"
        q2 = f"Where is material {mat_name} stored?"
        test_data.extend([
            {"question": q1, "question_typo": generate_typo_question_safe(q1), "answer": stock},
            {"question": q2, "question_typo": generate_typo_question_safe(q2), "answer": location}
        ])

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
        q = f"How many orders with the drawing number {drawno} are in the ERP?"
        test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": str(count)})

    for drawno, order in latest_orders.items():
        if random.random() > 0.5:
            continue
        q = f"Give me the newest order with the drawing number {drawno}."
        test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": order['ordername']})

    for _, row in production_orders_df.iterrows():
        if random.random() > 0.5:
            continue
        orderid = row['orderid']
        orderstatus = row.get('orderstatus', 'Unknown')
        drawno = row.get('drawno', None)
        createdate = row.get('createdate', None)

        q = f"How far is the order {orderid} in the ERP system?"
        test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": orderstatus})

        if drawno and pd.notna(createdate):
            q2 = f"When was the first order with this drawing number {drawno} placed?"
            test_data.append({"question": q2, "question_typo": generate_typo_question_safe(q2), "answer": createdate.strftime('%Y-%m-%d')})

    for pos in materials_df['materialwarehousepos'].dropna().unique():
        if random.random() > 0.5:
            continue
        materials = materials_df[materials_df['materialwarehousepos'] == pos]['materialname'].dropna().unique()
        if len(materials) > 0:
            material_list = ", ".join(materials)
            q = f"What is currently in warehouse position {pos}?"
            a = f"The following materials are stored in warehouse position {pos}: {material_list}."
            test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": a})

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

    q1 = "List all warehouse positions and the number of articles stored there."
    q2 = "Which warehouse position contains the most different articles?"
    q3 = "Sort all warehouse positions by the number of different articles stored."

    test_data.append({"question": q1, "question_typo": generate_typo_question_safe(q1), "answer": json.dumps(lst_warehouse_count, ensure_ascii=False)})
    test_data.append({"question": q2, "question_typo": generate_typo_question_safe(q2), "answer": f"Warehouse: {most_diverse_warehouse} + Unique Articles: {most_diverse_count}"})
    test_data.append({"question": q3, "question_typo": generate_typo_question_safe(q3), "answer": json.dumps(lst_sorted_warehouse, ensure_ascii=False)})

    date_threshold = pd.to_datetime("2024-01-01", errors='coerce')

    if 'articlelnumber' in production_orders_df.columns:
        for articlenumber in production_orders_df['articlelnumber'].dropna().unique():
            if random.random() > 0.5:
                pass
            filtered = production_orders_df[
                (production_orders_df['articlelnumber'] == articlenumber) &
                (production_orders_df['deliverydate'] > date_threshold)
            ]
            count = len(filtered)
            q = f"How many times was article {articlenumber} delivered after 01.01.2024?"
            test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": count})

    articles_df['createdate'] = pd.to_datetime(articles_df['createdate'], errors='coerce')
    articles_df['changedate'] = pd.to_datetime(articles_df['changedate'], errors='coerce')
    production_orders_df['deliverydate'] = pd.to_datetime(production_orders_df['deliverydate'], errors='coerce')

    for _, row in articles_df.dropna(subset=['articelnumber']).iterrows():
        if random.random() > 0.5:
            continue
        articlenumber = row['articelnumber']
        location = row.get('warehousepos', 'Unknown')
        q = f"Where is the article {articlenumber} stored?"
        test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": location})

    for _, row in articles_df.dropna(subset=['articelnumber']).iterrows():
        if random.random() > 0.5:
            continue
        articlenumber = row['articelnumber']
        price = row.get('price', 'Unknown')
        q = f"What does the article {articlenumber} cost?"
        test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": price})

    delivery_2024 = production_orders_df[
        (production_orders_df['deliverydate'] >= "2024-01-01") &
        (production_orders_df['deliverydate'] < "2025-01-01")
    ]
    if 'articlelnumber' in production_orders_df.columns:
        for articlenumber in production_orders_df['articlenumber'].dropna().unique():
            if random.random() > 0.5:
                continue
            count = len(delivery_2024[delivery_2024['articlenumber'] == articlenumber])
            q = f"How often was the article {articlenumber} delivered in 2024?"
            test_data.append({"question": q, "question_typo": generate_typo_question_safe(q), "answer": count})

    return test_data
