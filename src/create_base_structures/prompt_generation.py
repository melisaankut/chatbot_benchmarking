import numpy as np
import pandas as pd
import json
from collections import defaultdict
import random
import nlpaug.augmenter.char as nac
import re
from transformers import pipeline
from typing import Tuple
import logging
from time import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os


SEED = 42

random.seed(SEED)
np.random.seed(SEED)


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


os.environ["PYTHONHASHSEED"] = str(SEED)


MODEL = "prithivida/parrot_paraphraser_on_T5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Constants ---
MAX_PER_TEMPLATE = 1     # per-template sample size
MAX_TOTAL = 10          # global QA limit

# Placeholder for RECORD_TYPES - adjust with your actual types/schemas
RECORD_TYPES = {
    "work_order": "work_order_schema",  # Replace with actual schema or identifier
    "logbook": "logbook_schema",        # Replace with actual schema or identifier
    "production_order": "production_order_schema" # Replace with actual schema or identifier
    # Add other record types as needed
}

# Category weights for balancing
CATEGORY_WEIGHTS = {
    # Basic Information Categories (Weight: 1.0)
    "order_info": 1.0,
    "order_description": 1.0,
    "order_status": 1.0,
    "order_progress": 1.0,
    "company_id": 1.0,
    
    # Material and Stock Categories (Weight: 1.2)
    "material_order_status": 1.2,
    "material_stock_level": 1.2,
    "material_storage": 1.2,
    "article_stock_quantity": 1.2,
    "article_storage": 1.2,
    "article_available_stock": 1.2,
    
    # Production Categories (Weight: 1.3)
    "order_workstation": 1.3,
    "order_quantity": 1.3,
    "work_order_time": 1.3,
    "work_order_stations": 1.3,
    
    # Cost and Price Categories (Weight: 1.4)
    "order_price": 1.4,
    "article_cost": 1.4,
    
    # Delivery and Planning Categories (Weight: 1.5)
    "delivery_date": 1.5,
    "delivery_post_threshold": 1.5,
    "delivery_2024_frequency": 1.5,
    
    # Drawing and Documentation Categories (Weight: 1.2)
    "drawing_number": 1.2,
    "drawing_order_count": 1.2,
    "drawing_newest_order": 1.2,
    "drawing_first_order": 1.2,
    
    # Warehouse Categories (Weight: 1.3)
    "warehouse_contents": 1.3,
    "warehouse_top10": 1.3,
    "warehouse_most_diverse": 1.3,
    "warehouse_sorted": 1.3,
    
    # Logbook Categories (Weight: 1.1)
    "logbook_reporter": 1.1,
    "logbook_status": 1.1,
    
    # Analytical Categories (Weight: 1.6)
    "process_flow": 1.6,
    "resource_optimization": 1.6,
    "quality_metrics": 1.6,
    "cost_efficiency": 1.6,
    "inventory_management": 1.6,
    "production_planning": 1.6,
    "cross_reference": 1.6,
    "performance_metrics": 1.6
}


def safe_lower(text):
    """Safely convert text to lowercase, handling None and float values."""
    if text is None:
        return ""
    if isinstance(text, (float, int)):
        return ""
    return str(text).lower()
def save_json_to_file(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


def validate_record(record: dict, record_type: str) -> Tuple[bool, dict]:
    """
    Validates a record against a given type/schema.
    Returns (is_valid, validated_record_or_error).
    """
    required_fields = {
        RECORD_TYPES["work_order"]: ["workstation", "time"],  # Add more as needed
        RECORD_TYPES["logbook"]: ["status", "employee"],      # Add more as needed
        RECORD_TYPES["production_order"]: ["ordername", "price", "createdate", "deliverydate"],  # Add more as needed
    }

    # Check if record_type is known
    if record_type not in required_fields:
        return False, {"error": f"Unknown record_type: {record_type}"}

    missing = []
    for field in required_fields[record_type]:
        if field not in record or record[field] in [None, ""]:
            missing.append(field)

    if missing:
        return False, {"error": f"Missing required fields: {missing}", "record": record}

    return True, record

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info("Loading translation pipelines...")
start = time()

# Initialize the typo augmenter
typo_aug = nac.KeyboardAug(
    aug_char_min=1,
    aug_char_max=3,
    aug_char_p=0.2,
    include_special_char=False,
    include_numeric=False,
    include_upper_case=False
)

# Initialize translation pipelines
en_to_de_pipeline = pipeline(
    "translation",
    model="facebook/m2m100_418M",
    tokenizer="facebook/m2m100_418M",
    src_lang="en",
    tgt_lang="de",
    device=0
)

de_to_en_pipeline = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    tokenizer="facebook/nllb-200-distilled-600M",
    src_lang="deu_Latn",   # ISO code for German
    tgt_lang="eng_Latn",   # ISO code for English
    device=0               # set to -1 if using CPU
)

# Define a glossary mapping to replace formal German words with informal ones
GLOSSARY = {
    r'\bSie\b':   'Du',
    r'\bIhnen\b': 'Dir',
    r'\bIhr\b':   'Dein',
    r'\bBestellung\b': 'Auftrag',
    r'\bdie Bestellung\b': 'den Auftrag',
}

def postprocess_german(text: str) -> str:
    """
    Post-process the translated German to apply informal pronouns
    and replace domain-specific terminology.
    """
    for pattern, replacement in GLOSSARY.items():
        text = re.sub(pattern, replacement, text)
    return text


def strip_braces(s: str) -> str:
    return re.sub(r"[{}]", "", s)


logger.info(f"Translation pipelines ready. Load time: {time() - start:.2f}s")


# --- Load Parrot Model ---

tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)

def mask_placeholders(text):
    phs = re.findall(r"\{.*?\}", text)
    tks = [f"TKN{i}" for i in range(len(phs))]
    m = text
    for t, p in zip(tks, phs):
        m = m.replace(p, t)
    return m, tks, phs

def restore(text, tks, phs):
    for t, p in zip(tks, phs):
        text = text.replace(t, p)
    return text

def generate_novelty_questions(question):
    # mask
    masked, tks, phs = mask_placeholders(question)

    inp = "paraphrase: " + masked
    enc = tok(inp, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = mdl.generate(
        **enc,
        max_length=128,
        num_beams=5,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    para = tok.decode(out[0], skip_special_tokens=True)
    return restore(para, tks, phs)

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

def generate_typo_question_mask_only(original_question):
    if not isinstance(original_question, str):
        raise TypeError(f"Expected string, got {type(original_question)}")

    # 1. Find placeholders: {ordername}, {ID}, etc.
    placeholders = re.findall(r"\{(.*?)\}", original_question)

    # 2. Apply typo to placeholder contents only
    typo_placeholders = []
    for p in placeholders:
        typo = typo_aug.augment(p)
        if isinstance(typo, list):
            typo = typo[0]
        typo_placeholders.append(typo)

    # 3. Replace original placeholders with typo versions
    typo_question = original_question
    for original, typoed in zip(placeholders, typo_placeholders):
        typo_question = typo_question.replace(f"{{{original}}}", f"{{{typoed}}}")

    return typo_question

def convert_english_to_german(english_text: str) -> str:
    """
    Translate English text to German using the translation pipeline.

    Args:
        english_text (str): The English text to be translated.

    Returns:
        str: The translated German text.
    """
    german_translation = en_to_de_pipeline(
        english_text,
        max_length=2000
    )[0]["translation_text"]

    return postprocess_german(german_translation)

def generate_answer_optimal(question, answer):
    """
    Generate a verbose, user-friendly answer based on the question context.
    """
    q = question.lower()
    ans_str = str(answer)

    if q.startswith("give me the info about the order"):
        return f"Here is the detailed information about the order: {ans_str}."

    if "what is the description of the" in q:
        return f"The description is: '{ans_str}'."

    if q.startswith("has the material for") and "already been ordered" in q:
        if answer:
            entries = [
                f"- Workorder {wo['workorderid']} @ {wo['workstation']}: {wo['description']}"
                for wo in answer
            ]
            return "Found these purchasing‐related work orders:\n" + "\n".join(entries)
        else:
            return "No purchasing‐related work orders found."
        
    if "how far is the order" in q:
        return f"The order was created on {ans_str}."

    if "at which workstation" in q:
        return f"The order is currently at workstation '{ans_str}'."

    if "what is the number of orders" in q:
        return f"There are {ans_str} orders for this item."

    if "what is the price" in q and "cost" in q:
        return f"The total cost is {ans_str}."

    if "what is the drawing no" in q:
        return f"The drawing number is '{ans_str}'."

    if "when is the delivery date" in q:
        return f"The delivery date is {ans_str}."

    if "what is the company id" in q:
        return f"The company ID is {ans_str}."

    if "where is order" in q and "stored" in q:
        return f"The order is stored at '{ans_str}'."

    if "what is the status of order" in q:
        return f"The current status of the order is '{ans_str}'."

    if "which articles do i need" in q:
        return f"Required articles: {ans_str}."

    if "which materials do i need" in q:
        return f"Required materials: {ans_str}."

    # Logbook entries
    if q.startswith("who reported work on order"):
        return f"Reported by: {ans_str}."

    if "status of the latest logbook entry" in q:
        return f"Latest logbook status: '{ans_str}'."

    # Work orders
    if "what is the total time planned for all work orders" in q:
        return f"Total planned time: {ans_str}."

    if "which workstations are involved in the work orders" in q:
        return f"Involved workstations: {ans_str}."

    # Article stock
    if "what is the stock quantity of article" in q:
        return f"Stock quantity: {ans_str}."

    if "where is article" in q and "stored" in q:
        return f"Article stored at '{ans_str}'."

    if "in which place is the article" in q:
        return f"Article location: {ans_str}."

    if "how much stock is available for article" in q:
        return f"Available stock: {ans_str}."

    if "when was the stock info for article" in q:
        return f"Stock info last updated on {ans_str}."

    # Material questions
    if "what is the stock level of material" in q:
        return f"Stock level: {ans_str}."

    if "where is material" in q and "stored" in q:
        return f"Material stored at '{ans_str}'."

    # Drawing number questions
    if "how many orders with the drawing number" in q:
        return f"There are {ans_str} orders with that drawing number."

    if q.startswith("give me the newest order with the drawing number"):
        return f"Newest order: '{ans_str}'."

    if q.startswith("when was the first order with this drawing number"):
        return f"First order placed on {ans_str}."

    # Warehouse position contents
    if "what is currently in warehouse position" in q:
        return f"Contents: {ans_str}."

    # Warehouse diversity
    if "list the top 10 warehouse positions by number of different articles stored" in q:
        return f"Top 10 warehouse positions by article count: {ans_str}."

    if "which warehouse position contains the most different articles" in q:
        return f"Position with most distinct articles: {ans_str}."

    if "sort the top 10 warehouse positions by the number of different articles stored" in q:
        return f"Sorted top 10 positions: {ans_str}."

    # Delivery thresholds
    if "how many times was article" in q and "delivered after" in q:
        return f"Delivered {ans_str} times after the given date."

    # Article cost
    if "what does the article" in q and "cost" in q:
        return f"Article cost: {ans_str}."

    # Delivery frequency in 2024
    if "how often was the article" in q and "delivered in 2024" in q:
        return f"Delivered {ans_str} times in 2024."

    # Fallback
    return f"{ans_str}. This information was retrieved from the ERP system." 

def generate_answer_long(question, answer):
    
    """
    Generate a verbose, user-friendly answer based on the question context.
    """
    q = question.lower()
    ans_str = str(answer)

    if q.startswith("give me the info about the order"):
        return (
            f"Certainly! Here is the complete and detailed information for the requested order: {ans_str}, "
            "which includes all relevant data currently available in our system."
        )

    if "what is the description of the" in q:
        return (
            f"The description provided for this order is as follows: '{ans_str}'"
            
        )

    if "has the material for" in q and "already been ordered" in q:
        if isinstance(answer, list) and answer:
            return (
                "According to our records, the necessary material has indeed been ordered already, "
                "and it is scheduled for delivery per the established timeline."
            )
        else:
            return (
                "Our system indicates that the required material has not been ordered yet. "
                "Please initiate the procurement process if you wish to proceed."
            )

    if "how far is the order" in q:
        return (
            f"The order's creation date is recorded as {ans_str}, "
            "which represents its current starting point in the workflow."
        )

    if "at which workstation" in q:
        return (
            f"At present, this order is positioned at workstation '{ans_str}', "
            "where the next phase of processing will take place."
        )

    if "what is the number of orders" in q:
        return (
            f"There are currently {ans_str} orders linked to this particular item, "
            "reflecting the total quantity in the system."
        )

    if "what is the price" in q and "cost" in q:
        return (
            f"The total cost associated with this order amounts to {ans_str}, "
            "which includes all line-item charges."
        )

    if "what is the drawing no" in q:
        return (
            f"The drawing number assigned to this order is '{ans_str}', "
            "as per the latest technical documentation."
        )

    if "when is the delivery date" in q:
        return (
            f"The scheduled delivery date for this order is {ans_str}, "
            "ensuring timely arrival as planned."
        )

    if "what is the company id" in q:
        return (
            f"The company ID associated with this order is {ans_str}, "
            "which identifies the client or internal department."
        )

    if "where is order" in q and "stored" in q:
        return (
            f"This order is currently stored at location '{ans_str}', "
            "which is its designated warehouse or storage area."
        )

    if "what is the status of order" in q:
        return (
            f"The current status of the order is '{ans_str}', "
            "indicating where it stands in the fulfillment process."
        )

    if "which articles do i need" in q:
        return (
            f"To complete this order, the following articles are required: {ans_str}. "
            "Please ensure they are available before proceeding."
        )

    if "which materials do i need" in q:
        return (
            f"The necessary materials for this order include: {ans_str}. "
            "Confirm that stock levels meet these requirements."
        )

    # Logbook entries
    if q.startswith("who reported work on order"):
        return (
            f"The logbook indicates that the following personnel reported work on this order: {ans_str}. "
            "Refer to the logbook for detailed timestamps and notes."
        )

    if "status of the latest logbook entry" in q:
        return (
            f"The most recent logbook entry shows the status as '{ans_str}', "
            "reflecting the latest on-site update."
        )

    # Work orders
    if "what is the total time planned for all work orders" in q:
        return (
            f"The aggregated planned time for all sub-work orders totals {ans_str}, "
            "which helps estimate overall completion."
        )

    if "which workstations are involved in the work orders" in q:
        return (
            f"The workstations involved in this order's processing sequence are: {ans_str}. "
            "This outlines the path through the production line."
        )

    # Article stock
    if "what is the stock quantity of article" in q:
        return (
            f"Current stock quantity for this article stands at {ans_str}, "
            "indicating immediate availability levels."
        )

    if "where is article" in q and "stored" in q:
        return (
            f"The article is located at storage position '{ans_str}', "
            "as recorded in the inventory module."
        )

    if "in which place is the article" in q:
        return (
            f"The designated place for this article is {ans_str}, "
            "corresponding to its current stock location."
        )

    if "how much stock is available for article" in q:
        return (
            f"There are {ans_str} units of this article currently available, "
            "based on the latest stock update."
        )

    if "when was the stock info for article" in q:
        return (
            f"The stock information was last updated on {ans_str}, "
            "ensuring data accuracy up to that timestamp."
        )

    # Material questions
    if "what is the stock level of material" in q:
        return (
            f"The material's stock level is recorded as {ans_str}, "
            "which reflects the current inventory count."
        )

    if "where is material" in q and "stored" in q:
        return (
            f"This material is stored at '{ans_str}', "
            "denoting its assigned warehouse or shelf."
        )

    # Drawing number questions
    if "how many orders with the drawing number" in q:
        return (
            f"There are {ans_str} orders associated with that drawing number, "
            "indicating its usage frequency."
        )

    if q.startswith("give me the newest order with the drawing number"):
        return (
            f"The most recent order for this drawing number is '{ans_str}', "
            "added to the system latest."
        )

    if q.startswith("when was the first order with this drawing number"):
        return (
            f"The first order bearing this drawing number was placed on {ans_str}, "
            "marking its initial entry into our records."
        )

    # Warehouse position contents
    if "what is currently in warehouse position" in q:
        return (
            f"The contents currently stored at this warehouse position are: {ans_str}. "
            "Refer to inventory for further classification."
        )

    # Warehouse diversity
    if "list the top 10 warehouse positions by number of different articles stored" in q:
        return (
            f"Here are the top 10 warehouse positions ranked by distinct article count: {ans_str}. "
            "This highlights the busiest storage areas."
        )

    if "which warehouse position contains the most different articles" in q:
        return (
            f"The warehouse position containing the highest number of different articles is {ans_str}, "
            "indicating it as the most diverse location."
        )

    if "sort the top 10 warehouse positions by the number of different articles stored" in q:
        return (
            f"Sorted list of the top 10 warehouse positions by distinct article count: {ans_str}. "
            "Useful for optimizing retrieval routes."
        )

    # Delivery thresholds
    if "how many times was article" in q and "delivered after" in q:
        return (
            f"The specified article was delivered {ans_str} times after the given date, "
            "according to delivery logs."
        )

    # Article cost
    if "what does the article" in q and "cost" in q:
        return (
            f"The cost for this article is {ans_str}, "
            "reflecting the current pricing structure."
        )

    # Delivery frequency in 2024
    if "how often was the article" in q and "delivered in 2024" in q:
        return (
            f"This article was delivered {ans_str} times during 2024, "
            "capturing its yearly delivery frequency."
        )

    # Fallback
    return (
        f"{ans_str}. "
        "This information was retrieved from the ERP system and represents the most up-to-date data available."
    ) 

def generate_qa_pairs_from_tree():
    """
    Generate a list of question-answer pairs from a hierarchical JSON tree of production orders.
    """
    logger.info("Starting QA generation from tree with sampling...")
    
    # Load JSON data
    with open("tree_structure.json", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build mapping: order ID and order name -> production record
    key_to_prod = {}
    for order_id, order_data in data.items():
        work_orders = order_data.get("work_orders", [])
        for prod in order_data.get("production_orders", []):
            order_name = prod.get("ordername", "Not Available")
            # Attach work orders list to each production record
            prod["work_orders"] = work_orders
            key_to_prod[order_id] = prod
            key_to_prod[order_name] = prod
    keys_list = list(key_to_prod.keys())
    
    PURCHASE_TERMS = [
        "einkauf",
        "bestellen der teile",
        "materialbestellung",
        "bestellen",
        "anfordern",
        "für die bestellung"
    ]
    
    # (template, extractor, category, subcategory)
    template_funcs = [
        ("Give me the info about the order {{{}}}.", lambda p: {k: v for k, v in p.items() if k != "work_orders"}, "order_info", "full_order"),
        ("What is the description of the {{{}}}.", lambda p: p.get("orderdesc", "There is no information."), "order_description", "description"),
        ("Has the material for {{{}}} already been ordered?", lambda p: [
            wo
            for wo in p.get("work_orders", [])
            if any(
                term in safe_lower(wo.get("description", ""))
                for term in PURCHASE_TERMS
            )
        ], "material_order_status", "purchase_check"),
        ("How far is the order {{{}}} in the ERP system?", lambda p: p.get("createdate", "There is no information."), "order_progress", "creation_date"),
        ("At which workstation is order {{{}}} currently?", lambda p: p.get("currentworkstation", "There is no information."), "order_workstation", "current_workstation"),
        ("What is the number of orders for {{{}}}?", lambda p: p.get("productionquantity", "There is no information."), "order_quantity", "production_quantity"),
        ("What is the price of the order {{{}}} cost?", lambda p: p.get("price", "There is no information."), "order_price", "price"),
        ("What is the drawing no for {{{}}}?", lambda p: p.get("drawno", "There is no information."), "drawing_number", "drawno"),
        ("When is the delivery date for order {{{}}}?", lambda p: p.get("deliverydate", "There is no information."), "delivery_date", "deliverydate"),
        ("What is the company ID for order {{{}}}?", lambda p: p.get("companyid", "There is no information."), "company_id", "companyid"),
        ("Where is order {{{}}} stored?", lambda p: p.get("warehousepos", "There is no information."), "order_storage", "warehousepos"),
        ("What is the status of order {{{}}}?", lambda p: p.get("orderstatus", "There is no information."), "order_status", "orderstatus"),
        ("Which articles do I need for order {{{}}}?", lambda p: ", ".join(a.get("productname", "") for a in p.get("articles", [])) or "No Articles", "article_requirements", "articles"),
        ("Which materials do I need for order {{{}}}?", lambda p: ", ".join(m.get("materialname", "") for m in p.get("materials", [])) or "No Materials", "material_requirements", "materials"),
    ]
    
    output = []  # Collect all QA dicts here
    
    # Generate QA for production orders
    for template, extractor, category, subcategory in template_funcs:
        # Randomly sample keys to limit per-template output
        sampled_keys = random.sample(keys_list, k=min(MAX_PER_TEMPLATE, len(keys_list)))
        for key in sampled_keys:
            # Stop if we've reached the global limit
            if len(output) >= MAX_TOTAL:
                return output
            
            # Create question variants
            question = template.format(key)
            typo_q = generate_typo_question_safe(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            
            # Extract and normalize answer
            prod_record = key_to_prod[key]
            answer_val = extractor(prod_record)
            if isinstance(answer_val, float) and pd.isna(answer_val):
                answer_val = "No data available"
            answer_str = str(answer_val)
            
            # Generate answer variants
            if isinstance(answer_val, (list, dict)):
                ans_opt = json.dumps(answer_val, ensure_ascii=False)
                ans_de = ans_opt
            else:
                ans_opt = generate_answer_optimal(question, answer_val)
                ans_de = convert_english_to_german(ans_opt)
            answer_long = generate_answer_long(question, answer_val)
            
            # Append QA dict
            output.append({
                "question": strip_braces(question),
                "question_typo": strip_braces(typo_q),
                "novelty_handling": strip_braces(back_q),
                "german_question": strip_braces(de_q),
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": category,
                "subcategory": subcategory
            })
    
    # Logbook entry questions
    logbook_recs = [r for r in key_to_prod.values() if r.get("logbooks")]
    for rec in random.sample(logbook_recs, k=min(MAX_PER_TEMPLATE, len(logbook_recs))):
        if len(output) >= MAX_TOTAL:
            return output
        order_name = rec.get("ordername", "Not Available")
        logbooks = rec.get("logbooks", [])
        employees = [lg.get("employee") for lg in logbooks if lg.get("employee")]
        latest_entry = max(logbooks, key=lambda x: x.get("reportingtime", ""))
        status = latest_entry.get("status", "Unknown")
        
        # Define two logbook questions
        logbook_qas = [
            (f"Who reported work on order {{{order_name}}} in the logbook?", ", ".join(employees), "logbook_reporter", "employee"),
            (f"What is the status of the latest logbook entry for {{{order_name}}}?", status, "logbook_status", "latest_status"),
        ]
        for q_text, ans_val, category, subcategory in logbook_qas:
            q_text = re.sub(r"[{}]", "", q_text)
            typo_q = generate_typo_question_safe(q_text)
            back_q = generate_novelty_questions(q_text)
            de_q = convert_english_to_german(q_text)
            ans_opt = generate_answer_optimal(q_text, ans_val)
            ans_de = convert_english_to_german(str(ans_opt))
            ans_long = generate_answer_long(q_text, ans_val)
            output.append({
                "question": q_text,
                "question_typo": typo_q,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": str(ans_val),
                "answer_long": ans_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": category,
                "subcategory": subcategory
            })
    
    # Work order summary questions
    workorder_recs = [r for r in key_to_prod.values() if r.get("work_orders")]
    for rec in random.sample(workorder_recs, k=min(MAX_PER_TEMPLATE, len(workorder_recs))):
        if len(output) >= MAX_TOTAL:
            return output
        order_name = rec.get("ordername", "Not Available")
        # Aggregate time and stations
        total_time = sum(w.get("time", 0) for w in rec.get("work_orders", []))
        stations = [w.get("workstation") for w in rec.get("work_orders", []) if w.get("workstation")]
        
        workorder_qas = [
            (f"What is the total time planned for all work orders in order {{{order_name}}}?", f"{total_time} seconds", "work_order_time", "total_time"),
            (f"Which workstations are involved in the work orders for {{{order_name}}}?", ", ".join(stations), "work_order_stations", "workstations"),
        ]
        for q_text, ans_val, category, subcategory in workorder_qas:
            q_text = re.sub(r"[{}]", "", q_text)
            typo_q = generate_typo_question_safe(q_text)
            back_q = generate_novelty_questions(q_text)
            de_q = convert_english_to_german(q_text)
            ans_opt = generate_answer_optimal(q_text, ans_val)
            ans_de = convert_english_to_german(str(ans_opt))
            ans_long = generate_answer_long(q_text, ans_val)
            output.append({
                "question": q_text,
                "question_typo": typo_q,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": str(ans_val),
                "answer_long": ans_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": category,
                "subcategory": subcategory
            })
    
    # Article stock questions
    article_recs = [r for r in key_to_prod.values() if r.get("articles")]
    for rec in random.sample(article_recs, k=min(MAX_PER_TEMPLATE, len(article_recs))):
        if len(output) >= MAX_TOTAL:
            return output
        for art in rec.get("articles", []):
            name = art.get("productname", "Unknown")
            place = art.get("stockdata", {}).get("place", "Unknown")
            qpairs = [
                (f"What is the stock quantity of article {{{name}}}?", art.get("stockquantity", "Unknown"), "article_stock_quantity", "stockquantity"),
                (f"Where is article {{{name}}} stored?", art.get("warehousepos", "Unknown"), "article_storage", "warehousepos"),
                (f"In which place is the article {{{name}}} located?", place, "article_storage", "place"),
                (f"How much stock is available for article {{{name}}} in {place}?", art.get("stockdata", {}).get("amount", "Unknown"), "article_available_stock", "amount"),
                (f"When was the stock info for article {{{name}}} last updated?", art.get("stockdata", {}).get("changedate", "Unknown"), "article_stock_quantity", "changedate"),
            ]
            for q_text, ans_val, category, subcategory in qpairs:
                q_text = re.sub(r"[{}]", "", q_text)
                typo_q = generate_typo_question_safe(q_text)
                back_q = generate_novelty_questions(q_text)
                de_q = convert_english_to_german(q_text)
                ans_opt = generate_answer_optimal(q_text, ans_val)
                ans_de = (convert_english_to_german(str(ans_val)) if ans_opt == str(ans_val) else ans_opt)
                ans_long = generate_answer_long(q_text, ans_val)
                output.append({
                    "question": q_text,
                    "question_typo": typo_q,
                    "novelty_handling": back_q,
                    "german_question": de_q,
                    "answer": str(ans_val),
                    "answer_long": ans_long,
                    "answer_optimal": ans_opt,
                    "answer_german": ans_de,
                    "category": category,
                    "subcategory": subcategory
                })
    
    # Enforce global limit
    if len(output) > MAX_TOTAL:
        output = random.sample(output, k=MAX_TOTAL)
    
    logger.info(f"Finished tree-based QA. Total: {len(output)}")
    return output

def generate_qa_pairs_from_database():
    """
    Generate question-answer pairs by sampling from Excel-based database tables.
    """
    logger.info("Starting QA generation from database Excel files with sampling and global limits...")
    
    test_data = []  # container for all QA entries
    
    # Load raw dataframes from Excel
    articles_df = pd.read_excel(r"data\\articles.xlsx")
    materials_df = pd.read_excel(r"data\\materials.xlsx")
    production_orders_df = pd.read_excel(r"data\\production_orders.xlsx")
    
    # Prepare materials table: parse dates and find latest per material
    materials_df['createdate'] = pd.to_datetime(materials_df['createdate'], errors='coerce')
    materials_df['changedate'] = pd.to_datetime(materials_df['changedate'], errors='coerce')
    materials_df['latestdate'] = materials_df[['changedate', 'createdate']].max(axis=1)
    latest_materials = (
        materials_df
        .sort_values(by='latestdate', ascending=False)
        .dropna(subset=['materialname'])
        .drop_duplicates(subset='materialname', keep='first')
    )
    sampled_materials = latest_materials.sample(n=min(MAX_PER_TEMPLATE, len(latest_materials)))
    
    # Generate QA for sampled materials
    for _, row in sampled_materials.iterrows():
        if len(test_data) >= MAX_TOTAL:
            return test_data
        mat_name = row['materialname']
        material_qas = [
            (f"What is the stock level of material {{{mat_name}}}?", row.get('stockquantity', 'Unknown'), "material_stock_level", "stockquantity"),
            (f"Where is material {{{mat_name}}} stored?", row.get('materialwarehousepos', 'Unknown'), "material_storage", "materialwarehousepos")
        ]
        for question, answer_val, category, subcategory in material_qas:
            question = re.sub(r"[{}]", "", question)
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            
            answer_str = str(answer_val)
            if isinstance(answer_val, (list, dict)):
                ans_opt = json.dumps(answer_val, ensure_ascii=False)
                ans_de = ans_opt
            else:
                ans_opt = generate_answer_optimal(question, answer_val)
                ans_de = convert_english_to_german(ans_opt)
            answer_long = generate_answer_long(question, answer_val)
            
            test_data.append({
                "question": question,
                "question_typo": typo_q,
                "question_typo_identity": typo_id,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": category,
                "subcategory": subcategory
            })
    
    # Count orders and track newest order names by drawing number
    production_orders_df['createdate'] = pd.to_datetime(production_orders_df['createdate'], errors='coerce')
    drawno_count = defaultdict(int)
    latest_orders = {}
    for _, row in production_orders_df.dropna(subset=['drawno']).iterrows():
        dr = row['drawno']
        dt = row['createdate']
        drawno_count[dr] += 1
        if dr not in latest_orders or dt > latest_orders[dr]['createdate']:
            latest_orders[dr] = {'createdate': dt, 'ordername': row.get('ordername', 'Unknown')}
    
    # QA: how many orders per drawing number
    for dr in random.sample(list(drawno_count.keys()), k=min(MAX_PER_TEMPLATE, len(drawno_count))):
        if len(test_data) >= MAX_TOTAL:
            return test_data
        question = f"How many orders with the drawing number {{{dr}}} are in the ERP?"
        question = re.sub(r"[{}]", "", question)
        answer_val = drawno_count[dr]
        
        typo_q = generate_typo_question_safe(question)
        typo_id = generate_typo_question_mask_only(question)
        back_q = generate_novelty_questions(question)
        de_q = convert_english_to_german(question)
        answer_str = str(answer_val)
        ans_opt = generate_answer_optimal(question, answer_val)
        ans_de = convert_english_to_german(ans_opt)
        answer_long = generate_answer_long(question, answer_val)
        test_data.append({
            "question": question,
            "question_typo": typo_q,
            "question_typo_identity": typo_id,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": answer_str,
            "answer_long": answer_long,
            "answer_optimal": ans_opt,
            "answer_german": ans_de,
            "category": "drawing_order_count",
            "subcategory": "drawno"
        })
    
    # QA: newest order name per drawing number
    for dr in random.sample(list(latest_orders.keys()), k=min(MAX_PER_TEMPLATE, len(latest_orders))):
        if len(test_data) >= MAX_TOTAL:
            return test_data
        rec = latest_orders[dr]
        question = f"Give me the newest order with the drawing number {{{dr}}}."
        question = re.sub(r"[{}]", "", question)
        answer_val = rec['ordername']
        
        typo_q = generate_typo_question_safe(question)
        typo_id = generate_typo_question_mask_only(question)
        back_q = generate_novelty_questions(question)
        de_q = convert_english_to_german(question)
        answer_str = answer_val
        ans_opt = generate_answer_optimal(question, answer_val)
        ans_de = convert_english_to_german(ans_opt)
        answer_long = generate_answer_long(question, answer_val)
        test_data.append({
            "question": question,
            "question_typo": typo_q,
            "question_typo_identity": typo_id,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": answer_str,
            "answer_long": answer_long,
            "answer_optimal": ans_opt,
            "answer_german": ans_de,
            "category": "drawing_newest_order",
            "subcategory": "drawno"
        })
    
    # First order placement date by drawing number
    first_orders = production_orders_df.dropna(subset=['orderid']).sample(n=min(MAX_PER_TEMPLATE, len(production_orders_df)))
    for _, row in first_orders.iterrows():
        if len(test_data) >= MAX_TOTAL:
            return test_data
        dr = row.get('drawno')
        dt = row.get('createdate')
        if dr and pd.notna(dt):
            question = f"When was the first order with this drawing number {{{dr}}} placed?"
            answer_val = dt.strftime('%Y-%m-%d')
            
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            answer_str = answer_val
            ans_opt = generate_answer_optimal(question, answer_val)
            ans_de = convert_english_to_german(ans_opt)
            answer_long = generate_answer_long(question, answer_val)
            test_data.append({
                "question": question,
                "question_typo": typo_q,
                "question_typo_identity": typo_id,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": "drawing_first_order",
                "subcategory": "drawno"
            })
    
    # Warehouse position contents
    positions = materials_df['materialwarehousepos'].dropna().unique().tolist()
    for pos in random.sample(positions, k=min(MAX_PER_TEMPLATE, len(positions))):
        if len(test_data) >= MAX_TOTAL:
            return test_data
        mats = materials_df[materials_df['materialwarehousepos']==pos]['materialname'].dropna().tolist()
        if not mats:
            continue
        question = f"What is currently in warehouse position {{{pos}}}?"
        answer_val = mats
        
        typo_q = generate_typo_question_safe(question)
        typo_id = generate_typo_question_mask_only(question)
        back_q = generate_novelty_questions(question)
        de_q = convert_english_to_german(question)
        ans_opt = json.dumps(answer_val, ensure_ascii=False)
        ans_de = ans_opt
        answer_str = ans_opt
        answer_long = generate_answer_long(question, answer_val)
        test_data.append({
            "question": question,
            "question_typo": typo_q,
            "question_typo_identity": typo_id,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": answer_str,
            "answer_long": answer_long,
            "answer_optimal": ans_opt,
            "answer_german": ans_de,
            "category": "warehouse_contents",
            "subcategory": "materialwarehousepos"
        })
    
    # Warehouse diversity rankings
    warehouse_counts = (
        articles_df.dropna(subset=['warehousepos','articelnumber'])
        .groupby('warehousepos')['articelnumber'].nunique()
        .sort_values(ascending=False)
    )
    top10 = warehouse_counts.head(10)
    lst_positions, lst_counts = top10.index.tolist(), top10.tolist()
    diversity_qas = [
        ("List the top 10 warehouse positions by number of different articles stored.", list(zip(lst_positions, lst_counts)), "warehouse_top10", "warehousepos"),
        ("Which warehouse position contains the most different articles?", (lst_positions[0], lst_counts[0]), "warehouse_most_diverse", "warehousepos"),
        ("Sort the top 10 warehouse positions by the number of different articles stored.", list(zip(lst_positions, lst_counts)), "warehouse_sorted", "warehousepos")
    ]
    for question, answer_val, category, subcategory in diversity_qas:
        if len(test_data) >= MAX_TOTAL:
            return test_data
        typo_q = generate_typo_question_safe(question)
        typo_id = generate_typo_question_mask_only(question)
        back_q = generate_novelty_questions(question)
        de_q = convert_english_to_german(question)
        ans_opt = json.dumps(answer_val, ensure_ascii=False)
        ans_de = ans_opt
        answer_str = ans_opt
        answer_long = generate_answer_long(question, answer_val)
        test_data.append({
            "question": question,
            "question_typo": typo_q,
            "question_typo_identity": typo_id,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": answer_str,
            "answer_long": answer_long,
            "answer_optimal": ans_opt,
            "answer_german": ans_de,
            "category": category,
            "subcategory": subcategory
        })
    
    # Delivery counts after 2024-01-01 threshold
    threshold = pd.to_datetime("2024-01-01")
    if 'articlelnumber' in production_orders_df.columns:
        artnums = (
            production_orders_df[production_orders_df['deliverydate'] > threshold]
            ['articlelnumber'].dropna().unique().tolist()
        )
        for art in random.sample(artnums, k=min(MAX_PER_TEMPLATE, len(artnums))):
            if len(test_data) >= MAX_TOTAL:
                return test_data
            count = int(
                production_orders_df[
                    (production_orders_df['articlenumber']==art) \
                    & (production_orders_df['deliverydate']>threshold)
                ].shape[0]
            )
            question = f"How many times was article {art} delivered after 01.01.2024?"
            answer_val = count
            
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            ans_opt = generate_answer_optimal(question, answer_val)
            ans_de = convert_english_to_german(ans_opt)
            answer_str = str(answer_val)
            answer_long = generate_answer_long(question, answer_val)
            test_data.append({
                "question": question,
                "question_typo": typo_q,
                "question_typo_identity": typo_id,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": "delivery_post_threshold",
                "subcategory": "articlenumber"
            })
    
    # Article storage & cost questions
    sampled_articles = (
        articles_df.dropna(subset=['articelnumber'])
        .sample(n=min(MAX_PER_TEMPLATE, len(articles_df)))
    )
    for _, row in sampled_articles.iterrows():
        if len(test_data) >= MAX_TOTAL:
            return test_data
        artnum = row['articelnumber']
        storage_cost_qas = [
            (f"Where is the article {artnum} stored?", row.get('warehousepos', 'Unknown'), "article_storage", "warehousepos"),
            (f"What does the article {{{artnum}}} cost?", row.get('price', 'Unknown'), "article_cost", "price")
        ]
        for question, answer_val, category, subcategory in storage_cost_qas:
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            answer_str = str(answer_val)
            ans_opt = generate_answer_optimal(question, answer_val)
            ans_de = convert_english_to_german(ans_opt)
            answer_long = generate_answer_long(question, answer_val)
            test_data.append({
                "question": question,
                "question_typo": typo_q,
                "question_typo_identity": typo_id,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": category,
                "subcategory": subcategory
            })
    
    # Delivery frequency in 2024
    delivery_2024 = production_orders_df[
        (production_orders_df['deliverydate'] >= "2024-01-01")
        & (production_orders_df['deliverydate'] < "2025-01-01")
    ]
    if 'articlenumber' in production_orders_df.columns:
        arts_24 = delivery_2024['articlenumber'].dropna().unique().tolist()
        for art in random.sample(arts_24, k=min(MAX_PER_TEMPLATE, len(arts_24))):
            if len(test_data) >= MAX_TOTAL:
                return test_data
            count = int(delivery_2024[delivery_2024['articlenumber']==art].shape[0])
            question = f"How often was the article {{{art}}} delivered in 2024?"
            answer_val = count
            
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            ans_opt = generate_answer_optimal(question, answer_val)
            ans_de = convert_english_to_german(ans_opt)
            answer_str = str(answer_val)
            answer_long = generate_answer_long(question, answer_val)
            test_data.append({
                "question": question,
                "question_typo": typo_q,
                "question_typo_identity": typo_id,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
                "answer_optimal": ans_opt,
                "answer_german": ans_de,
                "category": "delivery_2024_frequency",
                "subcategory": "articlenumber"
            })
    
    logger.info(f"Finished database-based QA with sampling. Total items: {len(test_data)}")
    return test_data

def generate_analytical_answers():
    """
    Generate answers for analytical questions based on the tree structure data.
    Focuses on process flow, resource optimization, quality metrics, and other analytical insights.
    """
    logger.info("Generating analytical answers...")
    
    # Load tree structure
    with open("tree_structure.json", encoding="utf-8") as f:
        tree_data = json.load(f)
    
    # Initialize data structures for analysis
    workstation_sequences = defaultdict(list)
    workstation_times = defaultdict(list)
    material_usage = defaultdict(int)
    quality_issues = defaultdict(int)
    order_costs = defaultdict(list)
    material_inventory = defaultdict(list)
    order_lead_times = defaultdict(list)
    workstation_utilization = defaultdict(int)
    
    # Process data for analysis
    for order_id, order_data in tree_data.items():
        # Process work orders for sequence and time analysis
        work_orders = order_data.get("work_orders", [])
        if work_orders:
            # Track workstation sequence
            sequence = []
            for wo in work_orders:
                is_valid, validated_wo = validate_record(wo, RECORD_TYPES["work_order"])
                if is_valid and validated_wo["workstation"]:
                    sequence.append(validated_wo["workstation"])
                    workstation_times[validated_wo["workstation"]].append(validated_wo["time"])
                    workstation_utilization[validated_wo["workstation"]] += validated_wo["time"]
            
            if sequence:
                workstation_sequences[tuple(sequence)].append(order_id)
        
        # Process materials for usage analysis
        materials = order_data.get("materials", [])
        for material in materials:
            material_name = material.get("materialname")
            if material_name:
                material_usage[material_name] += 1
                if material.get("createdate"):
                    material_inventory[material_name].append(material.get("createdate"))
        
        # Process production orders for cost and quality analysis
        for prod in order_data.get("production_orders", []):
            is_valid, validated_prod = validate_record(prod, RECORD_TYPES["production_order"])
            if is_valid:
                if validated_prod.get("price"):
                    order_costs[validated_prod["ordername"]].append(validated_prod["price"])
                
                created = validated_prod.get("createdate")
                delivery = validated_prod.get("deliverydate")
                if created and delivery:
                    order_lead_times[validated_prod["ordername"]].append((created, delivery))
        
        # Process logbooks for quality issues
        for log in order_data.get("logbooks", []):
            is_valid, validated_log = validate_record(log, RECORD_TYPES["logbook"])
            if is_valid:
                status = safe_lower(validated_log["status"])
                if "error" in status or "issue" in status or "problem" in status:
                    quality_issues[order_id] += 1
    
    qa_pairs = []
    
    MAX_PER_ANALYTICAL_CATEGORY = 1  # or any number you prefer

    # 1. Process Flow Analysis
    sequence_keys = list(workstation_sequences.keys())
    if len(sequence_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        sequence_keys = random.sample(sequence_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for sequence in sequence_keys:
        orders = workstation_sequences[sequence]
        if len(orders) > 1:  # Only consider sequences that appear multiple times
            question = f"What is the typical sequence of workstations for orders following pattern {sequence}?"
            answer = f"This sequence appears in {len(orders)} orders: {', '.join(orders)}"
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "process_flow",
                "subcategory": "workstation_sequence"
            })
    
    # 2. Resource Optimization
    workstation_keys = list(workstation_times.keys())
    if len(workstation_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        workstation_keys = random.sample(workstation_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for workstation in workstation_keys:
        times = workstation_times[workstation]
        if times:
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            question = f"What is the average processing time and total utilization for workstation {workstation}?"
            answer = f"Average time: {avg_time:.2f} seconds, Total utilization: {total_time:.2f} seconds"
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "resource_optimization",
                "subcategory": "workstation_performance"
            })
    
    # 3. Quality Metrics
    quality_keys = list(quality_issues.keys())
    if len(quality_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        quality_keys = random.sample(quality_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for order_id in quality_keys:
        issues = quality_issues[order_id]
        if issues > 0:
            question = f"How many quality issues were reported for order {order_id}?"
            answer = f"Total quality issues: {issues}"
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "quality_metrics",
                "subcategory": "issue_tracking"
            })
    
    # 4. Cost Efficiency
    cost_keys = list(order_costs.keys())
    if len(cost_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        cost_keys = random.sample(cost_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for order_name in cost_keys:
        costs = order_costs[order_name]
        if costs:
            avg_cost = sum(costs) / len(costs)
            question = f"What is the average cost for order {order_name}?"
            answer = f"Average cost: {avg_cost:.2f}"
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "cost_efficiency",
                "subcategory": "order_costs"
            })
    
    # 5. Inventory Management
    material_keys = list(material_inventory.keys())
    if len(material_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        material_keys = random.sample(material_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for material in material_keys:
        timestamps = material_inventory[material]
        if len(timestamps) > 1:
            # Convert all timestamps to datetime
            dt_timestamps = [pd.to_datetime(ts, errors='coerce') for ts in timestamps if pd.notna(ts)]
            dt_timestamps = [ts for ts in dt_timestamps if ts is not pd.NaT]
            if len(dt_timestamps) > 1:
                dt_timestamps.sort()
                avg_time = (dt_timestamps[-1] - dt_timestamps[0]).total_seconds() / len(dt_timestamps)
                question = f"What is the average time material {material} spends in inventory?"
                answer = f"Average inventory time: {avg_time/86400:.2f} days"
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "category": "inventory_management",
                    "subcategory": "material_turnover"
                })
    
    # 6. Production Planning
    lead_time_keys = list(order_lead_times.keys())
    if len(lead_time_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        lead_time_keys = random.sample(lead_time_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for order_name in lead_time_keys:
        lead_times = order_lead_times[order_name]
        if lead_times:
            dt_lead_times = []
            for start, end in lead_times:
                dt_start = pd.to_datetime(start, errors='coerce')
                dt_end = pd.to_datetime(end, errors='coerce')
                if pd.notna(dt_start) and pd.notna(dt_end):
                    dt_lead_times.append((dt_start, dt_end))
            if dt_lead_times:
                avg_lead_time = sum((end - start).total_seconds() for start, end in dt_lead_times) / len(dt_lead_times)
                question = f"What is the average lead time for order {order_name}?"
                answer = f"Average lead time: {avg_lead_time/86400:.2f} days"
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "category": "production_planning",
                    "subcategory": "lead_time_analysis"
                })
    
    # 7. Cross-Reference Analysis
    usage_keys = list(material_usage.keys())
    if len(usage_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        usage_keys = random.sample(usage_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for material in usage_keys:
        usage_count = material_usage[material]
        if usage_count > 1:
            question = f"How many orders use material {material}?"
            answer = f"Material {material} is used in {usage_count} orders"
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "cross_reference",
                "subcategory": "material_usage"
            })
    
    # 8. Performance Metrics
    utilization_keys = list(workstation_utilization.keys())
    if len(utilization_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        utilization_keys = random.sample(utilization_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for workstation in utilization_keys:
        utilization = workstation_utilization[workstation]
        question = f"What is the total utilization time for workstation {workstation}?"
        answer = f"Total utilization: {utilization:.2f} seconds"
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "category": "performance_metrics",
            "subcategory": "workstation_utilization"
        })
    
    # --- Sampling and MAX_TOTAL enforcement ---
    # (No need to sample again here, just enforce MAX_TOTAL)
    if len(qa_pairs) > MAX_TOTAL:
        qa_pairs = random.sample(qa_pairs, MAX_TOTAL)

    # Add variants for each QA pair
    final_qa_pairs = []
    for qa in qa_pairs:
        # Generate variants
        typo_q = generate_typo_question_safe(qa["question"])
        back_q = generate_novelty_questions(qa["question"])
        de_q = convert_english_to_german(qa["question"])

        # Generate answer variants
        ans_opt = generate_answer_optimal(qa["question"], qa["answer"])
        ans_de = convert_english_to_german(ans_opt)
        ans_long = generate_answer_long(qa["question"], qa["answer"])

        final_qa_pairs.append({
            "question": qa["question"],
            "question_typo": typo_q,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": qa["answer"],
            "answer_long": ans_long,
            "answer_optimal": ans_opt,
            "answer_german": ans_de,
            "category": qa["category"],
            "subcategory": qa["subcategory"]
        })

    logger.info(f"Generated {len(final_qa_pairs)} analytical QA pairs (sampled and capped)")
    return final_qa_pairs

def generate_balanced_test_dataset():
    """
    Generate a balanced test dataset with weighted distribution of question types.
    """
    # Generate from both sources
    tree_qa = generate_qa_pairs_from_tree()
    db_qa = generate_qa_pairs_from_database()
    analytical_qa = generate_analytical_answers()
    
    # Combine all QA pairs
    all_qa = tree_qa + db_qa + analytical_qa
    
    # Categorize questions
    categorized_qa = defaultdict(list)
    for qa in all_qa:
        category = qa.get("category", "other")
        categorized_qa[category].append(qa)
    
    # Calculate target samples per category based on weights
    total_weight = sum(CATEGORY_WEIGHTS.values())
    balanced_qa = []
    
    # First pass: Sample based on weights
    for category, qa_list in categorized_qa.items():
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        target_samples = int((weight / total_weight) * MAX_TOTAL)
        if qa_list:
            sampled_qa = random.sample(qa_list, min(target_samples, len(qa_list)))
            balanced_qa.extend(sampled_qa)
    
    # Second pass: Fill remaining slots
    remaining_slots = MAX_TOTAL - len(balanced_qa)
    if remaining_slots > 0:
        remaining_qa = []
        for qa_list in categorized_qa.values():
            remaining_qa.extend([qa for qa in qa_list if qa not in balanced_qa])
        
        if remaining_qa:
            balanced_qa.extend(random.sample(remaining_qa, min(remaining_slots, len(remaining_qa))))
    
    # Shuffle the final dataset
    random.shuffle(balanced_qa)
    
    # Log the distribution
    logger.info(f"Generated balanced dataset with {len(balanced_qa)} questions")
    for category, qa_list in categorized_qa.items():
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        logger.info(f"Category '{category}' (weight: {weight}): {len(qa_list)} questions")
    
    return balanced_qa
