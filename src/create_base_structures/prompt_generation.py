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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

logger.info("Loading translation pipelines...")
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
device_id = 0 if device == "cuda" else -1
logger.info(f"Using device: {device}")

# --- Constants ---
MAX_PER_TEMPLATE = 10   # per-template sample size
MAX_TOTAL = 200        # global QA limit
MAX_PER_ANALYTICAL_CATEGORY = 10  # or any number you prefer

# Placeholder for RECORD_TYPES - adjust with your actual types/schemas
RECORD_TYPES = {
    "work_order": ["workstation", "time"],
    "logbook": ["status", "employee"],
    "production_order": ["ordername", "price", "createdate", "deliverydate"]
}

# Category weights for balancing
CATEGORY_WEIGHTS = {
    # Order-related
    "order_info": 1.0,
    "order_description": 1.0,
    "order_status": 1.0,
    "order_progress": 1.0,
    "order_workstation": 1.3,
    "order_quantity": 1.3,
    "order_price": 1.4,
    "order_storage": 1.2,
    "company_id": 1.0,

    # Material and Article
    "material_order_status": 1.2,
    "material_stock_level": 1.2,
    "material_storage": 1.2,
    "material_requirements": 1.2,
    "article_requirements": 1.2,
    "article_stock_quantity": 1.2,
    "article_storage": 1.2,
    "article_available_stock": 1.2,
    "article_cost": 1.4,

    # Drawing and Documentation
    "drawing_number": 1.2,
    "drawing_order_count": 1.2,
    "drawing_newest_order": 1.2,
    "drawing_first_order": 1.2,

    # Warehouse
    "warehouse_contents": 1.3,
    "warehouse_top10": 1.3,
    "warehouse_most_diverse": 1.3,
    "warehouse_sorted": 1.3,

    # Logbook
    "logbook_reporter": 1.1,
    "logbook_status": 1.1,

    # Work order
    "work_order_time": 1.3,
    "work_order_stations": 1.3,

    # Delivery
    "delivery_date": 1.5,
    "delivery_post_threshold": 1.5,
    "delivery_2024_frequency": 1.5,

    # Analytical (from generate_analytical_answers)
    "process_flow": 1.6,
    "resource_optimization": 1.6,
    "quality_metrics": 1.6,
    "cost_efficiency": 1.6,
    "inventory_management": 1.6,
    "production_planning": 1.6,
    "cross_reference": 1.6,
    "performance_metrics": 1.6,
}

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

nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(device)

en_to_de_pipeline = pipeline(
    "translation",
    model=nllb_model,
    tokenizer=nllb_tokenizer,
    src_lang="eng_Latn",
    tgt_lang="deu_Latn",
    device=device_id
)

logger.info(f"Translation pipelines ready. Load time: {time() - start:.2f}s")


# --- Load Parrot Model ---
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)

# Define a glossary mapping to replace formal German words with informal ones
GLOSSARY = {
    
    # Kişi zamirleri ve hitap
    r'\bSie\b': 'Du',
    r'\bIhnen\b': 'Dir',
    r'\bIhr\b': 'Dein',
    r'\bIhre\b': 'Deine',
    r'\bIhrem\b': 'Deinem',
    r'\bIhren\b': 'Deinen',
    r'\bIhres\b': 'Deines',

    # Emir kipleri ve doğal kalıplar
    r'\bGeben Du mir\b': 'Gib mir',
    r'\bGeben Sie mir\b': 'Gib mir',
    r'\bGeben Sie\b': 'Gib mir',
    r'\bGib bitte\b': 'Gib mir',
    r'\bBitte gib mir\b': 'Gib mir',
    r'\bBitte\b': '',

    # ERP terimleri ve doğru artikeller
    r'\bdie Ordnung\b': 'den Auftrag',
    r'\bder Ordnung\b': 'dem Auftrag',
    r'\bOrdnung\b': 'Auftrag',
    r'\bdie Bestellung\b': 'den Auftrag',
    r'\bder Bestellung\b': 'dem Auftrag',
    r'\bBestellungen\b': 'Aufträge',
    r'\bBestellung\b': 'Auftrag',
    r'\bArtikel\b': 'Artikel',
    r'\bMaterial\b': 'Material',
    r'\bArbeitsstation\b': 'Arbeitsplatz',
    r'\bZeichnung\b': 'Zeichnung',
    r'\bZeichnungsnummer\b': 'Zeichnungsnummer',
    r'\bLagerposition\b': 'Lagerplatz',
    r'\bLiefertermin\b': 'Lieferdatum',
    r'\bFirma-ID\b': 'Firmen-ID',

    # Sadeleştirilmiş sorular
    r'Könnten Sie bitte': 'Kannst Du mal',
    r'Könnten Sie': 'Kannst Du',
    r'Würden Sie bitte': 'Würdest Du mal',
    r'Würden Sie': 'Würdest Du',

    # Bilgi isteme kalıpları
    r'Bitte geben Sie mir die Informationen über den Auftrag': 'Gib mir die Information über den Auftrag',
    r'Informationen über den Auftrag': 'Information über den Auftrag',

}

ANSWER_GLOSSARY = {
    # Placeholder and status expressions
    r'\bNicht ausgelöst\b': 'Noch nicht gestartet',
    r'\bAusgelöst\b': 'Gestartet',
    r'\bPausiert\b': 'In Bearbeitung unterbrochen',
    r'\bNicht verfügbar\b': 'Nicht im System verfügbar',
    r'\bKeine Daten verfügbar\b': 'Keine Informationen vorhanden',
    r'\bUnbekannt\b': 'Nicht bekannt',

    # Common prefixes
    r'^Die aktuelle Status ist\b': 'Der aktuelle Status ist',
    r'^Die Status ist\b': 'Der Status ist',

    # Cost/price phrases
    r'Die Preis ist\b': 'Der Preis ist',
    r'Die Gesamtkosten sind\b': 'Die Gesamtkosten betragen',
    r'Die Artikelkosten sind\b': 'Die Artikelkosten betragen',
    r'Gesamtkosten:': 'Gesamtkosten sind:',
    r'Durchschnittliche Kosten sind\b': 'Die durchschnittlichen Kosten betragen',

    # Stock and warehouse
    r'Artikel Lagerposition ist\b': 'Die Lagerposition des Artikels ist',
    r'Artikel befindet sich in\b': 'Der Artikel befindet sich in',
    r'In Lagerposition\b': 'In der Lagerposition',

    # Time expressions
    r'Sekunden\b': 'Sekunden',
    r'Durchschnittliche Zeit ist\b': 'Die durchschnittliche Zeit beträgt',
    r'Durchschnittliche Lagerzeit ist\b': 'Die durchschnittliche Lagerzeit beträgt',
    r'Vorlaufzeit ist\b': 'Die Vorlaufzeit beträgt',
    r'Gesamtauslastung ist\b': 'Die gesamte Auslastung beträgt',

    # Lists and entities
    r'Erforderliche Materialien sind\b': 'Die benötigten Materialien sind',
    r'Benötigte Artikel sind\b': 'Die benötigten Artikel sind',
    r'Beteiligte Arbeitsplätze sind\b': 'Folgende Arbeitsplätze sind beteiligt',
    r'Qualitätsprobleme Gesamtzahl\b': 'Anzahl der gemeldeten Qualitätsprobleme',

    # Miscellaneous
    r'Gib mir die Information\b': 'Hier ist die Information',
    r'Details über\b': 'Informationen über',
}

def convert_english_to_german(text: str) -> str:
    """
    Translates English text into natural-sounding German using:
    - Early handling of known placeholders
    - Numeric unit reformatting (e.g., "66.62 days" → "66,62 Tage")
    - Neural machine translation via Hugging Face pipeline
    - Postprocessing using regex-based glossary replacements
    """
    if not isinstance(text, str):
        return str(text)

    text = text.strip()


    try:
        translation = en_to_de_pipeline(
            text,
            max_length=512,
            num_beams=5,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )[0]["translation_text"]
    except Exception as e:
        logger.warning(f"[Translation failed] {str(e)}")
        return text

    # Apply glossary-based replacements using regex
    for pattern, replacement in GLOSSARY.items():
        translation = re.sub(pattern, replacement, translation)

    # Clean up punctuation and spacing
    translation = re.sub(r"\s+([.,!?])", r"\1", translation)
    translation = re.sub(r"([.,!?])\s+", r"\1 ", translation)
    translation = re.sub(r"\s+", " ", translation).strip()

    # Ensure question has a question mark if appropriate
    if translation and not translation.endswith("?") and translation.split()[0].capitalize() in ["Wie", "Was", "Wann", "Wo", "Welche", "Wer"]:
        translation += "?"

    return translation

def postprocess_german(text: str) -> str:
    """
    Almanca metni daha doğal ve sade hale getirir.
    """
    # Önce sözlük eşleştirmelerini uygula
    for pattern, replacement in GLOSSARY.items():
        text = re.sub(pattern, replacement, text)
    
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Noktalama işaretlerini düzelt
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\s+', r'\1 ', text)
    
    # Büyük/küçük harf düzeltmeleri
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    
    # Soru işareti kontrolü
    if not text.endswith('?') and any(text.startswith(w) for w in ['Wie', 'Was', 'Wo', 'Wann', 'Welche', 'Welcher', 'Welches']):
        text += '?'
    
    return text.strip()

def postprocess_answer_german(text: str) -> str:
    """
    Clean up and normalize German answer text using regex-based replacements.
    """
    for pattern, replacement in ANSWER_GLOSSARY.items():
        text = re.sub(pattern, replacement, text)
    
    # Final formatting
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([.,!?])\s+", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_braces(s: str) -> str:
    return re.sub(r"[{}]", "", s)



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
    """
    Generate a meaning-preserving paraphrase for novelty_handling, based on the question type.
    Ensures the output is never identical to the original question.
    """
    masked, tks, phs = mask_placeholders(question)
    novelty = None
    try:
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
        novelty = restore(para, tks, phs)
        # Only use if meaning is preserved (basic check: contains main noun/number)
        if not (any(x in novelty for x in re.findall(r"\d+", question)) or any(x in novelty for x in re.findall(r"[A-Za-z]+", question))):
            novelty = None
    except Exception as e:
        logger.warning(f"Parrot model failed: {str(e)}")
        novelty = None

    # 2. Rule-based, question-type-aware paraphrasing
    q = question.strip()
    q_lower = q.lower()
    if novelty is None or novelty.strip().lower() == question.strip().lower():
        # Where is article ... stored?
        if q_lower.startswith("where is article") and "stored" in q_lower:
            novelty = q.replace("Where is article", "Could you tell me the storage location of article").replace("where is article", "Could you tell me the storage location of article")
        # In which place is the article ... located?
        elif q_lower.startswith("in which place is the article") and "located" in q_lower:
            novelty = q.replace("In which place is the article", "Could you tell me where article").replace("in which place is the article", "Could you tell me where article")
        # What is the description of ...
        elif q_lower.startswith("what is the description of"):
            novelty = q.replace("What is the description of", "Could you describe").replace("what is the description of", "Could you describe")
        # How far is the order ...
        elif q_lower.startswith("how far is the order"):
            novelty = q.replace("How far is the order", "Can you provide the progress of order").replace("how far is the order", "Can you provide the progress of order")
        # What is the average cost for order ...
        elif "what is the average cost for order" in q_lower:
            novelty = q.replace("What is the average cost for order", "Could you tell me the average cost for order").replace("what is the average cost for order", "Could you tell me the average cost for order")
        # What is the price of the order ... cost?
        elif "what is the price of the order" in q_lower and "cost" in q_lower:
            novelty = q.replace("What is the price of the order", "Could you tell me the price of the order").replace("what is the price of the order", "Could you tell me the price of the order")
        # What is the drawing no for ...
        elif q_lower.startswith("what is the drawing no for"):
            novelty = q.replace("What is the drawing no for", "Could you tell me the drawing number for").replace("what is the drawing no for", "Could you tell me the drawing number for")
        # When is the delivery date for order ...
        elif q_lower.startswith("when is the delivery date for order"):
            novelty = q.replace("When is the delivery date for order", "Could you tell me the delivery date for order").replace("when is the delivery date for order", "Could you tell me the delivery date for order")
        # What is the company ID for order ...
        elif q_lower.startswith("what is the company id for order"):
            novelty = q.replace("What is the company ID for order", "Could you tell me the company ID for order").replace("what is the company id for order", "Could you tell me the company ID for order")
        # Where is order ... stored?
        elif q_lower.startswith("where is order") and "stored" in q_lower:
            novelty = q.replace("Where is order", "Could you tell me where order").replace("where is order", "Could you tell me where order")
        # What is the status of order ...
        elif q_lower.startswith("what is the status of order"):
            novelty = q.replace("What is the status of order", "Could you tell me the status of order").replace("what is the status of order", "Could you tell me the status of order")
        # Which articles do I need for order ...
        elif q_lower.startswith("which articles do i need for order"):
            novelty = q.replace("Which articles do I need for order", "Could you list the articles required for order").replace("which articles do i need for order", "Could you list the articles required for order")
        # Which materials do I need for order ...
        elif q_lower.startswith("which materials do i need for order"):
            novelty = q.replace("Which materials do I need for order", "Could you list the materials required for order").replace("which materials do i need for order", "Could you list the materials required for order")
        # What is the stock quantity of article ...
        elif q_lower.startswith("what is the stock quantity of article"):
            novelty = q.replace("What is the stock quantity of article", "Could you tell me the stock quantity of article").replace("what is the stock quantity of article", "Could you tell me the stock quantity of article")
        # How much stock is available for article ...
        elif q_lower.startswith("how much stock is available for article"):
            novelty = q.replace("How much stock is available for article", "Could you tell me how much stock is available for article").replace("how much stock is available for article", "Could you tell me how much stock is available for article")
        # When was the stock info for article ... last updated?
        elif q_lower.startswith("when was the stock info for article") and "last updated" in q_lower:
            novelty = q.replace("When was the stock info for article", "Could you tell me when the stock info for article").replace("when was the stock info for article", "Could you tell me when the stock info for article")
        # What does the article ... cost?
        elif q_lower.startswith("what does the article") and "cost" in q_lower:
            novelty = q.replace("What does the article", "Could you tell me the cost of the article").replace("what does the article", "Could you tell me the cost of the article")
        # How many orders use material ...
        elif q_lower.startswith("how many orders use material"):
            novelty = q.replace("How many orders use material", "Could you tell me how many orders use material").replace("how many orders use material", "Could you tell me how many orders use material")
        # At which workstation is order ... currently?
        elif q_lower.startswith("at which workstation is order") and "currently" in q_lower:
            novelty = q.replace("At which workstation is order", "Could you tell me at which workstation order").replace("at which workstation is order", "Could you tell me at which workstation order")
        # What is the number of orders for ...
        elif q_lower.startswith("what is the number of orders for"):
            novelty = q.replace("What is the number of orders for", "Could you tell me the number of orders for").replace("what is the number of orders for", "Could you tell me the number of orders for")
        # Give me the info about the order ...
        elif q_lower.startswith("give me the info about the order"):
            novelty = q.replace("Give me the info about the order", "Could you provide the information for order").replace("give me the info about the order", "Could you provide the information for order")
        # Fallback: generic template
        else:
            novelty = f"Could you tell me {q}" if not q.endswith('?') else f"Could you tell me {q[:-1]}?"

    # Son kontrol: Hala aynıysa, başına/sonuna küçük bir ekleme yap
    if novelty.strip().lower() == question.strip().lower():
        novelty = f"Could you please tell me {question}" if not question.endswith('?') else f"Could you please tell me {question[:-1]}?"

    return novelty

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

def safe_lower(text):
    """Safely convert text to lowercase, handling None and float values."""
    if text is None:
        return ""
    if isinstance(text, (float, int)):
        return ""
    return str(text).lower()

def validate_record(record: dict, record_type: str) -> Tuple[bool, dict]:
    """
    Validates a record
    Returns (is_valid, validated_record_or_error).
    """
    required_fields = RECORD_TYPES
    if record_type not in required_fields:
        return False, {"error": f"Unknown record_type: {record_type}"}

    missing = []
    for field in required_fields[record_type]:
        if field not in record or record[field] in [None, ""]:
            missing.append(field)

    if missing:
        return False, {"error": f"Missing required fields: {missing}", "record": record}

    return True, record

def validate_warehouse_position(pos):
    """Validate warehouse position values."""
    if pos in ["WAREHOUSEPOS", "Unknown", "No data available"]:
        return "No data available"
    return pos

def validate_status(status):
    """Validate status values and return their meaning with explanation."""
    if str(status) == "0":
        return "0 (not triggered)"
    if str(status) == "1":
        return "1 (triggered)"
    if str(status) == "100":
        return "100 (on hold)"
    if str(status) == "-1" or status == -1 or status == "Unknown":
        return "No data available"
    return f"{status} (custom status)"

def validate_price(price):
    """Validate price values."""
    if price == 0.0 or pd.isna(price):
        return "No data available"
    return price

def validate_list(items):
    """Validate list values."""
    if not items or items in ["No Articles", "No Materials"]:
        return "No data available"
    return items

def validate_answer(answer, category):
    """Validate the answer content based on general and category-specific rules."""

    # 1. Check if the answer is an empty list, tuple, or array
    if isinstance(answer, (list, tuple, np.ndarray)):
        if len(answer) == 0 or all(not str(x).strip() for x in answer):
            return "No data available"
        return answer

    # 2. Return "No data available" for NaN or None values
    if pd.isna(answer) or answer is None:
        return "No data available"

    # 3. Normalize the answer to a lowercase string
    ans_str = str(answer).strip().lower()

    # 4. Check for meaningless string values
    meaningless = {"", "none", "null", "nan", "na", "unknown", "-1", "0", "0.0", "0.00", "nicht gemeldet"}
    if ans_str in meaningless:
        return "No data available"

    # 5. Check if the answer string contains only zero or "0.00"-style expressions
    if isinstance(answer, str):
        # Remove any labels like "Average time:" or "Total utilization:" and analyze numeric values
        numeric_parts = re.findall(r"\b0+(\.0+)?\b", answer)
        nonzero_numeric = re.findall(r"\b(?!0+(?:\.0+)?\b)\d+(?:\.\d+)?\b", answer)
        if numeric_parts and not nonzero_numeric:
            return "No data available"
    if isinstance(answer, str):
        negative_match = re.findall(r"-\d+(?:\.\d+)?", answer)
        if negative_match:
            return "No data available"

    # 6. Try numeric zero check
    try:
        if float(ans_str) == 0.0:
            return "No data available"
    except ValueError:
        pass  # Skip non-numeric string

    # 7. German indicator phrases (let them pass)
    german_indicators = [
        "nicht gemeldet", "unbekannt", "keine daten", "nicht verfügbar",
        "warehousepos", "speicher", "teil", "stoff", "plan"
    ]
    if any(ind in ans_str for ind in german_indicators):
        return answer  # Pass through for later postprocessing

    # 8. Special categories that should skip validation
    if category in ["drawing_number", "drawing_newest_order", "order_progress", "delivery_date", "drawing_first_order"]:
        # Only check if empty or None
        if not answer or str(answer).strip() == "":
            return "No data available"
        return answer

    # 9. Category-specific rules
    if category in ["warehouse_contents", "material_storage"]:
        result = validate_warehouse_position(answer)
        return "No data available" if result is None else result
    elif category == "article_storage":
        # For article storage, just validate if it's a valid warehouse position
        if answer in ["WAREHOUSEPOS", "Unknown", "No data available"]:
            return "No data available"
        return answer
    elif category in ["order_status", "logbook_status"]:
        result = validate_status(answer)
        return "No data available" if result is None else result
    elif category in ["order_price", "article_cost"]:
        result = validate_price(answer)
        return "No data available" if result is None else result
    elif category in ["article_requirements", "material_requirements"]:
        result = validate_list(answer)
        return "No data available" if not result else result

    # 10. Fallback: return the original answer if no condition matched
    return answer

def generate_answer_optimal(question, answer):
    """
    Generate a clear and concise answer based on the question context.
    For unavailable answers, return a user-friendly message.
    """
    q = question.lower()
    ans_str = str(answer) if answer is not None else "Not Available"
    unavailable = ["unknown", "no data available", "nan", "none", "-1", "nicht gemeldet"]
    
    if ans_str.strip().lower() in unavailable:
        if "status" in q:
            return "The order status information is not available in the system"
        if "stock" in q or "quantity" in q:
            return "The current stock information is not available in the system"
        if "price" in q or "cost" in q:
            return "The price information is not available in the system"
        if "date" in q:
            return "The date information is not available in the system"
        if "location" in q or "stored" in q:
            return "The storage location information is not available in the system"
        if "workstation" in q:
            return "The workstation information is not available in the system"
        if "material" in q:
            return "The material information is not available in the system"
        if "article" in q:
            return "The article information is not available in the system"
        if "company id" in q:
            return "The company ID information is not available in the system"
        if "description" in q:
            return "The order description is not available in the system"
        if "quantity" in q:
            return "The quantity information is not available in the system"
        return "The requested information is not available in the system"

    # Article storage questions
    if "where is article" in q and "stored" in q:
        return f"The article is stored at location {ans_str}"
    if "in which place is the article" in q:
        return f"The article is located at {ans_str}"

    # Resource optimization questions
    if "average processing time and total utilization" in q:
        avg_time, total_time = ans_str.split(", ")
        return f"Average processing time: {avg_time}, Total utilization: {total_time}"

    # Inventory management questions
    if "average time material" in q and "spends in inventory" in q:
        return f"Average inventory time: {ans_str}"

    # Process flow questions
    if "typical sequence of workstations" in q:
        return f"Found {ans_str} following this sequence"
    
    # Performance metrics questions
    if "total utilization time" in q:
        return f"Total utilization time: {ans_str}"

    # Cross reference questions
    if "how many orders use material" in q:
        return f"This material is used in {ans_str} orders"

    # Warehouse questions
    if "list the top 10 warehouse positions" in q:
        try:
            positions = json.loads(ans_str)
            return f"Top 10 warehouse positions: {', '.join(f'{pos[0]} ({pos[1]} articles)' for pos in positions)}"
        except:
            return ans_str

    # Order info questions
    if q.startswith("give me the info about the order"):
        return f"Here is the order information: {ans_str}"

    # Company ID questions
    if "what is the company id" in q:
        return f"The company ID is {ans_str}"

    # Order description questions
    if "what is the description" in q:
        return f"The order description is: {ans_str}"

    # Order quantity questions
    if "what is the number of orders" in q:
        return f"The production quantity is {ans_str}"

    # Drawing number questions
    if "what is the drawing no" in q:
        return f"The drawing number is {ans_str}"
    if q.startswith("give me the newest order with the drawing number"):
        return f"The newest order is {ans_str}"
    if "when was the first order with this drawing number" in q:
        return f"The first order was placed on {ans_str}"
    if "how many orders with the drawing number" in q:
        return f"There are {ans_str} orders with this drawing number"

    # Date related questions
    if "how far is the order" in q:
        return f"The order was created on {ans_str}"
    if "when is the delivery date" in q:
        return f"The delivery date is {ans_str}"

    # Material storage questions
    if "where is material" in q and "stored" in q:
        return f"The material is stored at location {ans_str}"

    # Order storage questions
    if "where is order" in q and "stored" in q:
        return f"The order is stored at location {ans_str}"

    # Price and cost related questions
    if "what is the price" in q and "order" in q:
        return f"The price of the order is {ans_str}"
    if "what is the price" in q and "cost" in q:
        return f"The total cost is {ans_str}"
    if "what does the article" in q and "cost" in q:
        return f"The article cost is {ans_str}"
    if "average cost" in q and "order" in q:
        return f"Average cost: {ans_str}"

    # Status questions
    if "what is the status of order" in q:
        if ans_str == "0":
            return "The order status is: Not triggered"
        elif ans_str == "1":
            return "The order status is: Triggered"
        elif ans_str == "100":
            return "The order status is: On hold"
        return f"The current status is {ans_str}"

    # Workstation questions
    if "at which workstation" in q:
        return f"The order is currently at workstation {ans_str}"
    if "which workstations are involved" in q:
        return f"The following workstations are involved: {ans_str}"

    # Requirements questions
    if "which articles do i need" in q:
        return f"Required articles: {ans_str}"
    if "which materials do i need" in q:
        return f"Required materials: {ans_str}"

    # Stock and quantity questions
    if "what is the stock level" in q:
        return f"Current stock level: {ans_str}"
    if "what is the stock quantity" in q:
        return f"Current stock quantity: {ans_str}"
    if "how much stock is available" in q:
        return f"Available stock: {ans_str}"

    # Warehouse questions
    if "what is currently in warehouse position" in q:
        return f"Warehouse contents: {ans_str}"
    if "list the top 10 warehouse positions" in q:
        return f"Top 10 warehouse positions: {ans_str}"
    if "which warehouse position contains the most" in q:
        return f"Most diverse position: {ans_str}"
    if "sort the top 10 warehouse positions" in q:
        return f"Sorted warehouse positions: {ans_str}"

    # Delivery questions
    if "how many times was article" in q and "delivered after" in q:
        return f"Delivery count after date: {ans_str}"
    if "how often was the article" in q and "delivered in 2024" in q:
        return f"2024 delivery frequency: {ans_str}"

    # Logbook questions
    if "who reported work on order" in q:
        return f"Work was reported by: {ans_str}"
    if "status of the latest logbook entry" in q:
        return f"Latest logbook status: {ans_str}"

    # Work order questions
    if "what is the total time planned" in q:
        return f"Total planned time: {ans_str}"

    # Fallback
    return ans_str

def generate_answer_long(question, answer):
    """
    Generate a detailed answer based on the question context.
    """
    q = question.lower()
    ans_str = str(answer)

    # Handle "No data available" cases more gracefully
    if ans_str in ["No data available", "-1", "Unknown", "None", "nan", "nicht gemeldet"]:
        if "status" in q:
            return "The order status information is not available in the system"
        if "stock" in q or "quantity" in q:
            return "The current stock information is not available in the system"
        if "price" in q or "cost" in q:
            return "The price information is not available in the system"
        if "date" in q:
            return "The date information is not available in the system"
        if "location" in q or "stored" in q:
            return "The storage location information is not available in the system"
        if "workstation" in q:
            return "The workstation information is not available in the system"
        if "material" in q:
            return "The material information is not available in the system"
        if "article" in q:
            return "The article information is not available in the system"
        if "company id" in q:
            return "The company ID information is not available in the system"
        if "description" in q:
            return "The order description is not available in the system"
        if "quantity" in q:
            return "The quantity information is not available in the system"
        return "The requested information is not available in the system"

    # Article storage questions
    if "where is article" in q and "stored" in q:
        return f"This article is stored at warehouse location {ans_str}"
    if "in which place is the article" in q:
        return f"The article is currently located at warehouse position {ans_str}"

    # Resource optimization questions
    if "average processing time and total utilization" in q:
        avg_time, total_time = ans_str.split(", ")
        return f"The workstation processes orders in an average of {avg_time}, with a total utilization time of {total_time}"

    # Inventory management questions
    if "average time material" in q and "spends in inventory" in q:
        return f"This material typically remains in stock for {ans_str}"

    # Process flow questions
    if "typical sequence of workstations" in q:
        return f"Based on the production data analysis, {ans_str} have been processed following this workstation sequence"
    
    # Performance metrics questions
    if "total utilization time" in q:
        return f"The workstation has been utilized for a total of {ans_str}"

    # Cross reference questions
    if "how many orders use material" in q:
        return f"Based on the production data, this material is currently being used in {ans_str} different production orders"

    # Warehouse questions
    if "list the top 10 warehouse positions" in q:
        try:
            positions = json.loads(ans_str)
            return f"Here are the top 10 warehouse positions by article count: {', '.join(f'{pos[0]} with {pos[1]} articles' for pos in positions)}"
        except:
            return ans_str

    # Order info questions
    if q.startswith("give me the info about the order"):
        return f"Here is the complete information for the requested order: {ans_str}"

    # Company ID questions
    if "what is the company id" in q:
        return f"The company ID assigned to this order is {ans_str}"

    # Order description questions
    if "what is the description" in q:
        return f"The complete order description is as follows: {ans_str}"

    # Order quantity questions
    if "what is the number of orders" in q:
        return f"The total production quantity for this order is {ans_str}"

    # Drawing number questions
    if "what is the drawing no" in q:
        return f"The drawing number assigned to this order is {ans_str}"
    if q.startswith("give me the newest order with the drawing number"):
        return f"The newest order with this drawing number is {ans_str}"
    if "when was the first order with this drawing number" in q:
        return f"The first order with this drawing number was placed on {ans_str}"
    if "how many orders with the drawing number" in q:
        return f"There are {ans_str} orders associated with this drawing number"

    # Date related questions
    if "how far is the order" in q:
        return f"The order was created in the ERP system on {ans_str}"
    if "when is the delivery date" in q:
        return f"The scheduled delivery date for this order is {ans_str}"

    # Material storage questions
    if "where is material" in q and "stored" in q:
        return f"This material is stored at warehouse location {ans_str}"

    # Order storage questions
    if "where is order" in q and "stored" in q:
        return f"This order is stored at warehouse location {ans_str}"

    # Price and cost related questions
    if "what is the price" in q and "order" in q:
        return f"The total price for this order is {ans_str}"
    if "what is the price" in q and "cost" in q:
        return f"The total cost for this order is {ans_str}"
    if "what does the article" in q and "cost" in q:
        return f"The cost for this article is {ans_str}"
    if "average cost" in q and "order" in q:
        return f"The average cost for this order type is: {ans_str}"

    # Status questions
    if "what is the status of order" in q:
        if ans_str == "0":
            return "The order status is: Not triggered"
        elif ans_str == "1":
            return "The order status is: Triggered"
        elif ans_str == "100":
            return "The order status is: On hold"
        return f"The current status of the order is {ans_str}"

    # Workstation questions
    if "at which workstation" in q:
        return f"The order is currently being processed at workstation {ans_str}"
    if "which workstations are involved" in q:
        return f"The following workstations are involved in the production process: {ans_str}"

    # Requirements questions
    if "which articles do i need" in q:
        return f"The following articles are required for this order: {ans_str}"
    if "which materials do i need" in q:
        return f"The required materials for this order are: {ans_str}"

    # Stock and quantity questions
    if "what is the stock level" in q:
        return f"The current stock level for this material is {ans_str}"
    if "what is the stock quantity" in q:
        return f"The current stock quantity for this article is {ans_str}"
    if "how much stock is available" in q:
        return f"The available stock for this article is {ans_str}"

    # Warehouse questions
    if "what is currently in warehouse position" in q:
        return f"The contents of this warehouse position are: {ans_str}"
    if "list the top 10 warehouse positions" in q:
        return f"Here are the top 10 warehouse positions by article count: {ans_str}"
    if "which warehouse position contains the most" in q:
        return f"The warehouse position with the most different articles is {ans_str}"
    if "sort the top 10 warehouse positions" in q:
        return f"Here are the sorted top 10 warehouse positions: {ans_str}"

    # Delivery questions
    if "how many times was article" in q and "delivered after" in q:
        return f"The article was delivered {ans_str} times after the specified date"
    if "how often was the article" in q and "delivered in 2024" in q:
        return f"The article was delivered {ans_str} times during 2024"

    # Logbook questions
    if "who reported work on order" in q:
        return f"The work on this order was reported by: {ans_str}"
    if "status of the latest logbook entry" in q:
        return f"The latest logbook entry shows the status as {ans_str}"

    # Work order questions
    if "what is the total time planned" in q:
        return f"The total planned time for all work orders is {ans_str}"

    # Fallback
    return f"{ans_str}"

def has_default_placeholder(text) -> bool:
    """
    Check if a string (question or answer) contains default placeholder values,
    including masked values like {WAREHOUSEPOS}, {nan}, etc.

    Handles str, float, int, list, dict, or None safely.
    """
    default_values = {
        "warehousepos", "unknown", "not available", "no data available",
        "none", "null", "nan", "-1", "0", "0.0"
    }

    # Patterns like {nan}, {UNKNOWN}, {0}, etc.
    placeholder_pattern = re.compile(r"\{(.*?)\}")

    # Handle None or NaN
    if text is None:
        return True
    if isinstance(text, float) and np.isnan(text):
        return True

    # Empty containers
    if isinstance(text, (list, dict)):
        return not text

    # Convert to string
    text_str = str(text).strip().lower()

    # Direct match with meaningless values
    if text_str in default_values:
        return True

    # Check placeholder content
    matches = placeholder_pattern.findall(text_str)
    for match in matches:
        match_str = match.strip().lower()
        if match_str in default_values or match_str == "nan":
            return True
        # Check if the match is a valid number or date
        try:
            float(match_str)
            continue  # Valid number, not a placeholder
        except ValueError:
            try:
                datetime.strptime(match_str, "%Y-%m-%dT%H:%M:%S")
                continue  # Valid date, not a placeholder
            except ValueError:
                try:
                    datetime.strptime(match_str, "%Y-%m-%d")
                    continue  # Valid date, not a placeholder
                except ValueError:
                    pass  # Not a valid date, might be a placeholder

    return False

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
    

    # (template, extractor, category, subcategory)
    template_funcs = [
        ("Give me the info about the order {{{}}}.", lambda p: {k: v for k, v in p.items() if k != "work_orders"}, "order_info", "full_order"),
        ("What is the description of the {{{}}}.", lambda p: p.get("orderdesc", "Not Available"), "order_description", "description"),
        ("What is the current status and progress of order {{{}}} in the ERP system?", lambda p: f"Status: {p.get('orderstatus', 'Not Available')}, Created on: {p.get('createdate', 'Not Available')}", "order_progress", "creation_date"),
        ("At which workstation is order {{{}}} currently?", lambda p: p.get("currentworkstation", "Not Available"), "order_workstation", "current_workstation"),
        ("What is the number of orders for {{{}}}?", lambda p: p.get("productionquantity", "Not Available"), "order_quantity", "production_quantity"),
        ("What is the price of the order {{{}}}?", lambda p: p.get("price", "Not Available"), "order_price", "price"),
        ("What is the drawing no for {{{}}}?", lambda p: p.get("drawno", "Not Available"), "drawing_number", "drawno"),
        ("When is the delivery date for order {{{}}}?", lambda p: p.get("deliverydate", "Not Available"), "delivery_date", "deliverydate"),
        ("What is the company ID for order {{{}}}?", lambda p: p.get("companyid", "Not Available"), "company_id", "companyid"),
        ("Where is order {{{}}} stored?", lambda p: p.get("warehousepos", "Not Available"), "order_storage", "warehousepos"),
        ("What is the status of order {{{}}}?", lambda p: p.get("orderstatus", "Not Available"), "order_status", "orderstatus"),
        ("Which articles do I need for order {{{}}}?", lambda p: ", ".join(a.get("productname", "") for a in p.get("articles", [])) or "Not Available", "article_requirements", "articles"),
        ("Which materials do I need for order {{{}}}?", lambda p: ", ".join(m.get("materialname", "") for m in p.get("materials", [])) or "Not Available", "material_requirements", "materials"),
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
            
            # Skip if question contains default placeholders
            if has_default_placeholder(question):
                
                continue
                
            typo_q = generate_typo_question_safe(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            
            # Extract and normalize answer
            prod_record = key_to_prod[key]
            answer_val = extractor(prod_record)
            
            if isinstance(answer_val, float) and pd.isna(answer_val):
                answer_val = "No data available"
            if isinstance(answer_val, list) and len(answer_val) == 0:
                answer_val = "No data available"
            answer_str = str(answer_val)
            if has_default_placeholder(answer_val):
                continue
            # Generate answer variants
            if isinstance(answer_val, (list, dict)) and answer_val != "No data available":
                ans_opt = json.dumps(answer_val, ensure_ascii=False)
                ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
                answer_long = ans_opt
            else:
                # Skip validation for special categories
                if category in ["drawing_number", "drawing_newest_order", "order_progress", "delivery_date", "drawing_first_order"]:
                    answer = answer_val
                else:
                    answer = validate_answer(answer_val, category)
                ans_opt = generate_answer_optimal(question, answer)
                ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
                answer_long = generate_answer_long(question, answer)
            
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
            answer = validate_answer(ans_val, category)
            answer_str = str(answer)
            if has_default_placeholder(answer_str):
                continue
            ans_opt = generate_answer_optimal(q_text, answer)
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_long = generate_answer_long(q_text, answer)
            output.append({
                "question": q_text,
                "question_typo": typo_q,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
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
            answer = validate_answer(ans_val, category)
            answer_str = str(answer)
            if has_default_placeholder(answer_str):
                continue
            ans_opt = generate_answer_optimal(q_text, answer)
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_long = generate_answer_long(q_text, answer)
            output.append({
                "question": q_text,
                "question_typo": typo_q,
                "novelty_handling": back_q,
                "german_question": de_q,
                "answer": answer_str,
                "answer_long": answer_long,
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
                answer = validate_answer(ans_val, category)
                answer_str = str(answer)
                if has_default_placeholder(answer_str):
                    continue
                ans_opt = generate_answer_optimal(q_text, answer)
                ans_de = (postprocess_answer_german(convert_english_to_german(ans_opt)))
                answer_long = generate_answer_long(q_text, answer)
                output.append({
                    "question": q_text,
                    "question_typo": typo_q,
                    "novelty_handling": back_q,
                    "german_question": de_q,
                    "answer": answer_str,
                    "answer_long": answer_long,
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
    data_dir = "data"
    articles_df = pd.read_excel(os.path.join(data_dir, "articles.xlsx"))
    materials_df = pd.read_excel(os.path.join(data_dir, "materials.xlsx"))
    production_orders_df = pd.read_excel(os.path.join(data_dir, "production_orders.xlsx"))
    
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
        
        # Skip if material name contains default placeholders
        if has_default_placeholder(mat_name):
            continue
            
        material_qas = [
            (f"What is the stock level of material {{{mat_name}}}?", row.get('stockquantity', 'Unknown'), "material_stock_level", "stockquantity"),
            (f"Where is material {{{mat_name}}} stored?", row.get('materialwarehousepos', 'Unknown'), "material_storage", "materialwarehousepos")
        ]
        for question, answer_val, category, subcategory in material_qas:
            # Skip if question contains default placeholders
            if has_default_placeholder(question):
                continue
                
            question = re.sub(r"[{}]", "", question)
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            
            answer_str = str(answer_val)
            if has_default_placeholder(answer_str):
                continue
            if isinstance(answer_val, (list, dict)):
                ans_opt = json.dumps(answer_val, ensure_ascii=False)
                ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            else:
                answer = validate_answer(answer_val, category)
                ans_opt = generate_answer_optimal(question, answer)
                ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_long = generate_answer_long(question, answer)
            
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
        answer = validate_answer(answer_val, "drawing_order_count")
        answer_str = str(answer)
        if has_default_placeholder(answer_str):
            continue
        ans_opt = generate_answer_optimal(question, answer)
        ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
        answer_long = generate_answer_long(question, answer)
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
        
        # Skip validation for drawing_newest_order category
        answer_str = str(answer_val)
        if has_default_placeholder(answer_str):
            continue
            
        ans_opt = generate_answer_optimal(question, answer_val)
        ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
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
    first_orders = production_orders_df.dropna(subset=['orderid', 'drawno']).sample(n=min(MAX_PER_TEMPLATE, len(production_orders_df)))
    for _, row in first_orders.iterrows():
        if len(test_data) >= MAX_TOTAL:
            return test_data
        dr = row.get('drawno')
        dt = row.get('createdate')
        if dr and pd.notna(dt):
            question = f"When was the first order with this drawing number {{{dr}}} placed?"
            answer_val = dt
            
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            answer = validate_answer(answer_val, "drawing_first_order")
            answer_str = answer_val
            if has_default_placeholder(answer_str):
                continue
            ans_opt = generate_answer_optimal(question, answer)
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_long = generate_answer_long(question, answer)
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
        
        # Skip if question contains default placeholders
        if has_default_placeholder(question):
            continue
            
        answer_val = mats
        
        typo_q = generate_typo_question_safe(question)
        typo_id = generate_typo_question_mask_only(question)
        back_q = generate_novelty_questions(question)
        de_q = convert_english_to_german(question)
        answer = validate_answer(answer_val, "warehouse_contents")
        ans_opt = json.dumps(answer_val, ensure_ascii=False)
        ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
        answer_str = ans_opt
        if has_default_placeholder(answer_str):
            continue
        answer_long = generate_answer_long(question, answer)
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
        answer = validate_answer(answer_val, category)
        ans_opt = json.dumps(answer_val, ensure_ascii=False)
        ans_de = ans_opt
        answer_str = ans_opt
        if has_default_placeholder(answer_str):
            continue
        answer_long = generate_answer_long(question, answer)
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
            answer = validate_answer(answer_val, "delivery_post_threshold")
            ans_opt = generate_answer_optimal(question, answer)
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_str = str(answer)
            if has_default_placeholder(answer_str):
                continue
            answer_long = generate_answer_long(question, answer)
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
            # Skip if question contains default placeholders
            if has_default_placeholder(question):
                continue
                
            typo_q = generate_typo_question_safe(question)
            typo_id = generate_typo_question_mask_only(question)
            back_q = generate_novelty_questions(question)
            de_q = convert_english_to_german(question)
            
            # Validate answer first
            answer = validate_answer(answer_val, category)
            answer_str = str(answer)
            
            if has_default_placeholder(answer_str):
                continue
                
            # Generate formatted answers
            if category == "article_storage":
                ans_opt = f"The article is stored at location {answer}"
                answer_long = f"This article is stored at warehouse location {answer}"
            else:
                ans_opt = generate_answer_optimal(question, answer)
                answer_long = generate_answer_long(question, answer)
                
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            
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
            answer = validate_answer(answer_val, "delivery_2024_frequency")
            ans_opt = generate_answer_optimal(question, answer)
            ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
            answer_str = str(answer)
            if has_default_placeholder(answer_str):
                continue        
            answer_long = generate_answer_long(question, answer)
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
                is_valid, validated_wo = validate_record(wo, "work_order")
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
            is_valid, validated_prod = validate_record(prod, "production_order")
            if is_valid:
                if validated_prod.get("price"):
                    order_costs[validated_prod["ordername"]].append(validated_prod["price"])
                
                created = validated_prod.get("createdate")
                delivery = validated_prod.get("deliverydate")
                if created and delivery:
                    order_lead_times[validated_prod["ordername"]].append((created, delivery))
        
        # Process logbooks for quality issues
        for log in order_data.get("logbooks", []):
            is_valid, validated_log = validate_record(log, "logbook")
            if is_valid:
                status = safe_lower(validated_log["status"])
                if "error" in status or "issue" in status or "problem" in status:
                    quality_issues[order_id] += 1
    
    qa_pairs = []
    

    # 1. Process Flow Analysis
    sequence_keys = list(workstation_sequences.keys())
    if len(sequence_keys) > MAX_PER_ANALYTICAL_CATEGORY:
        sequence_keys = random.sample(sequence_keys, MAX_PER_ANALYTICAL_CATEGORY)
    for sequence in sequence_keys:
        orders = workstation_sequences[sequence]
        if len(orders) > 1:  # Only consider sequences that appear multiple times
            question = f"Can you find the orders that follow this workstation sequence: {sequence}?"
            answer = f"This sequence is followed by {len(orders)} orders. The orders are: {', '.join(orders)}. "
            if has_default_placeholder(answer):
                continue
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
            answer = f"{avg_time:.2f} seconds, {total_time:.2f} seconds"
            if has_default_placeholder(answer):
                continue
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
            answer = f"{issues}"
            if has_default_placeholder(answer):
                continue
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "category": "quality_metrics",
                "subcategory": "issue_tracking"
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
                answer = f"{avg_time/86400:.2f} days"
                if has_default_placeholder(answer):
                    continue
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
                answer = f"{avg_lead_time/86400:.2f} days"
                if has_default_placeholder(answer):
                    continue
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
            answer = f"{usage_count}"
            if has_default_placeholder(answer):
                continue
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
        answer = f"{utilization:.2f} seconds"
        if has_default_placeholder(answer):
            continue
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
        answer = validate_answer(qa["answer"], qa["category"])
        answer_str = str(answer)

        ans_opt = generate_answer_optimal(qa["question"], answer)
        ans_de = postprocess_answer_german(convert_english_to_german(ans_opt))
        answer_long = generate_answer_long(qa["question"], answer)

        final_qa_pairs.append({
            "question": qa["question"],
            "question_typo": typo_q,
            "novelty_handling": back_q,
            "german_question": de_q,
            "answer": answer_str,
            "answer_long": answer_long,
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
    Skip time-related questions for now.
    """
    # Generate from both sources
    tree_qa = generate_qa_pairs_from_tree()
    db_qa = generate_qa_pairs_from_database()
    analytical_qa = generate_analytical_answers()
    
    # Combine all QA pairs
    all_qa = tree_qa + db_qa + analytical_qa
    
    # Define time-related categories to skip
    time_related_categories = {
        'work_order_time',
        'resource_optimization',  # Contains average processing time
        'production_planning',    # Contains lead time analysis
        'performance_metrics',    # Contains utilization time
    }
    
    # Filter out time-related questions
    filtered_qa = [qa for qa in all_qa if qa.get("category") not in time_related_categories]
    
    # Categorize questions
    categorized_qa = defaultdict(list)
    for qa in filtered_qa:
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
    logger.info(f"Generated balanced dataset with {len(balanced_qa)} questions (excluding time-related questions)")
    for category, qa_list in categorized_qa.items():
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        logger.info(f"Category '{category}' (weight: {weight}): {len(qa_list)} questions")
    
    return balanced_qa

def generate_english_only_dataset():
    """
    Generate a subset of 100 questions in English only, excluding time-related questions.
    This dataset will only include the original English questions without German translations
    or typo variants.
    """
    # Generate the full balanced dataset
    full_dataset = generate_balanced_test_dataset()
    
    # Filter out time-related categories
    time_related_categories = {
        'work_order_time',
        'resource_optimization',  # Contains average processing time
        'production_planning',    # Contains lead time analysis
        'performance_metrics',    # Contains utilization time
    }
    
    # Create English-only version of each QA pair
    english_only_qa = []
    for qa in full_dataset:
        if qa.get("category") not in time_related_categories:
            english_only_qa.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "answer_long": qa["answer_long"],
                "answer_optimal": qa["answer_optimal"],
                "category": qa["category"],
                "subcategory": qa["subcategory"]
            })
    
    # Sample 100 questions if we have more
    if len(english_only_qa) > 100:
        english_only_qa = random.sample(english_only_qa, 100)
    
    # Log the distribution
    categorized_qa = defaultdict(list)
    for qa in english_only_qa:
        category = qa.get("category", "other")
        categorized_qa[category].append(qa)
    
    logger.info(f"Generated English-only dataset with {len(english_only_qa)} questions")
    for category, qa_list in categorized_qa.items():
        logger.info(f"Category '{category}': {len(qa_list)} questions")
    
    return english_only_qa