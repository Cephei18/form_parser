"""Phase K — externalized semantic vocabulary.

All keyword lists, validation patterns, field expectations and region-preference
priors used by ``semantic_classifier`` live here as data, not buried in code.
This is deliberately separate so the vocabulary can be reviewed, tuned and
(later) swapped for a template- or learned-source without touching logic.

Region-family strings mirror the answer-region / table-cell vocabulary
(UNDERLINE, DOTTED, BROKEN, MULTILINE, GRID, COMB, WHITESPACE, FREEFORM,
TABLE_CELL, SIGNATURE, PHOTO) plus widget families RADIO / CHECKBOX and the
geometric RECTANGLE / VALUE / UNKNOWN.
"""

from __future__ import annotations

# --- Semantic types --------------------------------------------------------- #
PERSON_NAME = "PERSON_NAME"
PHONE = "PHONE"
EMAIL = "EMAIL"
DATE = "DATE"
ADDRESS = "ADDRESS"
MULTILINE_TEXT = "MULTILINE_TEXT"

PAN = "PAN"
AADHAAR = "AADHAAR"
ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
IFSC = "IFSC"
IDENTIFIER = "IDENTIFIER"

AMOUNT = "AMOUNT"
PERCENTAGE = "PERCENTAGE"
NUMBER = "NUMBER"

GENDER = "GENDER"
RELATIONSHIP = "RELATIONSHIP"
MARITAL_STATUS = "MARITAL_STATUS"
CATEGORY = "CATEGORY"

SIGNATURE = "SIGNATURE"
PHOTO = "PHOTO"
TABLE = "TABLE"
FREE_TEXT = "FREE_TEXT"
UNKNOWN = "UNKNOWN"

SEMANTIC_TYPES = frozenset({
    PERSON_NAME, PHONE, EMAIL, DATE, ADDRESS, MULTILINE_TEXT,
    PAN, AADHAAR, ACCOUNT_NUMBER, IFSC, IDENTIFIER,
    AMOUNT, PERCENTAGE, NUMBER,
    GENDER, RELATIONSHIP, MARITAL_STATUS, CATEGORY,
    SIGNATURE, PHOTO, TABLE, FREE_TEXT, UNKNOWN,
})

# --- Keyword gazetteers (ordered most-specific first) ----------------------- #
# Single-token keywords are matched on word boundaries; multi-word phrases are
# matched as substrings of the normalised label.
SEMANTIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (EMAIL, ("email", "e-mail", "email id", "email address", "e mail")),
    (PAN, ("pan", "permanent account number", "pan card")),
    (AADHAAR, ("aadhaar", "aadhar", "adhaar", "uid", "unique identification")),
    (IFSC, ("ifsc", "ifsc code")),
    (ACCOUNT_NUMBER, ("account number", "account no", "a/c no", "a/c number", "acct number", "bank account", "account #")),
    (DATE, ("date of birth", "dob", "date", "dd/mm/yyyy", "dd-mm-yyyy", "birth date")),
    (PHONE, ("mobile number", "mobile", "phone", "contact number", "telephone", "cell number", "mobile no")),
    (GENDER, ("gender", "sex")),
    (MARITAL_STATUS, ("marital status", "married/unmarried", "marital")),
    (RELATIONSHIP, ("relationship", "relation with applicant", "relation", "relationship with")),
    (PERCENTAGE, ("percentage", "percent", "share %", "allocation %", "% share", "share percentage")),
    (AMOUNT, ("amount", "premium", "investment amount", "sum assured", "rupees", "amount (rs", "amount in rs")),
    (ADDRESS, ("communication address", "permanent address", "correspondence address", "residential address", "address")),
    (PERSON_NAME, ("full name", "first name", "last name", "middle name", "applicant name", "father name",
                   "father's name", "mother name", "mother's name", "nominee name", "guardian name", "name of",
                   "name")),
    (IDENTIFIER, ("registration number", "reference number", "ref no", "folio number", "folio no", "customer id",
                  "client id", "identification number", "passport number", "id number")),
    (NUMBER, ("number of", "no. of", "units", "quantity", "age", "count")),
    (SIGNATURE, ("signature", "sign here", "signed", "authorised signatory", "authorized signatory")),
    (PHOTO, ("photograph", "photo", "passport size photo", "affix photo", "paste photo")),
    (MULTILINE_TEXT, ("remarks", "comments", "description", "particulars", "details", "notes", "reason", "narration")),
    (CATEGORY, ("category", "type", "class", "occupation")),
]

# --- Validation patterns ---------------------------------------------------- #
VALIDATION_PATTERNS: dict[str, str] = {
    "PAN": r"^[A-Za-z]{5}[0-9]{4}[A-Za-z]$",
    "AADHAAR": r"^\d{4}\s?\d{4}\s?\d{4}$",
    "IFSC": r"^[A-Za-z]{4}0[A-Za-z0-9]{6}$",
    "EMAIL": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "PHONE": r"^\+?\d{10,13}$",
    "DATE": r"^\d{2}[/-]\d{2}[/-]\d{4}$",
    "NUMERIC": r"^\d+(\.\d+)?$",
    "PERCENT": r"^\d{1,3}(\.\d+)?\s?%?$",
    "DIGITS": r"^\d+$",
}

# --- Field expectations ----------------------------------------------------- #
# preferred_widget, preferred_region_types, expected_length, validation key.
EXPECTATIONS: dict[str, dict] = {
    PAN: {"widget": "comb", "regions": ["COMB", "GRID"], "length": 10, "validation": "PAN"},
    AADHAAR: {"widget": "comb", "regions": ["COMB", "GRID"], "length": 12, "validation": "AADHAAR"},
    IFSC: {"widget": "comb", "regions": ["COMB", "GRID"], "length": 11, "validation": "IFSC"},
    ACCOUNT_NUMBER: {"widget": "comb", "regions": ["COMB", "GRID", "UNDERLINE"], "length": None, "validation": "DIGITS"},
    IDENTIFIER: {"widget": "comb", "regions": ["COMB", "GRID", "UNDERLINE"], "length": None, "validation": None},
    DATE: {"widget": "comb", "regions": ["COMB", "GRID", "UNDERLINE"], "length": 8, "validation": "DATE"},
    PHONE: {"widget": "comb", "regions": ["COMB", "UNDERLINE"], "length": 10, "validation": "PHONE"},
    EMAIL: {"widget": "text", "regions": ["UNDERLINE", "WHITESPACE"], "length": None, "validation": "EMAIL"},
    PERSON_NAME: {"widget": "text", "regions": ["UNDERLINE", "WHITESPACE"], "length": None, "validation": None},
    ADDRESS: {"widget": "multiline", "regions": ["MULTILINE", "FREEFORM"], "length": None, "validation": None},
    MULTILINE_TEXT: {"widget": "multiline", "regions": ["MULTILINE", "FREEFORM"], "length": None, "validation": None},
    GENDER: {"widget": "radio", "regions": ["RADIO", "CHECKBOX"], "length": None, "validation": None},
    MARITAL_STATUS: {"widget": "radio", "regions": ["RADIO", "CHECKBOX"], "length": None, "validation": None},
    CATEGORY: {"widget": "radio", "regions": ["RADIO", "CHECKBOX"], "length": None, "validation": None},
    RELATIONSHIP: {"widget": "text", "regions": ["UNDERLINE", "WHITESPACE"], "length": None, "validation": None},
    AMOUNT: {"widget": "text", "regions": ["COMB", "UNDERLINE"], "length": None, "validation": "NUMERIC"},
    PERCENTAGE: {"widget": "text", "regions": ["COMB", "UNDERLINE"], "length": None, "validation": "PERCENT"},
    NUMBER: {"widget": "text", "regions": ["COMB", "UNDERLINE"], "length": None, "validation": "NUMERIC"},
    SIGNATURE: {"widget": "signature", "regions": ["SIGNATURE", "UNDERLINE"], "length": None, "validation": None},
    PHOTO: {"widget": "photo", "regions": ["PHOTO"], "length": None, "validation": None},
    TABLE: {"widget": None, "regions": ["TABLE_CELL"], "length": None, "validation": None},
    FREE_TEXT: {"widget": "text", "regions": ["UNDERLINE", "WHITESPACE"], "length": None, "validation": None},
    UNKNOWN: {"widget": None, "regions": [], "length": None, "validation": None},
}

# --- Region-preference priors ----------------------------------------------- #
# semantic_type -> {region_family: score delta}. Positive = prefer, negative =
# penalise. Applied to candidate scores (scaled by classifier confidence) so a
# field steers toward geometrically-plausible regions for its meaning.
REGION_PRIORS: dict[str, dict[str, float]] = {
    ADDRESS: {"MULTILINE": 0.40, "FREEFORM": 0.30, "WHITESPACE": 0.15, "COMB": -0.50, "GRID": -0.40, "DOTTED": -0.10},
    MULTILINE_TEXT: {"MULTILINE": 0.40, "FREEFORM": 0.35, "WHITESPACE": 0.15, "COMB": -0.50, "GRID": -0.40},
    DATE: {"COMB": 0.40, "GRID": 0.30, "FREEFORM": -0.40, "MULTILINE": -0.30},
    PAN: {"COMB": 0.40, "GRID": 0.30, "MULTILINE": -0.30, "FREEFORM": -0.30},
    AADHAAR: {"COMB": 0.45, "GRID": 0.30, "MULTILINE": -0.30, "FREEFORM": -0.30},
    IFSC: {"COMB": 0.35, "GRID": 0.25, "MULTILINE": -0.25, "FREEFORM": -0.25},
    ACCOUNT_NUMBER: {"COMB": 0.30, "GRID": 0.20, "MULTILINE": -0.20},
    IDENTIFIER: {"COMB": 0.30, "GRID": 0.20, "MULTILINE": -0.20},
    PHONE: {"COMB": 0.25, "GRID": 0.15, "MULTILINE": -0.20},
    AMOUNT: {"COMB": 0.10, "MULTILINE": -0.20, "FREEFORM": -0.20},
    PERCENTAGE: {"COMB": 0.10, "MULTILINE": -0.20},
    NUMBER: {"COMB": 0.10, "MULTILINE": -0.15},
    GENDER: {"RADIO": 0.50, "CHECKBOX": 0.30, "MULTILINE": -0.30, "COMB": -0.30},
    MARITAL_STATUS: {"RADIO": 0.50, "CHECKBOX": 0.30, "MULTILINE": -0.30},
    CATEGORY: {"RADIO": 0.40, "CHECKBOX": 0.25, "MULTILINE": -0.20},
    PERSON_NAME: {"UNDERLINE": 0.10, "WHITESPACE": 0.05, "MULTILINE": -0.20, "COMB": -0.10},
    RELATIONSHIP: {"UNDERLINE": 0.10, "MULTILINE": -0.10},
    EMAIL: {"UNDERLINE": 0.10, "WHITESPACE": 0.05, "COMB": -0.20, "MULTILINE": -0.20},
    SIGNATURE: {"SIGNATURE": 0.50, "UNDERLINE": 0.20, "COMB": -0.30, "MULTILINE": -0.10},
    PHOTO: {"PHOTO": 0.50, "RECTANGLE": 0.20},
}
