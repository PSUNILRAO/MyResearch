"""
Bank Document Classifier
────────────────────────
Small ML model (TF-IDF + SVM) with confidence gating.
If confidence < threshold, silently falls back to Claude LLM.
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DOCUMENT_CLASSES = [
    "loan_application",
    "kyc_identity",
    "account_statement",
    "cheque_payment",
    "contract_agreement",
]

CONFIDENCE_THRESHOLD = 0.95  # Below this → silent LLM fallback
MODEL_PATH = Path(__file__).parent / "saved_model.pkl"

# ─── Synthetic Training Data ───────────────────────────────────────────────────
# Used to bootstrap the model before real labelled data is available.
# Each entry is (text_snippet, label). Expand these lists with real examples
# as you collect them — more data = higher accuracy.

TRAINING_DATA = [
    # ── Loan Applications ──────────────────────────────────────────────────────
    ("personal loan application form borrower name annual income employment status credit score requested amount repayment tenure", "loan_application"),
    ("home loan mortgage application property address down payment co-applicant income verification loan to value ratio", "loan_application"),
    ("business loan application company registration turnover collateral DSCR working capital facility overdraft", "loan_application"),
    ("vehicle loan application car model on-road price dealer invoice hypothecation insurance comprehensive", "loan_application"),
    ("education loan application student name college admission letter course fees tuition scholarship disbursement", "loan_application"),
    ("loan application EMI schedule principal outstanding balance prepayment charges foreclosure noc", "loan_application"),
    ("applicant salary slip form 16 income tax returns ITR bank statement six months employment letter", "loan_application"),
    ("credit appraisal memo underwriting decision risk grade bureau score CIBIL Experian repayment capacity", "loan_application"),
    ("loan sanction letter approved amount interest rate floating fixed processing fee disbursement date", "loan_application"),
    ("personal loan pre-approved offer customer id branch code bureau pull consent form signed", "loan_application"),

    # ── KYC / Identity Docs ───────────────────────────────────────────────────
    ("know your customer KYC form aadhaar number PAN card passport photograph signature date of birth", "kyc_identity"),
    ("customer identification program CIP proof of identity proof of address utility bill voter id", "kyc_identity"),
    ("anti money laundering AML politically exposed person PEP declaration source of funds occupation", "kyc_identity"),
    ("re-kyc periodic update customer due diligence risk category high medium low FATF compliance", "kyc_identity"),
    ("driving license national id card biometric fingerprint iris scan digital KYC video kyc vkyc", "kyc_identity"),
    ("passport number expiry date nationality country of birth visa stamp consulate embassy", "kyc_identity"),
    ("aadhaar based ekyc otp authentication uid buid masked aadhaar xml uidai nsdl", "kyc_identity"),
    ("ckyc central kyc registry cersai token number download fetch latest records", "kyc_identity"),
    ("beneficial owner ultimate beneficial ownership UBO trust deed shareholding pattern", "kyc_identity"),
    ("foreign account tax compliance act FATCA CRS common reporting standard self certification form", "kyc_identity"),

    # ── Account Statements ────────────────────────────────────────────────────
    ("account statement opening balance closing balance debit credit transaction date narration", "account_statement"),
    ("bank statement current account savings account CASA passbook mini statement month wise", "account_statement"),
    ("transaction history cheque number rtgs neft imps upi reference number beneficiary name ifsc", "account_statement"),
    ("monthly statement charges service tax gst interest credited overdraft utilisation sweep", "account_statement"),
    ("quarterly statement average monthly balance AMB minimum balance charges penalty debit", "account_statement"),
    ("account summary fixed deposit recurring deposit maturity value interest rate tenure", "account_statement"),
    ("credit card statement payment due date minimum amount outstanding total dues finance charges", "account_statement"),
    ("loan account statement principal paid interest paid emi due date outstanding principal", "account_statement"),
    ("forex account statement foreign currency usd gbp eur exchange rate conversion charges", "account_statement"),
    ("consolidated statement portfolio holdings mutual funds demat account securities net worth", "account_statement"),

    # ── Cheques / Payment Orders ──────────────────────────────────────────────
    ("cheque leaf payee name amount in words figures account number date bank branch micr code", "cheque_payment"),
    ("demand draft DD payable at beneficiary city issuing branch serial number amount", "cheque_payment"),
    ("pay order banker cheque cashier cheque crossing account payee not negotiable", "cheque_payment"),
    ("cheque return memo dishonour reason insufficient funds signature mismatch account closed", "cheque_payment"),
    ("electronic clearing service ECS nach mandate debit instruction standing instruction SI", "cheque_payment"),
    ("rtgs form remitter account ifsc beneficiary account amount purpose code value date", "cheque_payment"),
    ("neft transfer form online payment instruction batch settlement RBI clearing", "cheque_payment"),
    ("imps upi payment p2p p2m vpa virtual payment address transaction id timestamp", "cheque_payment"),
    ("swift wire transfer correspondent bank bic code foreign outward remittance purpose", "cheque_payment"),
    ("payment advice vendor invoice number tds deduction net payable gl code cost centre", "cheque_payment"),

    # ── Contracts / Agreements ────────────────────────────────────────────────
    ("loan agreement deed executed borrower lender terms conditions interest rate covenants default", "contract_agreement"),
    ("facility agreement sanction letter acceptance covenants financial non-financial event of default", "contract_agreement"),
    ("mortgage deed hypothecation agreement charge creation registered sub-registrar stamp duty", "contract_agreement"),
    ("guarantee agreement personal guarantee corporate guarantee surety indemnity obligor", "contract_agreement"),
    ("service level agreement SLA vendor contract scope of work deliverables penalty liquidated damages", "contract_agreement"),
    ("account opening agreement terms conditions privacy policy data protection consent signature", "contract_agreement"),
    ("lease agreement rent property landlord tenant security deposit termination notice period", "contract_agreement"),
    ("non disclosure agreement NDA confidentiality proprietary information binding arbitration clause", "contract_agreement"),
    ("settlement agreement full final settlement waiver release claims without prejudice", "contract_agreement"),
    ("tripartite agreement builder buyer bank disbursement schedule construction linked plan", "contract_agreement"),
]


# ─── Text Preprocessor ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, remove noise, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """TF-IDF vectoriser + calibrated SVM for probability estimates."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("clf", CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
            cv=3,
            method="sigmoid",
        )),
    ])


def train(extra_samples: Optional[list] = None) -> Pipeline:
    """
    Train on synthetic + any extra real samples provided.
    extra_samples: list of (text, label) tuples from real documents.
    """
    data = list(TRAINING_DATA)
    if extra_samples:
        data.extend(extra_samples)

    texts = [preprocess(t) for t, _ in data]
    labels = [l for _, l in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=DOCUMENT_CLASSES))

    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=5, scoring="accuracy")
    logger.info(f"Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Persist ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved → {MODEL_PATH}")

    return pipeline


def load_model() -> Pipeline:
    """Load persisted model; train fresh if not found."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    logger.info("No saved model found — training from scratch.")
    return train()


# ─── LLM Fallback ─────────────────────────────────────────────────────────────

def llm_classify(text: str) -> dict:
    """
    Silent LLM fallback via Anthropic Claude.
    Returns classification + extracted fields.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    system_prompt = """You are a bank document classification expert.
Classify the document into exactly one of these categories:
  loan_application, kyc_identity, account_statement, cheque_payment, contract_agreement

Also extract 3-5 key fields relevant to the document type.

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
  "classification": "<category>",
  "confidence": <float 0-1>,
  "extracted_fields": {"field_name": "value", ...},
  "reasoning": "<one sentence>"
}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Document text:\n\n{text[:3000]}"}],
    )

    raw = message.content[0].text.strip()
    result = json.loads(raw)
    result["source"] = "llm_fallback"
    return result


# ─── Main Classifier ──────────────────────────────────────────────────────────

class BankDocumentClassifier:
    """
    Unified classifier:
      1. Run small SVM model
      2. If confidence ≥ CONFIDENCE_THRESHOLD → return prediction
      3. Else → silently call LLM and return its result
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold
        self.model = load_model()
        logger.info("BankDocumentClassifier ready.")

    def classify(self, text: str) -> dict:
        clean = preprocess(text)

        # ── Small model prediction ─────────────────────────────────────────
        proba = self.model.predict_proba([clean])[0]
        classes = self.model.classes_
        top_idx = int(np.argmax(proba))
        top_class = classes[top_idx]
        top_conf = float(proba[top_idx])

        all_scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        if top_conf >= self.threshold:
            logger.info(f"[LOCAL MODEL] {top_class} @ {top_conf:.2%}")
            return {
                "classification": top_class,
                "confidence": round(top_conf, 4),
                "all_scores": all_scores,
                "source": "local_model",
                "extracted_fields": {},
                "reasoning": f"Local model confidence {top_conf:.2%} ≥ threshold {self.threshold:.0%}",
            }

        # ── Silent LLM fallback ────────────────────────────────────────────
        logger.info(f"[LLM FALLBACK] local conf={top_conf:.2%} < {self.threshold:.0%}")
        result = llm_classify(text)
        result["local_model_best_guess"] = top_class
        result["local_model_confidence"] = round(top_conf, 4)
        result["all_scores"] = all_scores
        return result

    def retrain(self, new_samples: list):
        """
        Incremental retraining with new labelled samples.
        new_samples: list of (text, label) tuples
        """
        logger.info(f"Retraining with {len(new_samples)} new samples...")
        self.model = train(extra_samples=new_samples)
