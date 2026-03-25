#!/usr/bin/env python3
"""
Bank Document Classifier — CLI
───────────────────────────────
Usage:
  python cli.py train                         # Train / retrain model
  python cli.py classify "...document text..."
  python cli.py classify --file path/to/doc.txt
  python cli.py demo                          # Run all 5 doc type demos
"""

import sys
import json
import argparse
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from classifier.model import BankDocumentClassifier, train


# ─── Demo Samples (one per class) ─────────────────────────────────────────────

DEMOS = {
    "loan_application": """
        Personal Loan Application
        Applicant: Ramesh Kumar
        Annual Income: INR 8,40,000
        Employment: Salaried – TCS Ltd
        Requested Loan Amount: INR 5,00,000
        Tenure: 36 months
        CIBIL Score: 762
        Purpose: Home renovation
        Supporting docs: Salary slips (3 months), Form 16, Bank statement (6 months)
    """,
    "kyc_identity": """
        KYC Verification Form
        Customer Name: Priya Sharma
        Date of Birth: 14-Mar-1988
        PAN: ABCPS1234D
        Aadhaar: 9876-5432-1100
        Address Proof: Utility bill (Electricity) – BSES Delhi
        Photo ID: Passport No. Z1234567 valid till 2029
        Politically Exposed Person: No
        Source of Funds: Salary
        Risk Category: Low
    """,
    "account_statement": """
        HDFC Bank — Account Statement
        Account Number: XXXX XXXX 4321
        Statement Period: 01-Nov-2024 to 30-Nov-2024
        Opening Balance: INR 42,350.00

        Date         Narration                       Debit       Credit      Balance
        03-Nov-2024  UPI/Swiggy Order #7821          450.00                  41,900.00
        07-Nov-2024  SALARY CREDIT – INFOSYS                  85,000.00     1,26,900.00
        15-Nov-2024  NEFT/LOAN EMI/HDFC              12,500.00               1,14,400.00
        28-Nov-2024  ATM WITHDRAWAL                   5,000.00               1,09,400.00

        Closing Balance: INR 1,09,400.00
    """,
    "cheque_payment": """
        Cheque No: 004521
        Bank: State Bank of India, MG Road Branch, Bangalore
        Date: 18-Mar-2025
        Pay: ABC Constructions Pvt Ltd
        Amount (Figures): INR 2,75,000/-
        Amount (Words): Two Lakhs Seventy Five Thousand Only
        Account No: 32145678901
        MICR Code: 560002021
        Crossing: Account Payee Not Negotiable
        Drawer Signature: Verified
    """,
    "contract_agreement": """
        HOME LOAN AGREEMENT
        This Agreement is entered into on 10th March 2025 between
        BORROWER: Mr. Anil Verma, residing at 45 Park Street, Mumbai
        LENDER: Axis Bank Limited, Mumbai
        Loan Amount: INR 45,00,000
        Rate of Interest: 8.75% per annum (floating, linked to RLLR)
        Tenure: 240 months (20 years)
        Covenants: Borrower shall not create further charge on property
        Event of Default: Non-payment of 2 consecutive EMIs
        Governing Law: Laws of India; Jurisdiction: Mumbai courts
        Security: Equitable mortgage of property bearing Survey No. 123/A
    """,
}


def run_demo(clf: BankDocumentClassifier):
    print("\n" + "═" * 70)
    print("  DEMO — One document per class")
    print("═" * 70)
    for doc_type, text in DEMOS.items():
        result = clf.classify(text)
        status = "✓" if result["classification"] == doc_type else "✗"
        src = "LOCAL" if result["source"] == "local_model" else "LLM  "
        print(f"\n[{status}] Expected : {doc_type}")
        print(f"    Got      : {result['classification']}  (conf={result['confidence']:.2%}, src={src})")
        if result.get("extracted_fields"):
            print(f"    Fields   : {json.dumps(result['extracted_fields'], indent=14)[1:]}")
    print("\n" + "═" * 70)


def main():
    parser = argparse.ArgumentParser(description="Bank Document Classifier CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Train / retrain the local model")

    cl = sub.add_parser("classify", help="Classify a document")
    cl.add_argument("text", nargs="?", help="Document text inline")
    cl.add_argument("--file", help="Path to a .txt file")

    sub.add_parser("demo", help="Run demo on all 5 document types")

    args = parser.parse_args()

    if args.cmd == "train":
        print("Training model on synthetic dataset...")
        train()
        print("Done.")

    elif args.cmd == "classify":
        clf = BankDocumentClassifier()
        if args.file:
            text = Path(args.file).read_text()
        elif args.text:
            text = args.text
        else:
            print("Paste document text (Ctrl+D to finish):")
            text = sys.stdin.read()

        result = clf.classify(text)
        print(json.dumps(result, indent=2))

    elif args.cmd == "demo":
        clf = BankDocumentClassifier()
        run_demo(clf)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
