"""
ForgeAI Phase 9: Demonstration Dataset Generator
Creates two distinct, realistic multi-format datasets:
1. Demo 1: HR Policy Assistant (PDF + TXT)
2. Demo 2: Ship-Spare Procurement Assistant (CSV + PDF + TXT)
"""

import csv
from pathlib import Path
import pymupdf


def _write_multipage_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), page_text, fontsize=10.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def create_hr_policy_dataset(base_dir: Path = Path("data/demos/hr_policy")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. hr_handbook.pdf (2 pages)
    pdf_pages = [
        (
            "ACME CORP HR HANDBOOK - CHAPTER 1: WORKPLACE & HYBRID POLICY\n\n"
            "1.1 Standard Working Hours\n"
            "Full-time employees work 40 hours per week, Monday through Friday.\n"
            "Standard office hours are 09:00 to 17:30 local time, with mandatory core\n"
            "collaboration hours between 10:00 and 16:00.\n\n"
            "1.2 Hybrid and Remote Work Eligibility\n"
            "After completing the initial 90-day onboarding probation period, eligible\n"
            "employees may work remotely up to 2 days per week with manager approval.\n"
            "New hires receive a one-time home-office equipment stipend of $750 USD\n"
            "for ergonomic chairs, monitors, or desks.\n"
        ),
        (
            "ACME CORP HR HANDBOOK - CHAPTER 2: EXPENSE & CONDUCT POLICY\n\n"
            "2.1 Business Expense Reimbursement\n"
            "All work-related travel and client meal expenses require itemized receipts\n"
            "submitted in the expense portal within 14 calendar days.\n"
            "Single expenses exceeding $250 USD require prior written approval from a\n"
            "department director.\n\n"
            "2.2 Code of Conduct Escalation\n"
            "Policy violations or workplace concerns must be reported to People Operations\n"
            "at people-ops@acmecorp.internal within 5 business days.\n"
        ),
    ]
    _write_multipage_pdf(base_dir / "hr_handbook.pdf", pdf_pages)

    # 2. leave_policy.txt
    (base_dir / "leave_policy.txt").write_text(
        "ACME CORP OFFICIAL LEAVE POLICY\n\n"
        "1. Annual Paid Time Off (PTO):\n"
        "Full-time employees accrue 22 days of paid annual vacation leave per calendar year.\n"
        "A maximum of 5 unused PTO days may be carried over into the next calendar year.\n\n"
        "2. Paid Sick Leave:\n"
        "Employees receive 10 paid sick days per year. Absence exceeding 3 consecutive\n"
        "working days requires a licensed physician's medical certificate.\n\n"
        "3. Parental Leave:\n"
        "All eligible parents (primary and secondary caregivers) receive 16 weeks of\n"
        "100% paid parental leave following birth or adoption.\n\n"
        "4. Bereavement Leave:\n"
        "Employees are granted 5 paid days of bereavement leave for immediate family.\n",
        encoding="utf-8",
    )

    # 3. employee_benefits.txt
    (base_dir / "employee_benefits.txt").write_text(
        "ACME CORP EMPLOYEE BENEFITS GUIDE\n\n"
        "1. Medical, Dental, and Vision Insurance:\n"
        "ACME Corp covers 90% of monthly medical insurance premiums for employees and\n"
        "75% for eligible dependents. Preventive dental and vision care are covered at 100%.\n\n"
        "2. Retirement 401(k) Plan:\n"
        "The company provides a 100% dollar-for-dollar 401(k) match up to 5% of base\n"
        "annual salary with immediate vesting from day one.\n\n"
        "3. Learning & Wellness Stipends:\n"
        "- Annual Learning & Development Stipend: $1,500 USD per year for courses, books, or conferences.\n"
        "- Annual Wellness Reimbursement: $600 USD per year for gym memberships or fitness equipment.\n",
        encoding="utf-8",
    )
    return base_dir


def create_ship_spare_dataset(base_dir: Path = Path("data/demos/ship_spare_procurement")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. product_catalogue.csv
    rows = [
        ["part_id", "part_name", "category", "unit_price_usd", "lead_time_days", "min_order_qty", "stock_status", "supplier_id"],
        ["SP-101", "Main Diesel Fuel Injector", "Fuel System", "1450.00", "7", "2", "In Stock", "SUP-01"],
        ["SP-102", "Turbocharger Cartridge Assembly", "Propulsion", "6800.00", "21", "1", "Low Stock", "SUP-02"],
        ["SP-103", "Seawater Cooling Pump Impeller", "Cooling", "320.00", "5", "4", "In Stock", "SUP-03"],
        ["SP-104", "Cylinder Head Gasket Kit", "Engine Block", "540.00", "10", "2", "In Stock", "SUP-01"],
        ["SP-105", "Auxiliary Bilge Float Switch", "Safety", "115.00", "3", "5", "In Stock", "SUP-03"],
        ["SP-106", "High-Pressure Hydraulic Hose 2m", "Hydraulics", "280.00", "14", "2", "Backorder", "SUP-02"],
    ]
    with open(base_dir / "product_catalogue.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    # 2. supplier_directory.pdf (2 pages)
    pdf_pages = [
        (
            "MARITIME APPROVED SUPPLIER DIRECTORY - SECTION A\n\n"
            "Supplier ID: SUP-01 | Nordic Marine Spares AS (Oslo, Norway)\n"
            "- Payment Terms: Net 30 days.\n"
            "- Certifications: ISO 9001:2015 and DNV-GL Marine Approved.\n"
            "- Emergency Expedite: 48-hour air-freight dispatch incurs an 18% expedite surcharge.\n\n"
            "Supplier ID: SUP-02 | Pacific Propulsion Ltd (Singapore)\n"
            "- Payment Terms: Net 45 days.\n"
            "- Certifications: Class NK and Lloyd's Register certified.\n"
            "- Freight Policy: Free port delivery in Singapore for orders >= $5,000 USD.\n"
        ),
        (
            "MARITIME APPROVED SUPPLIER DIRECTORY - SECTION B\n\n"
            "Supplier ID: SUP-03 | Aegean Valve & Pump Corp (Piraeus, Greece)\n"
            "- Payment Terms: Net 30 days.\n"
            "- Certifications: ISO 14001 and EU-Origin EUR.1 certificate provided.\n"
            "- Warranty: 12-month OEM defect warranty on bronze impellers and switches.\n\n"
            "Procurement Compliance Rule:\n"
            "All purchase orders exceeding $10,000 USD require Fleet Superintendent sign-off.\n"
        ),
    ]
    _write_multipage_pdf(base_dir / "supplier_directory.pdf", pdf_pages)

    # 3. spare_parts_specs.txt
    (base_dir / "spare_parts_specs.txt").write_text(
        "SHIP-SPARE TECHNICAL SPECIFICATION MANUAL\n\n"
        "1. SP-101 Main Diesel Fuel Injector:\n"
        "- Maximum rated injection pressure: 1,800 bar.\n"
        "- Compatible Engine Series: Wartsila 32 & MAN L32/40.\n"
        "- Recommended OEM replacement interval: every 6,000 running hours.\n\n"
        "2. SP-102 Turbocharger Cartridge Assembly:\n"
        "- Maximum continuous exhaust gas temperature: 620 degrees Celsius.\n"
        "- Rotor dynamic balance tolerance: 0.5 g-mm.\n\n"
        "3. SP-103 Seawater Cooling Pump Impeller:\n"
        "- Material: Nickel-Aluminum Bronze & Nitrile composite.\n"
        "- Maximum flow capacity: 140 cubic meters per hour at 1,450 RPM.\n\n"
        "4. SP-104 Cylinder Head Gasket Kit:\n"
        "- Multi-layer stainless steel construction.\n"
        "- Final cylinder head bolt torque specification: 240 Nm.\n",
        encoding="utf-8",
    )
    return base_dir


if __name__ == "__main__":
    p1 = create_hr_policy_dataset()
    p2 = create_ship_spare_dataset()
    print(f"Created HR dataset at: {p1}")
    print(f"Created Ship-Spare dataset at: {p2}")
