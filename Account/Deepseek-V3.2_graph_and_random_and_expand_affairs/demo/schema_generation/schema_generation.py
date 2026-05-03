import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI
def load_yaml_prompt(file_path, sop_title, additional_context):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text"]
    text = text.replace("{{sop_title}}", sop_title)
    text = text.replace("{{additional_context}}", additional_context)
    return text


def main():
    # === 输入参数 ===
    sop_title = "Banking Account Opening Business Handling Process"
    additional_context = '''
    ### 1. Business Overview

    *   **Scenario Name:** Retail & Corporate Bank Account Opening
    *   **Industry:** Financial Services / Banking
    *   **Primary Purpose / Value:** To formally establish a contractual relationship between a customer (individual or entity) and the bank, enabling the customer to deposit funds, facilitate transactions, and access financial products. The core value lies in customer acquisition, regulatory compliance (KYC/AML), and setting the foundation for cross-selling future banking services.

    ### 2. Business Types

    *   **Sub-business A: Retail Individual Account Opening (In-Branch):** The traditional process where an individual visits a physical branch to open a checking or savings account with the assistance of a bank teller or personal banker.
    *   **Sub-business B: Digital Remote Account Opening (Mobile/Web):** An automated, self-service process allowing individuals to open accounts via a smartphone app or website using biometric verification and OCR technology.
    *   **Sub-business C: Corporate/Business Account Opening:** A complex process for legal entities (LLCs, Corporations, Sole Proprietorships) requiring extensive documentation to verify business legitimacy, beneficial ownership, and authorized signatories.

    ### 3. Process Description

    *   **Sub-business A Process: Retail Individual Account Opening (In-Branch)**
        1.  **Customer Reception & Needs Assessment:** The banker greets the customer and conducts a brief interview to determine the appropriate account type (e.g., Student Checking, High-Yield Savings) based on their financial needs.
        2.  **Identity Verification (CIP):** The banker requests and validates physical identification documents (Passport, Driver's License) and proof of address (Utility Bill) to satisfy Customer Identification Program (CIP) requirements.
        3.  **Profile Creation & Data Entry:** The banker manually inputs customer demographic data (Name, SSN/Tax ID, Address, Employment status) into the Core Banking System.
        4.  **Regulatory Screening:** The system automatically runs the customer's data against watchlists (OFAC, Politically Exposed Persons) and checking history databases (e.g., ChexSystems) to assess risk.
        5.  **Account Selection & Disclosures:** The banker presents the specific account terms, fee schedules, and interest rates. The customer signs the signature card and consent forms physically or on a signature pad.
        6.  **Initial Deposit Processing:** The customer provides cash or a check for the minimum opening deposit, which the banker processes immediately at the teller station.
        7.  **Welcome Kit Issuance:** The banker provides a starter kit (temporary debit card, checkbook, online banking credentials) and the account number is generated.

    *   **Sub-business B Process: Digital Remote Account Opening (Mobile/Web)**
        1.  **User Registration:** The customer downloads the app or visits the site, entering a mobile number and email address to receive a One-Time Password (OTP) for device verification.
        2.  **ID Capture & OCR Extraction:** The customer takes a photo of the front and back of their government-issued ID. The system uses Optical Character Recognition (OCR) to auto-fill the application form.
        3.  **Liveness Check (Biometrics):** The customer is prompted to take a "video selfie" or perform specific facial movements (blink, turn head) to verify they are a live person and match the photo on the ID.
        4.  **Data Validation & E-KYC:** The system validates the extracted data against external credit bureaus and government databases in real-time to confirm identity and residency.
        5.  **Risk Decisioning:** An automated decision engine evaluates the application. It returns a result: "Approved," "Declined," or "Refer for Manual Review" (based on fraud scores or credit history).
        6.  **Electronic Signature (e-Sign):** The customer reviews the terms and conditions and taps to "sign" the agreement digitally.
        7.  **Funding the Account:** The customer links an external bank account (via Plaid or similar API) or uses a debit card to transfer the initial deposit.
        8.  **Account Activation:** The system generates the account number and routing number, displaying them on the screen immediately.

    *   **Sub-business C Process: Corporate/Business Account Opening**
        1.  **Application Initiation:** The authorized company representative requests an account opening meeting and submits the initial application form outlining the business structure and expected transaction volume.
        2.  **Document Collection:** The banker collects required legal documents: Articles of Incorporation, EIN/Tax ID certification, Operating Agreements, and Business License.
        3.  **Beneficial Ownership Verification (UBO):** The banker identifies all individuals who own 25% or more of the equity interests or have significant control. Personal identification is collected for all Beneficial Owners.
        4.  **Due Diligence Review (CDD/EDD):** The compliance team reviews the nature of the business to ensure it fits the bank's risk appetite (e.g., screening against high-risk industries like gambling or cannabis, if restricted).
        5.  **Signatory Authority Setup:** The banker establishes who has the authority to sign checks, initiate wires, and manage the account. Specimen signatures are collected for all authorized signers.
        6.  **System Entry & Approval:** All data and scanned documents are entered into the onboarding platform. A branch manager or back-office operations officer reviews the file for completeness and approves the account.
        7.  **Account Generation:** The system generates the business account number and initiates the order for business debit cards and check stock.

    ### 4. Key Roles and Responsibilities

    *   **Retail Personal Banker:** Responsible for face-to-face customer interaction, data entry accuracy, initial KYC checks, and cross-selling banking products.
    *   **Compliance / AML Officer:** Responsible for reviewing "Refer for Manual Review" cases, investigating potential watchlist hits (OFAC), and ensuring the bank adheres to the Bank Secrecy Act (BSA).
    *   **Business Relationship Manager:** Responsible for managing corporate client onboarding, understanding complex business structures, and gathering necessary legal documentation.
    *   **System Administrator:** Responsible for maintaining the availability of the digital onboarding platform and configuring the rules for the automated risk decision engine.
    *   **Back-Office Operations:** Responsible for final quality assurance (QA) on document scans and archiving physical files.

    ### 5. System and Tool Support

    *   **Core Banking System (e.g., Fiserv, Jack Henry):** The central database where the account is actually created and the "system of record" for balances and customer data.
    *   **Digital Onboarding Platform (e.g., nCino, Backbase):** The front-end interface that customers or bankers use to input data, which orchestrates the workflow.
    *   **Identity Verification Service (e.g., Jumio, Onfido):** A tool integrated via API to perform ID document authentication, facial comparison, and liveness checks.
    *   **Watchlist Screening Tool (e.g., LexisNexis, World-Check):** Used to screen names against global sanctions lists, terrorist watchlists, and adverse media.
    *   **e-Signature Platform (e.g., DocuSign, Adobe Sign):** Enables the legal capturing of consent and signatures without physical paper.

    ### 6. Risks and Control Points

    *   **Risk Point 1: Identity Theft / Synthetic Fraud:** A bad actor uses stolen or fabricated details to open an account.
        *   *Control:* Implementation of multi-factor authentication (MFA) during signup and biometric liveness detection to ensure the applicant is physically present.
    *   **Risk Point 2: Money Laundering (Placement Phase):** Criminals open accounts to deposit illicit cash.
        *   *Control:* Strict Customer Due Diligence (CDD) and Enhanced Due Diligence (EDD) for high-risk customers; automated transaction monitoring thresholds set upon account opening.
    *   **Risk Point 3: Data Entry Errors:** Incorrect typing of SSN or Name by a banker results in failed tax reporting.
        *   *Control:* Automated validation rules (e.g., SSN format check) and "four-eyes" review (secondary approval) for manual business account entries.
    *   **Risk Point 4: Regulatory Non-Compliance:** Failure to provide required disclosures (e.g., Truth in Savings Act).
        *   *Control:* System-enforced workflows that prevent account finalization until the customer has acknowledged receipt of disclosures via a checkbox or signature.

    ### 7. Typical Metrics and Measurement Methods

    *   **Metric 1: Average Handling Time (AHT) / Cycle Time:** The elapsed time from the start of the application to the generation of the account number.
        *   *Measurement:* Timestamp of "Start Application" minus Timestamp of "Account Created" in system logs.
    *   **Metric 2: Abandonment Rate:** The percentage of potential customers who start the digital application but drop out before completion.
        *   *Measurement:* (Number of incomplete applications / Total applications started) * 100.
    *   **Metric 3: Auto-Approval Rate:** The percentage of digital applications approved by the system without human intervention.
        *   *Measurement:* (Number of auto-approved applications / Total digital applications) * 100.
    *   **Metric 4: First Time Yield (FTY):** The percentage of account applications that are processed correctly the first time without being returned for missing documents or errors.
        *   *Measurement:* (Applications with 0 rework loops / Total applications).

    ### 8. Best Practices or Improvement Suggestions

    *   **Suggestion 1: Implement "Save and Resume" Functionality:** In digital channels, allow customers to save their progress and resume later across different devices (omnichannel experience) to reduce abandonment rates.
    *   **Suggestion 2: Integration of Government Data APIs:** Instead of manual entry or OCR reliance, integrate directly with government tax or motor vehicle databases (where legally permissible) to pull verified data, reducing fraud and entry errors.
    *   **Suggestion 3: Dynamic Questioning:** Configure the onboarding form to show/hide questions based on previous answers (e.g., only ask for "Source of Wealth" if the initial deposit exceeds a certain threshold) to minimize friction for low-risk customers.
    *   **Suggestion 4: Pre-fill for Existing Customers:** If a customer already has a credit card with the bank and wants to open a checking account, pre-fill all known data to shorten the process to just a few clicks.

    '''
    yaml_file = "/Account/claude_graph_and_random_and_expand_affairs/prompts/Schema_Generation.yaml"
    system_prompt = "You are an SOP dataset schema generator. Note that <Field_to_be_obtained></Field_to_be_obtained> should include several final output fields from the SOP, preferably numerical values, that can be used to determine if the process is correct."
    # 初始化客户端
    client = OpenAI(
        api_key="",  # 替换为你的API Key
        base_url="https://"  # API中转地址
    )
    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, additional_context)

    # 创建对话
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="claude-sonnet-4-5-20250929",  # 指定模型
    )

    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
