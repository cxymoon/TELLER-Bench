import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI
def load_yaml_prompt(file_path, sop_title, additional_context, sample_schema):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text"]
    text = text.replace("{{sop_title}}", sop_title)
    text = text.replace("{{additional_context}}", additional_context)
    text = text.replace("{{sample_schema}}", sample_schema)
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
    sample_schema = '''
        <schema>
        <Existing_fields>
            application_id - string, unique identifier for the account opening application (e.g., "APP_2025041812345").
            application_channel - enum, channel through which the application was submitted (e.g., "in_branch", "mobile_app", "web_portal").
            business_type - enum, type of account opening business (e.g., "retail_individual", "digital_remote", "corporate_business").
            application_start_timestamp - datetime, timestamp when the application was initiated (e.g., "2025-04-18T10:15:30Z").
            customer_type - enum, classification of customer (e.g., "individual", "sole_proprietorship", "llc", "corporation").
            account_type_requested - enum, type of account requested (e.g., "checking", "savings", "student_checking", "high_yield_savings", "business_checking").
            applicant_full_name - string, full legal name of the applicant (e.g., "John Michael Smith").
            applicant_date_of_birth - date, date of birth of the applicant (e.g., "1985-06-15").
            applicant_ssn_tax_id - string, Social Security Number or Tax ID (encrypted/masked) (e.g., "XXX-XX-1234").
            applicant_email - string, email address of the applicant (e.g., "john.smith@email.com").
            applicant_phone - string, phone number of the applicant (e.g., "+1-555-123-4567").
            applicant_residential_address - string, current residential address (e.g., "123 Main St, Apt 4B, New York, NY 10001").
            applicant_employment_status - enum, employment status (e.g., "employed", "self_employed", "unemployed", "student", "retired").
            applicant_annual_income - integer, annual income in USD (e.g., 65000).
            id_document_type - enum, type of identification document provided (e.g., "passport", "drivers_license", "state_id").
            id_document_number - string, identification document number (e.g., "DL123456789").
            id_document_expiry_date - date, expiration date of the ID document (e.g., "2028-06-15").
            id_document_issuing_country - string, country that issued the ID (e.g., "US").
            proof_of_address_type - enum, type of proof of address document (e.g., "utility_bill", "bank_statement", "lease_agreement").
            initial_deposit_amount - float, amount of initial deposit in USD (e.g., 500.00).
            initial_deposit_method - enum, method of initial deposit (e.g., "cash", "check", "external_transfer", "debit_card").
            external_bank_account_linked - boolean, whether an external bank account was linked (e.g., True).
            otp_verification_completed - boolean, whether One-Time Password verification was completed (e.g., True).
            id_ocr_extracted_data - JSON, data extracted from ID via OCR (e.g., {"name": "John Michael Smith", "dob": "1985-06-15", "id_number": "DL123456789", "address": "123 Main St, New York, NY"}).
            biometric_liveness_check_completed - boolean, whether biometric liveness check was performed (e.g., True).
            facial_match_confidence_score - float, confidence score of facial comparison (0-100) (e.g., 96.5).
            business_legal_name - string, legal name of the business entity (for corporate accounts) (e.g., "Acme Technologies LLC").
            business_ein - string, Employer Identification Number (for corporate accounts) (e.g., "12-3456789").
            business_structure - enum, legal structure of the business (e.g., "llc", "corporation", "sole_proprietorship", "partnership").
            business_industry - string, industry classification of the business (e.g., "technology_services").
            business_registration_date - date, date when business was registered (e.g., "2020-03-15").
            business_monthly_transaction_volume_estimate - integer, estimated monthly transaction volume in USD (e.g., 150000).
            beneficial_owners - list[JSON], list of beneficial owners with 25%+ ownership (e.g., [{"name": "Jane Doe", "ownership_percentage": 60, "ssn": "XXX-XX-5678", "dob": "1978-11-20"}]).
            authorized_signers - list[JSON], list of authorized signers for the account (e.g., [{"name": "John Doe", "title": "CEO", "signature_captured": true}]).
            documents_submitted - list[JSON], list of all documents submitted (e.g., [{"doc_type": "articles_of_incorporation", "file_path": "/docs/APP_123_articles.pdf", "upload_timestamp": "2025-04-18T10:30:00Z"}]).
            banker_id - string, identifier of the banker/employee handling the application (e.g., "EMP_8765").
            branch_code - string, code of the branch where application was submitted (e.g., "BR_NYC_001").
            customer_consent_forms_signed - list[string], list of consent forms signed (e.g., ["signature_card", "terms_and_conditions", "truth_in_savings_disclosure", "privacy_notice"]).
            esignature_timestamp - datetime, timestamp of electronic signature (e.g., "2025-04-18T10:45:22Z").
            esignature_ip_address - string, IP address from which e-signature was captured (e.g., "192.168.1.100").
            marketing_opt_in - boolean, whether customer opted in for marketing communications (e.g., False).
            referral_source - enum, how customer learned about the bank (e.g., "online_search", "friend_referral", "branch_walk_in", "advertisement").
            product_bundle_offered - string, any product bundle offered during account opening (e.g., "premium_checking_with_rewards").
            cross_sell_products_accepted - list[string], additional products customer accepted (e.g., ["debit_card", "online_banking", "mobile_app"]).
            application_data_entry_method - enum, how data was entered (e.g., "manual_banker_entry", "ocr_auto_fill", "customer_self_service").
        </Existing_fields>
        
        <Field_to_be_obtained>
            cip_verification_status - enum, result of Customer Identification Program verification (e.g., "verified", "failed", "pending_manual_review").
            watchlist_screening_result - enum, result of OFAC/sanctions screening (e.g., "clear", "potential_match", "confirmed_match").
            chexsystems_score - integer, ChexSystems risk score (300-899 scale) (e.g., 625).
            credit_bureau_validation_status - enum, result of credit bureau identity validation (e.g., "verified", "not_verified", "insufficient_data").
            fraud_risk_score - float, automated fraud risk score (0-100, higher = riskier) (e.g., 15.3).
            auto_decision_result - enum, automated decision engine result (e.g., "auto_approved", "auto_declined", "refer_for_manual_review").
            manual_review_required - boolean, whether manual compliance review is required (e.g., False).
            manual_review_reason - string, reason for manual review if applicable (e.g., "High-risk industry flagged").
            compliance_officer_id - string, identifier of compliance officer who reviewed (e.g., "COMP_4521").
            compliance_review_timestamp - datetime, timestamp of compliance review completion (e.g., "2025-04-18T14:22:15Z").
            compliance_review_notes - string, notes from compliance review (e.g., "Verified source of funds documentation. Approved.").
            due_diligence_level - enum, level of due diligence applied (e.g., "standard_cdd", "enhanced_edd", "simplified_dd").
            final_approval_status - enum, final approval decision (e.g., "approved", "declined", "conditional_approval").
            decline_reason - string, reason for decline if applicable (e.g., "Failed identity verification").
            account_number_generated - string, generated bank account number (e.g., "1234567890").
            routing_number - string, bank routing number assigned (e.g., "021000021").
            account_creation_timestamp - datetime, timestamp when account was created in core banking system (e.g., "2025-04-18T14:30:45Z").
            total_processing_time_minutes - integer, total time from application start to account creation in minutes (e.g., 18).
            first_time_yield - boolean, whether application was processed correctly without rework (e.g., True).
            rework_count - integer, number of times application was returned for corrections (e.g., 0).
            welcome_kit_issued - boolean, whether welcome kit was provided to customer (e.g., True).
            debit_card_number_issued - string, debit card number issued (masked) (e.g., "XXXX-XXXX-XXXX-1234").
            online_banking_credentials_created - boolean, whether online banking access was set up (e.g., True). 
            temporary_checks_issued - boolean, whether temporary checks were provided (e.g., False).  
            account_funding_completed - boolean, whether initial deposit was successfully processed (e.g., True).    
            account_activation_status - enum, current status of the account (e.g., "active", "pending_funding", "dormant", "closed").
            application_abandoned - boolean, whether the application was started but not completed (e.g., False).
            abandonment_stage - string, stage at which application was abandoned if applicable (e.g., "ID_verification").
        </Field_to_be_obtained>
        </schema>
        '''
    yaml_file = "Account/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_Generation.yaml"
    system_prompt = "You are an SOP complete process generator. By executing the SOP you design, several definite execution results, preferably non-random numerical values, should ultimately be obtained. Also, note that all data related to each business application is already stored in the database; therefore, steps such as generating an application_id are not necessary, as this data can be read directly from the database."

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, additional_context, sample_schema)

    # 初始化客户端
    client = OpenAI(
        api_key="",  # 请替换为你的 API Key
        base_url="https://"  # API 中转地址
    )

    # 创建流式对话
    response_chunks = client.chat.completions.create(
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
        max_tokens=49152,
        stream=True  # ➤ 开启流模式
    )

    full_response = ""  # 用于累积完整输出
    # 逐 chunk 读取流式输出
    for chunk in response_chunks:
        # 假设每个 chunk 的结构类似 OpenAI 官方： choices[0].delta.content
        # 你需要根据你客户端库的具体属性名做调整
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            text = delta.content
            print(text, end="", flush=True)     # 实时打印输出
            full_response += text                # 累积内容

    # 当流结束后，full_response 即为完整的回复内容
    print("\n\n【流式输出结束】")

    # 将完整回复写入 txt 文件
    with open("response_output.txt", "w", encoding="utf-8") as file:
        file.write(full_response)

    print("完整回复已保存到 response_output.txt 文件")

if __name__ == "__main__":
    main()
