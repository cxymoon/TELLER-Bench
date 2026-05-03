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
    sop_title = "Banking Deposit Business Handling Process"
    additional_context = '''
        ### 1. Business Overview
        *   **Scenario Name:** Banking Deposit Services Management
        *   **Industry:** Financial Services / Retail Banking
        *   **Primary Purpose / Value:** To provide secure, compliant, and efficient channels for customers to store funds, earning interest while ensuring the bank has the necessary liquidity to lend and invest. This scenario facilitates the primary inflow of capital for a bank's balance sheet.

        ### 2. Business Types
        *   **Retail Demand Deposit (Savings/Checking):** Processing day-to-day transaction account openings where funds are easily accessible.
        *   **Term Deposit (Certificate of Deposit - CD):** Handling fixed-term investments where customers lock in funds for a specific period at a higher interest rate.
        *   **Corporate/Business Deposit:** Managing high-volume deposit accounts for business entities, often requiring complex documentation and authorization structures.
        *   **Cash Deposit Operations:** The physical act of depositing cash via branch tellers or ATMs.

        ### 3. Process Description

        *   **Retail Demand Deposit Process:**
            1)  **Customer Verification (KYC):** The bank officer collects the customer's ID (passport, driver's license), proof of address, and tax identification number. The system performs automatic sanction screening and identity verification.
            2)  **Product Selection & Terms Agreement:** The customer selects a specific account type (e.g., Student Checking, High-Yield Savings). The officer explains fees, minimum balance requirements, and interest rates. The customer signs the digital or physical deposit agreement.
            3)  **Account Creation:** The officer inputs customer data into the Core Banking System (CBS), which generates a unique account number and links it to the customer's unique Customer Information File (CIF).
            4)  **Initial Funding:** The customer provides the opening balance via cash, check, or internal transfer. The system validates the funding source.
            5)  **Access Setup:** Issuance of debit card, checkbook ordering, and setup of online banking credentials.
            6)  **Output Generation:** The system outputs the new Account Number, Initial Balance Value, and Account Status (Active).

        *   **Term Deposit (CD) Process:**
            1)  **Needs Analysis & Rate Locking:** The customer inquires about current rates for different terms (e.g., 6 months, 1 year, 5 years). The system displays the current treasury-approved rates.
            2)  **Fund Allocation:** The customer specifies the principal amount to be invested. The system checks the source account for sufficient funds to cover the principal.
            3)  **Contract Generation:** The system generates a Term Deposit contract specifying the Principal Amount, Maturity Date, Interest Rate (APY), and Early Withdrawal Penalties.
            4)  **Execution & Debiting:** Upon customer approval, the system debits the source account and credits the new Term Deposit GL account.
            5)  **Output Generation:** The system outputs the CD Certificate Number, Maturity Date, Principal Amount, and Calculated Interest at Maturity.

        *   **Cash Deposit Operations (Counter) Process:**
            1)  **Transaction Initiation:** The customer presents a deposit slip (or swipes a debit card) and hands over physical cash to the teller.
            2)  **Cash Validation & Counting:** The teller runs the bills through a currency counter/discriminator to verify authenticity and total count.
            3)  **System Entry:** The teller inputs the account number and the total cash amount into the teller system.
            4)  **Double-Entry Confirmation:** The system validates the account status. The teller confirms the amount matches the machine count.
            5)  **Receipt Generation:** The system processes the credit to the customer's ledger balance. A receipt is printed.
            6)  **Output Generation:** The system outputs the Transaction Reference ID, Updated Account Balance, and Cash Drawer Total.

        ### 4. Key Roles and Responsibilities
        *   **Bank Teller / CSR (Customer Service Representative):** Responsible for face-to-face interaction, physical cash handling, basic identity verification, and data entry for deposits.
        *   **Personal Banker / Relationship Manager:** Handles complex account openings (Term Deposits, Business Accounts), advisory services, and relationship deepening.
        *   **Branch Manager:** Approves large cash transactions, overrides system blocks (within authority limits), and ensures audit compliance.
        *   **Compliance Officer / KYC Analyst:** Reviews flagged applications for potential money laundering (AML) risks or incomplete documentation.
        *   **System Administrator:** Ensures the Core Banking System and peripheral devices (cash counters, card readers) are operational and secure.

        ### 5. System and Tool Support
        *   **Core Banking System (CBS):** The central ledger (e.g., Oracle Flexcube, Temenos, Fiserv) that maintains account balances, customer master data, and processes transactions in real-time.
        *   **Teller Capture System:** The front-end interface used by tellers to input cash and check transactions.
        *   **CRM (Customer Relationship Management):** Stores interaction history and prompts bankers with cross-selling opportunities during the deposit process.
        *   **AML / Sanctions Screening Tool:** Automatically checks customer names against government watchlists (OFAC, Interpol) during onboarding.
        *   **Currency Discriminator / Counter:** Hardware used to count large volumes of cash and detect counterfeit bills.

        ### 6. Risks and Control Points
        *   **Risk Point 1: Money Laundering / Terrorist Financing:** Bad actors using deposits to legitimize illicit funds.
            *   *Control Measure:* Mandatory KYC/CIP (Customer Identification Program) checks; Automatic CTR (Currency Transaction Report) filing for cash deposits over a specific threshold (e.g., $10,000).
        *   **Risk Point 2: Counterfeit Currency:** Acceptance of fake bills resulting in financial loss.
            *   *Control Measure:* Use of certified currency detection machines; manual UV light checks for older bills; teller training on security features.
        *   **Risk Point 3: Data Entry Error:** Depositing funds into the wrong account due to a typo.
            *   *Control Measure:* Name-to-account matching verification (system displays account holder name for confirmation before posting); Dual control/supervisor approval for amounts over a certain limit.
        *   **Risk Point 4: Fraudulent Account Opening:** Identity theft used to open accounts for check fraud.
            *   *Control Measure:* Multi-factor authentication of ID documents (scanning barcodes on IDs); Third-party database verification (e.g., ChexSystems) to check banking history.

        ### 7. Typical Metrics and Measurement Methods
        *   **Average Account Opening Time (Turnaround Time):** Measured by the timestamps between the start of the application in the system and the generation of the account number.
        *   **Net New Money (NNM):** The numerical value of new funds brought into the bank minus funds withdrawn, measured daily or monthly.
        *   **Deposit Mix:** The ratio of low-cost deposits (checking/savings) to high-cost deposits (CDs) measured by aggregating total balances by product code.
        *   **Teller Transaction Accuracy:** The percentage of transactions processed without error corrections or balancing discrepancies at the end of the day.

        ### 8. Best Practices or Improvement Suggestions
        *   **Digital Onboarding Implementation:** Minimize in-branch processing by allowing customers to scan IDs and take selfies via mobile app for OCR data extraction, reducing manual entry errors and processing time.
        *   **Paperless Workflows:** Utilize e-signature pads and email delivery for receipts and contracts to reduce paper costs and storage risks.
        *   **Real-time Fraud Scoring:** Integrate AI-driven fraud detection that scores the risk of a deposit transaction in real-time based on location, device ID, and historical behavior.
        *   **Unified Banker Model:** Train staff to handle both teller duties (transactions) and platform duties (account opening) to optimize workforce utilization during peak and off-peak hours.

    '''
    sample_schema = '''
        <schema>
        <Existing_fields>
            transaction_id - string, unique identifier for the deposit transaction (e.g., "TXN_DEP_20250418_001234").
            customer_id - string, unique customer identifier in the Core Banking System (e.g., "CIF_789456123").
            account_number - string, target account number for the deposit (e.g., "4532100012345678").
            transaction_timestamp - datetime, timestamp when the deposit transaction was initiated (e.g., "2025-04-18T10:23:45Z").
            deposit_type - enum, type of deposit transaction; options: ["RETAIL_DEMAND", "TERM_DEPOSIT", "CORPORATE_DEPOSIT", "CASH_COUNTER", "ATM_DEPOSIT", "MOBILE_DEPOSIT"] (e.g., "CASH_COUNTER").
            account_type - enum, type of account receiving the deposit; options: ["SAVINGS", "CHECKING", "CD_6M", "CD_1Y", "CD_5Y", "BUSINESS_CHECKING", "MONEY_MARKET"] (e.g., "SAVINGS").
            deposit_channel - enum, channel through which deposit is made; options: ["BRANCH_TELLER", "ATM", "MOBILE_APP", "ONLINE_BANKING", "WIRE_TRANSFER"] (e.g., "BRANCH_TELLER").
            branch_code - string, branch identifier where transaction occurred (e.g., "BR_NY_5012").
            teller_id - string, employee ID of the teller or banker handling the transaction (e.g., "EMP_45823").
            customer_name - string, full name of the customer (e.g., "John Michael Smith").
            customer_id_type - enum, type of identification document; options: ["PASSPORT", "DRIVERS_LICENSE", "NATIONAL_ID", "SSN"] (e.g., "DRIVERS_LICENSE").
            customer_id_number - string, identification document number (e.g., "D1234567").
            customer_address - string, current residential address of customer (e.g., "123 Main Street, New York, NY 10001").
            customer_tax_id - string, tax identification number (e.g., "123-45-6789").
            customer_phone - string, contact phone number (e.g., "+1-212-555-0123").
            customer_email - string, email address (e.g., "john.smith@email.com").
            deposit_amount - float, principal amount being deposited in USD (e.g., 5000.00).
            deposit_currency - string, currency code (e.g., "USD").
            funding_source_type - enum, source of funds; options: ["CASH", "CHECK", "INTERNAL_TRANSFER", "WIRE", "ACH"] (e.g., "CASH").
            funding_source_account - string, source account number if internal transfer (e.g., "4532100087654321" or null).
            cash_denomination_breakdown - JSON, breakdown of physical cash by denomination; format: {"100": 20, "50": 10, "20": 15, "10": 5, "5": 10, "1": 25} representing count per denomination (e.g., {"100": 50, "20": 0, "10": 0, "5": 0, "1": 0}).
            term_length_months - integer, duration of term deposit in months; null for non-term deposits (e.g., 12 or null).
            interest_rate_apy - float, annual percentage yield for the deposit product (e.g., 3.75).
            maturity_date - datetime, maturity date for term deposits; null for demand deposits (e.g., "2026-04-18T00:00:00Z" or null).
            early_withdrawal_penalty_rate - float, penalty rate for early CD withdrawal as percentage (e.g., 0.5 or null).
            minimum_balance_requirement - float, minimum balance required to avoid fees (e.g., 500.00).
            monthly_maintenance_fee - float, monthly account maintenance fee in USD (e.g., 5.00).
            previous_account_balance - float, account balance before this deposit transaction (e.g., 1234.56).
            kyc_verification_status - enum, KYC verification completion status; options: ["PENDING", "VERIFIED", "FAILED", "REQUIRES_REVIEW"] (e.g., "VERIFIED").
            kyc_verification_timestamp - datetime, timestamp of KYC verification completion (e.g., "2025-04-18T10:15:32Z").
            aml_screening_result - enum, AML/sanctions screening result; options: ["CLEAR", "WATCHLIST_HIT", "MANUAL_REVIEW_REQUIRED"] (e.g., "CLEAR").
            aml_risk_score - integer, automated risk score from AML system; range: 0-100 (e.g., 15).
            sanctions_screening_timestamp - datetime, timestamp of sanctions screening (e.g., "2025-04-18T10:16:05Z").
            counterfeit_detection_result - enum, result from currency discriminator; options: ["PASS", "FAIL", "NOT_APPLICABLE"] (e.g., "PASS").
            cash_counter_machine_id - string, identifier of currency counting machine used (e.g., "CCM_BR5012_01" or null).
            supervisor_approval_required - boolean, whether transaction requires supervisor override (e.g., false).
            supervisor_id - string, employee ID of approving supervisor; null if not required (e.g., "EMP_12034" or null).
            supervisor_approval_timestamp - datetime, timestamp of supervisor approval; null if not required (e.g., "2025-04-18T10:25:12Z" or null).
            ctr_filing_required - boolean, whether Currency Transaction Report filing is required (e.g., false).
            ctr_threshold_amount - float, threshold amount triggering CTR filing (e.g., 10000.00).
            debit_card_issued - boolean, whether debit card was issued with this account (e.g., true).
            debit_card_number - string, masked debit card number; null if not issued (e.g., "****-****-****-1234" or null).
            online_banking_enabled - boolean, whether online banking access was set up (e.g., true).
            mobile_app_enabled - boolean, whether mobile app access was enabled (e.g., true).
            checkbook_ordered - boolean, whether physical checkbook was ordered (e.g., false).
            cross_sell_products_offered - list[string], additional products offered during interaction (e.g., ["CREDIT_CARD", "PERSONAL_LOAN"]).
            customer_signature_captured - boolean, whether customer signature was obtained (e.g., true).
            signature_capture_method - enum, method of signature capture; options: ["ELECTRONIC_PAD", "PAPER_SCANNED", "MOBILE_APP", "NOT_CAPTURED"] (e.g., "ELECTRONIC_PAD").
            receipt_delivery_method - enum, how receipt was delivered; options: ["PRINTED", "EMAIL", "SMS", "MOBILE_APP"] (e.g., "PRINTED").
            transaction_notes - string, free-text notes from teller or banker (e.g., "Customer requested expedited card delivery").
        </Existing_fields>
        
        <Field_to_be_obtained>
            transaction_approved - boolean, whether the deposit transaction was successfully approved and processed (e.g., true).
            approval_rejection_reason - string, reason for rejection if transaction was not approved; null if approved (e.g., "Failed AML screening" or null).
            new_account_balance - float, updated account balance after deposit is posted (e.g., 6234.56).
            transaction_reference_id - string, unique reference ID generated upon successful transaction completion (e.g., "REF_2025041810234567").
            cd_certificate_number - string, certificate number for term deposits; null for demand deposits (e.g., "CD_CERT_2025_987654" or null).
            calculated_interest_at_maturity - float, projected interest earnings at maturity for term deposits; null for demand deposits (e.g., 187.50 or null).
            account_status - enum, current status of the account; options: ["ACTIVE", "DORMANT", "FROZEN", "CLOSED"] (e.g., "ACTIVE").
            processing_time_seconds - integer, total time taken to process the transaction in seconds (e.g., 142).
            cash_drawer_total_after_transaction - float, total cash in teller drawer after this transaction; null for non-cash deposits (e.g., 25678.50 or null).
            fraud_detection_score - float, real-time fraud risk score; range: 0.0-1.0, where 1.0 is highest risk (e.g., 0.12).
            compliance_check_passed - boolean, whether all compliance checks (KYC, AML, sanctions) passed (e.g., true).
            final_transaction_fee - float, any transaction fees charged to customer in USD (e.g., 0.00).
            receipt_number - string, unique receipt number printed for customer (e.g., "RCPT_20250418_5012_089").
        </Field_to_be_obtained>
        </schema>
        '''
    yaml_file = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_Generation.yaml"
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
