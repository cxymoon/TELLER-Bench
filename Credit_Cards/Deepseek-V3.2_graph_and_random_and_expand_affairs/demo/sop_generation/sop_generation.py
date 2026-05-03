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
    sop_title = "Banking Credit Card Issuance Scenario"
    additional_context = '''
    ### 1. Business Overview
    *   **Scenario Name:** Banking Credit Card Issuance & Origination
    *   **Industry:** Financial Services / Retail Banking
    *   **Primary Purpose / Value:** To systematically evaluate applicant creditworthiness, mitigate financial risk, and issue credit facilities (cards) to qualified customers. This process balances the bank's need for asset growth and customer acquisition with the strict requirement for regulatory compliance and risk management. The ultimate value is the creation of a profitable lending relationship and a frictionless onboarding experience for the customer.

    ### 2. Business Types (Sub-businesses)
    *   **Sub-business A: Retail Credit Card Application Processing:** Handling applications from individual consumers for personal use credit cards (e.g., Rewards, Cashback, Low Interest).
    *   **Sub-business B: Small Business/Corporate Card Issuance:** Processing applications from business entities for corporate expense management or working capital.
    *   **Sub-business C: Secured Credit Card Issuance:** Processing applications for customers with limited or poor credit history, requiring a cash collateral deposit.

    ### 3. Process Description
    *Note: These processes focus strictly on the issuance phase, ending with the final decision and credit limit assignment (deterministic outputs).*

    *   **Sub-business A: Retail Credit Card Application Processing**
        1)  **Step 1: Application Intake & Data Capture:** The system captures applicant PII (Personally Identifiable Information), employment data, income details, and housing status via digital channels (web/app) or branch entry.
        2)  **Step 2: Identity Verification (KYC/CIP):** The system automatically validates the applicant's identity against government databases and internal watchlists (OFAC, AML) to prevent fraud.
        3)  **Step 3: Credit Bureau Inquiry:** The system triggers a hard inquiry to major credit bureaus (Experian, TransUnion, Equifax) to retrieve the credit score, utilization ratios, and payment history.
        4)  **Step 4: Debt-to-Income (DTI) Calculation:** The engine calculates the applicant's ability to pay by comparing declared income against debt obligations found in the credit report.
        5)  **Step 5: Automated Decisioning & Scorecard Evaluation:** The application is run through the bank’s internal credit scorecard model. It generates a final risk score (e.g., 0-1000).
        6)  **Step 6: Final Output Generation:** The system outputs a binary decision (Approve/Decline) and, if approved, calculates the specific **Credit Limit (e.g., $5,000)** and **APR (e.g., 18.99%)**.

    *   **Sub-business B: Small Business/Corporate Card Issuance**
        1)  **Step 1: Entity & Guarantor Data Collection:** Collecting legal business documents (Articles of Incorporation, TIN/EIN) and personal details of the beneficial owner/guarantor.
        2)  **Step 2: Business Verification (KYB):** Validating the existence and active status of the business entity via Secretary of State registries and commercial credit bureaus (e.g., Dun & Bradstreet).
        3)  **Step 3: Blended Credit Assessment:** Pulling both the commercial credit report of the business and the personal credit report of the guarantor.
        4)  **Step 4: Cash Flow Analysis:** Analyzing business bank statements or tax returns (if uploaded) to determine average monthly revenue and volatility.
        5)  **Step 5: Underwriting Decision:** A credit underwriter (or automated system for small limits) reviews the blended risk profile.
        6)  **Step 6: Final Output Generation:** Deterministic output of **Approved Credit Line (e.g., $50,000)** and **Cardholder Limit Distribution** (how much limit per employee card).

    *   **Sub-business C: Secured Credit Card Issuance**
        1)  **Step 1: Application & Collateral Request:** Applicant submits personal details and agrees to provide a security deposit.
        2)  **Step 2: Deposit Verification:** The system verifies the receipt of the security deposit into a holding account (locking funds).
        3)  **Step 3: Risk Hygiene Check:** Basic checks for bankruptcy or active fraud alerts (credit score is less relevant here, but "derogatory marks" are checked).
        4)  **Step 4: Ratio Configuration:** The system applies the bank's LTV (Loan-to-Value) policy (usually 100% or 90% of the deposit).
        5)  **Step 5: Final Output Generation:** Outputs the **Approved Credit Limit** strictly equal to or representing a percentage of the **Verified Deposit Amount**.

    ### 4. Key Roles and Responsibilities
    *   **Applicant (Customer):** Provides accurate personal and financial data; consents to credit checks.
    *   **Credit Underwriter:** Reviews complex or borderline applications (manual review queues) that the automated system cannot decide; validates income documentation.
    *   **Fraud Analyst:** Investigates alerts generated by the identity verification system (e.g., address mismatches, synthetic identity suspicion).
    *   **Product Manager:** Defines the scorecard criteria, APR ranges, and credit limit strategies for specific card products.
    *   **Compliance Officer:** Ensures the application process adheres to regulations like ECOA (Equal Credit Opportunity Act) and FCRA (Fair Credit Reporting Act).

    ### 5. System and Tool Support
    *   **LOS (Loan Origination System):** The central platform that manages the workflow from application ingestion to final decisioning.
    *   **Decision Engine (BRE - Business Rule Engine):** Executes logic (e.g., "If Credit Score < 600, then Decline") and calculates the internal scorecard.
    *   **External Data Interface (API Gateway):** Connects to credit bureaus (Experian/Equifax) and fraud databases (LexisNexis/ChexSystems).
    *   **Document Management System (DMS):** Stores uploaded proofs of income (pay stubs) or identity (driver's license images) securely.

    ### 6. Risks and Control Points
    *   **Risk Point 1: Identity Fraud (Synthetic IDs):**
        *   *Control Measure:* Multi-factor authentication (OTP) during application; Knowledge-Based Authentication (KBA); device fingerprinting analysis.
    *   **Risk Point 2: Inaccurate Credit Decisioning (Bad Debt):**
        *   *Control Measure:* Regular back-testing of credit scorecards against actual default rates; automated DTI threshold enforcement.
    *   **Risk Point 3: Regulatory Non-Compliance (Fair Lending violations):**
        *   *Control Measure:* Automated auditing of decline reasons to ensure no decisions are based on protected class attributes (race, gender, etc.); generation of Adverse Action Letters.
    *   **Risk Point 4: Data Privacy Leakage:**
        *   *Control Measure:* Encryption of PII at rest and in transit; strict role-based access control (RBAC) within the LOS.

    ### 7. Typical Metrics and Measurement Methods
    *   **Approval Rate:** (Total Approved Applications / Total Applications Received) × 100. Indicates the quality of the applicant pool and the strictness of credit policy.
    *   **Turnaround Time (TAT):** Average time elapsed from "Application Submit" to "Final Decision Output." Measured in seconds for auto-decisions and hours/days for manual review.
    *   **Cost per Acquisition (CPA):** Total marketing and processing costs divided by the number of funded/activated accounts.
    *   **Auto-Decision Rate:** Percentage of applications processed without any human intervention. High rates indicate efficient automation.

    ### 8. Best Practices or Improvement Suggestions
    *   **Suggestion 1: Implement Open Banking API Integration:** Instead of asking customers to upload PDF bank statements or enter income manually, allow them to connect their primary bank account securely to instantly verify income and cash flow, reducing fraud and manual review time.
    *   **Suggestion 2: Dynamic Friction:** Apply "lighter" KYC checks for low-risk applicants (trusted devices/IPs) and step-up verification (document scan/biometrics) only for high-risk signals to improve conversion rates.
    *   **Suggestion 3: Pre-Qualification (Soft Pull):** Allow customers to see if they are likely to be approved without impacting their credit score (Soft Inquiry) before the formal submission (Hard Inquiry), improving the user experience and the quality of final applications.

    '''
    sample_schema = '''
        <schema>
        <Existing_fields>
            application_id - string, unique identifier for the credit card application (e.g., "APP_CC_2025041801234").
            application_timestamp - datetime, time when the application was submitted (e.g., "2025-04-18T10:23:45Z").
            application_channel - enum[string], channel through which application was received: "web", "mobile_app", "branch", "phone" (e.g., "mobile_app").
            business_type - enum[string], type of credit card application: "retail", "small_business", "corporate", "secured" (e.g., "retail").
            product_type - string, specific card product applied for (e.g., "Rewards Platinum", "Cashback Gold", "Business Expense Card").
            applicant_id - string, anonymized unique identifier for the applicant (e.g., "CUST_789456").
            applicant_age - integer, age of the applicant in years (e.g., 34).
            applicant_employment_status - enum[string], employment status: "employed_full_time", "employed_part_time", "self_employed", "unemployed", "retired" (e.g., "employed_full_time").
            applicant_annual_income - float, declared annual income in USD (e.g., 75000.00).
            applicant_housing_status - enum[string], housing situation: "own", "rent", "mortgage", "living_with_family" (e.g., "mortgage").
            applicant_monthly_housing_cost - float, monthly housing payment in USD (e.g., 1800.00).
            applicant_years_at_address - integer, number of years at current address (e.g., 3).
            applicant_years_employed - integer, number of years with current employer (e.g., 5).
            business_entity_name - string, legal name of business entity (for business cards only, null for retail) (e.g., "Tech Solutions LLC").
            business_tin_ein - string, Tax Identification Number or Employer Identification Number (e.g., "XX-XXXXXXX").
            business_incorporation_date - date, date when business was incorporated (e.g., "2018-06-15").
            business_annual_revenue - float, declared annual business revenue in USD (e.g., 500000.00).
            business_industry_code - string, NAICS industry classification code (e.g., "541511").
            guarantor_personal_income - float, personal income of business guarantor in USD (e.g., 120000.00).
            security_deposit_amount - float, amount of security deposit for secured cards in USD (e.g., 500.00).
            deposit_verification_status - enum[string], status of deposit verification: "pending", "verified", "failed" (e.g., "verified").
            deposit_account_id - string, holding account identifier for security deposit (e.g., "DEP_ACC_445566").
            identity_verification_method - enum[string], method used for identity verification: "government_database", "document_scan", "biometric", "kba" (e.g., "government_database").
            identity_verification_timestamp - datetime, time when identity was verified (e.g., "2025-04-18T10:24:12Z").
            watchlist_check_result - enum[string], result of OFAC/AML screening: "clear", "potential_match", "confirmed_match" (e.g., "clear").
            fraud_alert_flags - list[string], list of fraud indicators detected (e.g., ["address_mismatch", "device_fingerprint_suspicious"]).
            device_fingerprint_id - string, unique identifier for the device used during application (e.g., "DEV_FP_998877").
            ip_address_location - string, geographic location derived from IP address (e.g., "New York, NY, US").
            credit_bureau_inquiry_timestamp - datetime, time when credit bureau was queried (e.g., "2025-04-18T10:24:30Z").
            credit_bureau_source - enum[string], credit bureau used: "experian", "transunion", "equifax", "dnb_commercial" (e.g., "experian").
            credit_score - integer, credit score retrieved from bureau (range: 300-850 for personal, 0-100 for commercial) (e.g., 720).
            credit_utilization_ratio - float, percentage of available credit currently used (e.g., 0.32).
            number_of_open_accounts - integer, total number of open credit accounts (e.g., 8).
            number_of_delinquent_accounts - integer, number of accounts with late payments (e.g., 0).
            total_outstanding_debt - float, total debt across all accounts in USD (e.g., 25000.00).
            bankruptcy_flag - boolean, whether applicant has bankruptcy on record (e.g., false).
            foreclosure_flag - boolean, whether applicant has foreclosure on record (e.g., false).
            months_since_last_delinquency - integer, months since most recent late payment, null if none (e.g., null).
            uploaded_documents - list[object], list of documents submitted with application (e.g., [{"doc_type": "pay_stub", "doc_id": "DOC_112233", "upload_timestamp": "2025-04-18T10:22:00Z"}]).
            manual_review_required - boolean, whether application was flagged for manual underwriter review (e.g., false).
            manual_review_reason - string, reason for manual review if applicable (e.g., "Income verification required").
            underwriter_id - string, identifier of underwriter who reviewed application, null if auto-decisioned (e.g., null).
            processing_start_timestamp - datetime, time when automated decisioning began (e.g., "2025-04-18T10:24:35Z").
            processing_end_timestamp - datetime, time when final decision was generated (e.g., "2025-04-18T10:25:02Z").
        </Existing_fields>
        
        <Field_to_be_obtained>
            debt_to_income_ratio - float, calculated DTI ratio (total debt / annual income) (e.g., 0.33).
            internal_risk_score - integer, proprietary credit scorecard output (range: 0-1000) (e.g., 785).
            final_decision - enum[string], final application decision: "approved", "declined", "pending_manual_review" (e.g., "approved").
            decline_reason_code - string, standardized reason code if declined (e.g., "INSUFFICIENT_INCOME", "LOW_CREDIT_SCORE", null if approved).
            adverse_action_letter_generated - boolean, whether adverse action notice was created (e.g., false).
            approved_credit_limit - float, approved credit line amount in USD, null if declined (e.g., 8000.00).
            assigned_apr - float, Annual Percentage Rate assigned to the account (e.g., 18.99).
            credit_limit_to_deposit_ratio - float, ratio of credit limit to security deposit for secured cards (e.g., 1.00).
            cardholder_limit_distribution - object, for business cards, distribution of limits per employee card (e.g., {"total_line": 50000.00, "per_card_limit": 5000.00, "max_cards": 10}).
            turnaround_time_seconds - integer, total processing time from submission to decision in seconds (e.g., 27).
            auto_decision_flag - boolean, whether decision was made automatically without human intervention (e.g., true).
            account_activation_eligible - boolean, whether approved application is eligible for immediate card activation (e.g., true).
            estimated_card_delivery_date - date, projected date for physical card delivery (e.g., "2025-04-25").
        </Field_to_be_obtained>
        </schema>
        '''
    yaml_file = "/Credit_cards/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_Generation.yaml"
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
