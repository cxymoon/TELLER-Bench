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
    sop_title = "Banking Invest Process"
    additional_context = '''
    ### 1. Business Overview
    - **Scenario Name:** Retail Wealth Management & Investment Product Subscription
    - **Industry:** Banking & Financial Services
    - **Primary Purpose / Value:** To facilitate the purchase and allocation of investment products (such as mutual funds, bonds, and structured deposits) for retail and high-net-worth clients, ensuring regulatory compliance (KYC/AML) while maximizing assets under management (AUM) and fee-based revenue for the bank.

    ### 2. Business Types
    - **Sub-business A: Risk Profiling & Investor Suitability Assessment:** The foundational process of determining a client's risk tolerance, financial situation, and investment knowledge before they can purchase any product.
    - **Sub-business B: Mutual Fund Subscription (Lump Sum):** The direct purchase of open-ended mutual fund units by a client using available funds in their savings account.
    - **Sub-business C: Structured Product Enrollment:** The sales and setup process for complex financial instruments (e.g., Dual Currency Investments or Principal Protected Notes) requiring specific disclosures and cooling-off periods.

    ### 3. Process Description

    - **Sub-business A Process: Risk Profiling & Investor Suitability Assessment**
        1)  **Step 1: Client Identification:** The Relationship Manager (RM) verifies the client's identity using government-issued ID and retrieves the client's current profile from the CRM.
        2)  **Step 2: Questionnaire Administration:** The client completes a regulatory-mandated "Risk Tolerance Questionnaire" (digital or paper-based) covering age, income, investment experience, and liquidity needs.
        3)  **Step 3: Scoring & Categorization:** The system calculates a total score based on the answers and maps the client to a specific Risk Profile (e.g., Conservative, Balanced, Aggressive).
        4)  **Step 4: Acknowledgment & Confirmation:** The client reviews the calculated risk level. If they agree, they digitally sign the assessment. If they disagree, a re-evaluation process is triggered with documented justification.
        5)  **Step 5: Validity Setting:** The system sets an expiration date for the assessment (typically 1-2 years) and updates the client's "Investable Status" flag to "Active."

    - **Sub-business B Process: Mutual Fund Subscription (Lump Sum)**
        1)  **Step 1: Product Selection:** Based on the client's Risk Profile, the RM recommends a specific mutual fund (e.g., "Global Tech Growth Fund"). The system verifies that the product's risk rating does not exceed the client's risk tolerance.
        2)  **Step 2: Fact Sheet & Disclosure Presentation:** The RM presents the Fund Fact Sheet, Prospectus, and Key Information Document (KID) to the client, highlighting fees, past performance, and risks.
        3)  **Step 3: Order Entry:** The RM enters the subscription amount (e.g., $50,000), the fund code, and the fee structure (front-end load discount, if applicable) into the Investment Order System.
        4)  **Step 4: Pre-Trade Compliance Check:** The system performs real-time checks for funds availability, AML sanctions screening, and suitability matching.
        5)  **Step 5: Client Authorization:** The client authorizes the transaction via a secure method (OTP, biometric signature, or physical signature on a deal ticket).
        6)  **Step 6: Transaction Execution:** The system debits the client's settlement account and generates a "Trade Reference Number" and "Estimated Units" based on the last known NAV (Net Asset Value).

    - **Sub-business C Process: Structured Product Enrollment**
        1)  **Step 1: Product Pitch & Scenario Analysis:** The RM explains the payoff scenarios of a structured note (e.g., "If Stock X stays above $100, you get 5% coupon; if it drops, capital is at risk").
        2)  **Step 2: Audio/Video Recording (VR):** For complex products, regulations often require the sales process to be recorded. The RM initiates the VR system, clearly stating the product risks and obtaining client verbal consent.
        3)  **Step 3: Cooling-Off Period Setup:** The RM inputs the order. Unlike mutual funds, the system places the order in a "Pending Cooling-Off" status (e.g., 24 hours), during which the client can cancel without penalty.
        4)  **Step 4: Final Confirmation:** After the cooling-off period expires, the system automatically checks for cancellation requests. If none, the order status changes to "Ready for Execution."
        5)  **Step 5: Order Batched for Treasury:** The system aggregates the order with others and transmits a buy order to the Bank's Treasury/Global Markets desk.
        6)  **Step 6: Booking Confirmation:** The Treasury desk confirms the strike price and value date. The system generates a "Contract Note Number" and "Notional Amount Booked."

    ### 4. Key Roles and Responsibilities
    - **Relationship Manager (RM):** Responsible for client consultation, product recommendation, conducting risk assessments, and ensuring the "Suitability" of products sold.
    - **Client:** Responsible for providing accurate financial information, reviewing disclosure documents, and authorizing transactions.
    - **Investment Product Specialist (IPS):** Subject matter expert who supports the RM in explaining complex products and assists in portfolio construction.
    - **Compliance Officer:** Reviews high-risk transactions, monitors audio/video recordings for miss-selling, and ensures adherence to regulatory guidelines.
    - **Operations/Back Office:** Handles the reconciliation of trades, settlement of funds, and communication with fund houses or custodians.

    ### 5. System and Tool Support
    - **Wealth Management System (WMS):** The core platform for managing client portfolios, executing trades, and storing product data.
    - **CRM (Customer Relationship Management):** Stores client KYC data, interaction logs, and triggers for risk profile renewals.
    - **Sales Compliance Recording System (SCRS):** A specialized tool for recording, encrypting, and archiving audio/video interactions during the sales process.
    - **Risk Engine:** An automated middleware that compares Client Risk Scores against Product Risk Ratings to prevent non-compliant sales (Hard blocking).
    - **Market Data Terminal (e.g., Bloomberg/Reuters):** Used by RMs and IPSs to view real-time pricing, NAVs, and market news.

    ### 6. Risks and Control Points
    - **Risk Point 1: Miss-selling (Suitability Risk)**
        - *Description:* Selling a high-risk product to a conservative investor.
        - *Control Measures:* Automated system hard-blocks preventing order entry if (Product Risk > Client Risk) unless a specific "Mismatch Form" is signed and approved by a supervisor.
    - **Risk Point 2: Money Laundering (AML Risk)**
        - *Description:* Large lump sum investments used to layer illicit funds.
        - *Control Measures:* Real-time transaction monitoring triggers for amounts exceeding a threshold (e.g., $100,000) requiring Source of Wealth (SOW) documentation upload before trade execution.
    - **Risk Point 3: Operational Failure (Trade Error)**
        - *Description:* Entering the wrong fund code or amount.
        - *Control Measures:* "Four-eye principle" (Maker-Checker) requirement for transactions above a certain value; mandatory read-back of order details to the client before final confirmation.
    - **Risk Point 4: Failure to Disclose Risks**
        - *Description:* Client not fully understanding the capital-at-risk nature of a product.
        - *Control Measures:* Mandatory "Double Recording" or Audio/Video recording where the RM must read a specific risk script and the client must verbally acknowledge understanding.

    ### 7. Typical Metrics and Measurement Methods
    - **Sales Conversion Rate:** (Number of successful subscription orders / Number of product pitches made) × 100%.
    - **Straight-Through Processing (STP) Rate:** Percentage of orders that pass from the RM input to the Fund House/Treasury without manual intervention or repair.
    - **Suitability Pass Rate:** Percentage of trades executed without requiring a "Risk Mismatch" waiver.
    - **Average Ticket Size:** Total subscription volume / Total number of transactions.
    - **Compliance Defect Ratio:** Number of transactions flagged for process errors (e.g., missing documents, unclear recording) / Total transactions reviewed.

    ### 8. Best Practices or Improvement Suggestions
    - **Suggestion 1: Digital Self-Service Integration:** Enable clients to complete Risk Profiling and simple Mutual Fund purchases via Mobile Banking app to reduce branch traffic and administrative load on RMs.
    - **Suggestion 2: Real-time Portfolio Simulation:** Integrate tools into the WMS that allow RMs to show clients "Before vs. After" scenarios of how a new investment affects their portfolio's risk/return characteristics during the sales pitch.
    - **Suggestion 3: Automated Fee Disclosure:** Automatically generate a simplified "Fee Impact Sheet" for every transaction that clearly calculates the exact dollar amount of fees (entry load + annual management fee) to improve transparency and trust.

    '''
    sample_schema = '''
        <schema>
        <Existing_fields>
            transaction_id - string, unique identifier for the investment transaction (e.g., "INV_2025041801234").
            client_id - string, unique anonymized client identifier (e.g., "CLI_987654").
            client_name - string, client's full name (e.g., "John Michael Smith").
            client_age - integer, client's age in years (e.g., 45).
            client_annual_income - float, client's declared annual income in USD (e.g., 150000.00).
            client_net_worth - float, client's total net worth in USD (e.g., 850000.00).
            client_investment_experience_years - integer, years of investment experience (e.g., 8).
            client_liquidity_needs - enum, client's liquidity requirement level: ["Low", "Medium", "High"] (e.g., "Low").
            relationship_manager_id - string, unique identifier for the RM handling the transaction (e.g., "RM_4521").
            relationship_manager_name - string, name of the relationship manager (e.g., "Sarah Johnson").
            transaction_timestamp - datetime, timestamp when the transaction was initiated (e.g., "2025-04-18T10:15:30Z").
            transaction_type - enum, type of investment process: ["Risk_Profiling", "Mutual_Fund_Subscription", "Structured_Product_Enrollment"] (e.g., "Mutual_Fund_Subscription").
            product_code - string, unique code for the investment product (e.g., "MF_TECH_001").
            product_name - string, name of the investment product (e.g., "Global Tech Growth Fund").
            product_risk_rating - enum, risk level of the product: ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"] (e.g., "Growth").
            product_category - enum, category of investment product: ["Mutual_Fund", "Bond", "Structured_Note", "Dual_Currency_Investment", "Principal_Protected_Note"] (e.g., "Mutual_Fund").
            subscription_amount - float, amount client wishes to invest in USD (e.g., 50000.00).
            front_end_load_percentage - float, front-end load fee percentage (e.g., 2.5).
            annual_management_fee_percentage - float, annual management fee percentage (e.g., 1.75).
            current_nav - float, current Net Asset Value per unit (e.g., 15.42).
            settlement_account_number - string, client's account number for fund settlement (e.g., "ACC_1234567890").
            available_balance - float, available balance in settlement account in USD (e.g., 75000.00).
            risk_questionnaire_responses - JSON, structured responses to risk tolerance questionnaire (e.g., {"q1_age_bracket": "41-50", "q2_income_stability": "Stable", "q3_investment_horizon": "5-10_years", "q4_loss_tolerance": "Moderate", "q5_previous_investments": "Stocks_Bonds"}).
            risk_questionnaire_total_score - integer, raw score from risk questionnaire (e.g., 68).
            previous_risk_profile - enum, client's previous risk profile if exists: ["None", "Conservative", "Balanced", "Aggressive"] (e.g., "Balanced").
            previous_risk_profile_expiry_date - datetime, expiration date of previous risk assessment (e.g., "2025-03-15T00:00:00Z").
            disclosure_documents_presented - list[string], list of documents shown to client (e.g., ["Fund_Fact_Sheet", "Prospectus", "KID"]).
            aml_sanctions_screening_status - enum, initial AML screening status: ["Pending", "Clear", "Flagged"] (e.g., "Clear").
            source_of_wealth_documented - boolean, whether source of wealth documentation is on file (e.g., True).
            transaction_threshold_exceeded - boolean, whether transaction exceeds monitoring threshold (e.g., False).
            monitoring_threshold_amount - float, AML monitoring threshold in USD (e.g., 100000.00).
            audio_video_recording_required - boolean, whether VR recording is mandatory for this product (e.g., False).
            audio_video_recording_id - string, unique identifier for the recording if applicable (e.g., "VR_2025041801234" or null).
            cooling_off_period_required - boolean, whether cooling-off period applies (e.g., False).
            cooling_off_period_hours - integer, duration of cooling-off period in hours (e.g., 0).
            client_authorization_method - enum, method used for client authorization: ["OTP", "Biometric", "Physical_Signature", "Digital_Signature"] (e.g., "OTP").
            client_authorization_timestamp - datetime, timestamp of client authorization (e.g., "2025-04-18T10:45:22Z").
            mismatch_waiver_required - boolean, whether risk mismatch waiver is needed (e.g., False).
            mismatch_waiver_document_id - string, identifier for mismatch waiver document if applicable (e.g., null).
            supervisor_approval_required - boolean, whether supervisor approval is needed (e.g., False).
            supervisor_id - string, identifier of approving supervisor if applicable (e.g., null).
            compliance_officer_id - string, identifier of compliance officer reviewing transaction (e.g., "CO_7821").
            market_data_source - enum, source of market data used: ["Bloomberg", "Reuters", "Internal_System"] (e.g., "Bloomberg").
            branch_code - string, code of the branch where transaction originated (e.g., "BR_NYC_001").
            channel - enum, channel through which transaction was initiated: ["Branch", "Phone_Banking", "Mobile_App", "Online_Banking"] (e.g., "Branch").
            region - string, geographic region of the transaction (e.g., "US_Northeast").
        </Existing_fields>
        
        <Field_to_be_obtained>
            calculated_risk_profile - enum, system-calculated risk profile based on questionnaire: ["Conservative", "Balanced", "Aggressive"] (e.g., "Balanced").
            risk_profile_expiry_date - datetime, expiration date for the new risk assessment (e.g., "2027-04-18T00:00:00Z").
            investable_status - enum, client's investment eligibility status: ["Active", "Inactive", "Suspended"] (e.g., "Active").
            suitability_check_passed - boolean, whether product suitability check passed (e.g., True).
            suitability_mismatch_flag - boolean, whether there is a risk mismatch between client and product (e.g., False).
            pre_trade_compliance_passed - boolean, whether all pre-trade compliance checks passed (e.g., True).
            funds_availability_verified - boolean, whether sufficient funds are available (e.g., True).
            estimated_units - float, estimated number of units to be purchased based on NAV (e.g., 3240.86).
            front_end_load_amount - float, calculated front-end load fee in USD (e.g., 1250.00).
            net_investment_amount - float, amount actually invested after fees in USD (e.g., 48750.00).
            annual_management_fee_amount - float, estimated annual management fee in USD (e.g., 875.00).
            trade_reference_number - string, unique reference number for the executed trade (e.g., "TRN_2025041812345").
            contract_note_number - string, contract note number for structured products (e.g., null or "CN_2025041856789").
            transaction_status - enum, final status of the transaction: ["Pending", "Pending_Cooling_Off", "Ready_for_Execution", "Executed", "Cancelled", "Failed"] (e.g., "Executed").
            execution_timestamp - datetime, timestamp when transaction was executed (e.g., "2025-04-18T11:30:45Z").
            settlement_date - datetime, expected settlement date for the transaction (e.g., "2025-04-20T00:00:00Z").
            straight_through_processing_flag - boolean, whether transaction processed without manual intervention (e.g., True).
            compliance_defect_flag - boolean, whether any compliance defects were identified (e.g., False).
            compliance_review_notes - string, notes from compliance officer review (e.g., "All documentation complete. No issues identified.").
            supervisor_approval_timestamp - datetime, timestamp of supervisor approval if required (e.g., null).
            total_processing_time_minutes - integer, total time from initiation to execution in minutes (e.g., 75).
        </Field_to_be_obtained>
        </schema>
        '''
    yaml_file = "/Invest/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_Generation.yaml"
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
