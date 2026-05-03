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
    yaml_file = "/Invest/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Schema_Generation.yaml"
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
