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
    sop_title = "Banking Business Handling Process"
    additional_context = '''
        ### 1. Business Overview
        - **Scenario Name:** Retail Mortgage Loan Origination & Approval
        - **Industry:** Banking / Financial Services
        - **Primary Purpose / Value:** To facilitate the lending of funds to individual customers for property purchase or refinancing, ensuring the bank manages credit risk effectively while generating interest income and securing collateral. The ultimate goal is to reach a deterministic decision (Approve/Decline) and determine specific loan terms (Interest Rate, Loan Amount).

        ### 2. Business Types
        - **Sub-business A: Mortgage Application Intake & Pre-qualification:** The initial gathering of borrower data to determine preliminary eligibility and maximum loan capability.
        - **Sub-business B: Credit Underwriting:** The detailed analysis of the borrower’s creditworthiness, income stability, and debt-to-income ratio.
        - **Sub-business C: Property Valuation (Appraisal):** The assessment of the collateral's market value to ensure the Loan-to-Value (LTV) ratio meets bank standards.

        ### 3. Process Description

        - **Sub-business A Process: Mortgage Application Intake & Pre-qualification**
            1) **Customer Identification (KYC):** The Loan Officer verifies the identity of the applicant using government-issued ID and collects basic demographic data (Name, SSN/National ID, Address).
            2) **Income & Asset Declaration:** Applicant inputs self-declared income, employment history, and liquid assets.
            3) **Credit Pull Authorization:** Applicant signs a consent form allowing the bank to pull their credit report from bureaus.
            4) **Systematic Pre-screen:** The system runs a preliminary rules-engine check against minimum FICO score and maximum Debt-to-Income (DTI) thresholds.
            5) **Output Generation:** The system outputs a "Pre-qualification Letter" with a tentative *Maximum Loan Amount* and *Estimated Interest Rate*.

        - **Sub-business B Process: Credit Underwriting**
            1) **Document Verification:** Underwriter reviews uploaded pay stubs, W-2s/tax returns, and bank statements against declared data.
            2) **Credit Report Analysis:** Underwriter analyzes the credit report for derogatory marks (bankruptcies, late payments) and calculates the exact Debt Service Coverage Ratio (DSCR).
            3) **Risk Scoring:** Internal risk models calculate a specific Probability of Default (PD) score based on the verified data.
            4) **Policy Exception Check:** If specific metrics (e.g., DTI > 43%) are borderline, the Underwriter determines if compensating factors (e.g., high cash reserves) exist.
            5) **Output Generation:** A deterministic *Credit Score Grade* (e.g., A, B, C) and a *Final Risk-Adjusted Interest Rate* are produced.

        - **Sub-business C Process: Property Valuation (Appraisal)**
            1) **Appraisal Order:** The bank triggers an order to an Appraisal Management Company (AMC) or internal appraisal desk.
            2) **Site Inspection:** The licensed appraiser visits the property to assess condition, square footage, and upgrades.
            3) **Comparable Analysis:** The appraiser identifies at least three recent sales of similar properties in the immediate vicinity ("comps").
            4) **Value Reconciliation:** Adjustments are made to the value based on differences between the subject property and the comps.
            5) **Output Generation:** The process yields a *Final Appraised Value* (Numerical Currency Amount) and the calculated *Loan-to-Value (LTV) Ratio*.

        ### 4. Key Roles and Responsibilities
        - **Loan Officer (Relationship Manager):** Responsible for customer interaction, data collection, and advising the client on loan products.
        - **Credit Underwriter:** Responsible for the forensic analysis of financial documents, risk assessment, and making the final credit decision.
        - **Appraiser:** An independent third party responsible for providing an unbiased estimate of the property's fair market value.
        - **Loan Processor:** Administrative support who ensures the loan file is complete, compliant, and ready for the underwriter.

        ### 5. System and Tool Support
        - **Loan Origination System (LOS):** The central platform (e.g., Encompass, Ellie Mae) where the application is housed, workflows are managed, and documents are stored.
        - **Automated Underwriting System (AUS):** Software (e.g., Fannie Mae's Desktop Underwriter) that provides an immediate recommendation (Approve/Refer) based on algorithmic logic.
        - **Credit Bureau Interface:** Direct API connection to agencies like Equifax, Experian, or TransUnion to retrieve credit reports.
        - **Appraisal Management System:** Tool for ordering, tracking, and receiving property valuation reports.

        ### 6. Risks and Control Points
        - **Risk Point 1: Income Fraud / Misrepresentation**
            - *Control:* Verification of Employment (VOE) calls to employers and 4506-T tax transcript requests directly from the tax authority (IRS).
        - **Risk Point 2: Collateral Valuation Inflation**
            - *Control:* Automated Valuation Models (AVM) run as a secondary check against the manual appraisal; if the variance is >10%, a second field review is triggered.
        - **Risk Point 3: Regulatory Compliance (Fair Lending)**
            - *Control:* Automated system checks for discriminatory pricing variances; regular fair lending audits to ensure no bias in approval rates across demographic groups.

        ### 7. Typical Metrics and Measurement Methods
        - **Turnaround Time (TAT):** Measured as the number of calendar days from "Application Completed" to "Final Decision Rendered." (Target: < 25 days).
        - **Pull-Through Rate:** The percentage of applications that result in a funded loan. (Formula: Funded Loans / Total Applications Received).
        - **First Pass Yield:** The percentage of loan files submitted to underwriting that are approved without conditions or rework required.
        - **Loan-to-Value (LTV) Ratio:** A critical risk metric calculated as (Loan Amount / Appraised Property Value).

        ### 8. Best Practices or Improvement Suggestions
        - **Digital Document Ingestion:** Implement OCR (Optical Character Recognition) AI to automatically extract data from pay stubs and bank statements to reduce manual data entry errors by the Loan Processor.
        - **API Integration for Asset Verification:** Use open banking APIs (e.g., Plaid) to verify assets in real-time rather than relying on PDF bank statements, reducing the risk of document alteration.
        - **Pre-Underwriting Triage:** Introduce a "scoping" step where a senior processor reviews the file complexity before assigning it to an underwriter, ensuring experienced staff handle complex self-employed borrower files.

    '''
    yaml_file = "/Loans/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Schema_Generation.yaml"
    system_prompt = "You are an SOP dataset schema generator. Note that <Field_to_be_obtained></Field_to_be_obtained> should include several final output fields from the SOP, preferably numerical values, that can be used to determine if the process is correct."
    # 初始化客户端
    client = OpenAI(
        api_key="your_api_key",  # 替换为你的API Key
        base_url="https://..."  # API中转地址
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
