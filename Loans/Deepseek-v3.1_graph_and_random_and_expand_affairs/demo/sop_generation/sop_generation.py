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
    sample_schema = '''
        <schema>
        <Existing_fields>
            application_id - string, unique identifier for the mortgage loan application (e.g., "MLN-2025-048732").
            application_timestamp - datetime, date and time when the application was submitted (e.g., "2025-04-18T10:15:30Z").
            loan_officer_id - string, identifier for the assigned loan officer (e.g., "LO-8492").
            
            <!-- Customer Identification (KYC) -->
            applicant_name - string, full legal name of the primary borrower (e.g., "John Michael Smith").
            applicant_ssn - string, social security number or national ID (masked for privacy) (e.g., "***-**-4829").
            applicant_dob - date, date of birth of the applicant (e.g., "1985-06-15").
            applicant_address - string, current residential address (e.g., "1234 Oak Street, Springfield, IL 62704").
            co_applicant_name - string, full legal name of co-borrower if applicable (e.g., "Jane Elizabeth Smith").
            co_applicant_ssn - string, co-borrower's SSN (masked) (e.g., "***-**-7391").
            
            <!-- Income & Asset Declaration -->
            employment_status - enum, current employment status (e.g., "Full-Time", "Self-Employed", "Part-Time", "Retired").
            employer_name - string, name of current employer (e.g., "Acme Corporation").
            employment_years - float, years with current employer (e.g., 5.5).
            gross_monthly_income - float, self-declared gross monthly income in USD (e.g., 8500.00).
            additional_income_monthly - float, other income sources per month (e.g., 1200.00).
            liquid_assets - float, total declared liquid assets in USD (e.g., 75000.00).
            retirement_accounts - float, value of retirement accounts in USD (e.g., 150000.00).
            
            <!-- Loan Request Details -->
            requested_loan_amount - float, amount of loan requested by applicant in USD (e.g., 380000.00).
            loan_purpose - enum, purpose of the loan (e.g., "Purchase", "Refinance", "Cash-Out Refinance").
            loan_term_months - integer, requested loan term in months (e.g., 360, 180).
            property_address - string, address of the property being financed (e.g., "5678 Elm Avenue, Springfield, IL 62705").
            property_type - enum, type of property (e.g., "Single Family", "Condo", "Townhouse", "Multi-Family").
            property_occupancy - enum, intended occupancy (e.g., "Primary Residence", "Investment Property", "Second Home").
            estimated_property_value - float, applicant's estimated value of property in USD (e.g., 450000.00).
            down_payment_amount - float, cash down payment in USD (e.g., 90000.00).
            
            <!-- Credit Pull Authorization -->
            credit_consent_signed - boolean, whether credit pull consent was obtained (e.g., true).
            credit_consent_timestamp - datetime, when credit consent was signed (e.g., "2025-04-18T10:22:45Z").
            
            <!-- Document Verification -->
            pay_stubs_uploaded - boolean, whether pay stubs were uploaded (e.g., true).
            tax_returns_uploaded - boolean, whether tax returns were uploaded (e.g., true).
            bank_statements_uploaded - boolean, whether bank statements were uploaded (e.g., true).
            w2_forms_uploaded - boolean, whether W-2 forms were uploaded (e.g., true).
            document_verification_timestamp - datetime, when documents were verified (e.g., "2025-04-20T14:30:00Z").
            
            <!-- Credit Report Data -->
            credit_pull_timestamp - datetime, when credit report was pulled (e.g., "2025-04-18T10:25:00Z").
            fico_score_equifax - integer, FICO score from Equifax (e.g., 742).
            fico_score_experian - integer, FICO score from Experian (e.g., 738).
            fico_score_transunion - integer, FICO score from TransUnion (e.g., 745).
            derogatory_marks_count - integer, number of derogatory marks on credit report (e.g., 0).
            bankruptcy_history - boolean, whether applicant has bankruptcy history (e.g., false).
            total_monthly_debt_obligations - float, total monthly debt payments in USD (e.g., 2800.00).
            
            <!-- Property Appraisal -->
            appraisal_order_timestamp - datetime, when appraisal was ordered (e.g., "2025-04-21T09:00:00Z").
            appraiser_id - string, identifier for the assigned appraiser (e.g., "APR-3291").
            appraisal_company - string, name of appraisal management company (e.g., "National Valuation Services").
            site_inspection_timestamp - datetime, when property inspection occurred (e.g., "2025-04-23T13:00:00Z").
            property_square_footage - integer, total square footage of property (e.g., 2400).
            property_condition - enum, assessed condition of property (e.g., "Excellent", "Good", "Fair", "Poor").
            comparable_properties_count - integer, number of comparable properties analyzed (e.g., 3).
            
            <!-- Process Tracking -->
            underwriter_id - string, identifier for the assigned underwriter (e.g., "UW-6174").
            processor_id - string, identifier for the loan processor (e.g., "LP-2938").
            application_status - enum, current status of application (e.g., "Pre-qualification", "Underwriting", "Appraisal Pending", "Decision Rendered").
        </Existing_fields>
        
        <Field_to_be_obtained>
            <!-- Pre-qualification Outputs -->
            prequalification_decision - enum, preliminary eligibility decision (e.g., "Qualified", "Conditionally Qualified", "Not Qualified").
            max_loan_amount_prequalified - float, maximum loan amount from pre-qualification in USD (e.g., 400000.00).
            estimated_interest_rate_prequalification - float, tentative interest rate as percentage (e.g., 6.75).
            
            <!-- Credit Underwriting Outputs -->
            verified_gross_monthly_income - float, underwriter-verified gross monthly income in USD (e.g., 8300.00).
            debt_to_income_ratio - float, calculated DTI ratio as percentage (e.g., 33.73).
            debt_service_coverage_ratio - float, calculated DSCR (e.g., 1.45).
            probability_of_default_score - float, calculated PD score as percentage (e.g., 2.3).
            compensating_factors_present - boolean, whether compensating factors exist for policy exceptions (e.g., true).
            credit_score_grade - enum, final credit grade assigned (e.g., "A", "B", "C", "D").
            final_risk_adjusted_interest_rate - float, final interest rate as percentage (e.g., 6.50).
            
            <!-- Property Valuation Outputs -->
            final_appraised_value - float, appraiser's final property value in USD (e.g., 465000.00).
            avm_estimated_value - float, automated valuation model estimate in USD (e.g., 458000.00).
            appraisal_avm_variance_percent - float, percentage variance between appraisal and AVM (e.g., 1.5).
            second_review_triggered - boolean, whether variance triggered second review (e.g., false).
            loan_to_value_ratio - float, calculated LTV ratio as percentage (e.g., 81.72).
            
            <!-- Final Decision Outputs -->
            final_decision - enum, ultimate loan decision (e.g., "Approved", "Approved with Conditions", "Declined").
            final_decision_timestamp - datetime, when final decision was rendered (e.g., "2025-04-25T16:45:00Z").
            approved_loan_amount - float, final approved loan amount in USD (e.g., 380000.00).
            approved_interest_rate - float, final approved interest rate as percentage (e.g., 6.50).
            approved_loan_term_months - integer, final approved term in months (e.g., 360).
            conditions_count - integer, number of conditions attached to approval (e.g., 2).
            decline_reason - string, reason for decline if applicable (e.g., "DTI exceeds maximum threshold").
            
            <!-- Process Metrics -->
            turnaround_time_days - integer, days from application to final decision (e.g., 18).
            first_pass_yield - boolean, whether approved without conditions on first submission (e.g., false).
            funded_status - boolean, whether loan was ultimately funded (e.g., true).
            funding_date - date, date when loan was funded if applicable (e.g., "2025-05-10").
            fair_lending_audit_flag - boolean, whether application was flagged for fair lending review (e.g., false).
        </Field_to_be_obtained>
        </schema>
        '''
    yaml_file = "/Loans/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_Generation.yaml"
    system_prompt = "You are an SOP complete process generator. By executing the SOP you design, several definite execution results, preferably non-random numerical values, should ultimately be obtained. Also, note that all data related to each business application is already stored in the database; therefore, steps such as generating an application_id are not necessary, as this data can be read directly from the database."

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, additional_context, sample_schema)




    # 初始化客户端
    client = OpenAI(
        api_key="your_api_key",  # 请替换为你的 API Key
        base_url="https://..."  # API 中转地址
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
