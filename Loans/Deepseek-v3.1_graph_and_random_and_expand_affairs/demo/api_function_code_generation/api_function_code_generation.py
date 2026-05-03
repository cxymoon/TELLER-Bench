import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI

def load_yaml_prompt(file_path, api, sample_data):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text"]
    text = text.replace("{{api}}", api)
    text = text.replace("{{sample_data}}", sample_data)
    return text

def main():
    # === 输入参数 ===
    api = '''
        <sop_tool_analysis>

        ## 1. SOP Section References

        This API tool collection implements the comprehensive mortgage origination workflow defined in:
        - **Section 5.1**: Mortgage Application Intake and Pre-qualification Phase (substeps 5.1.1-5.1.7)
        - **Section 5.2**: Credit Underwriting Phase (substeps 5.2.1-5.2.9)
        - **Section 5.3**: Property Valuation and Appraisal Phase (substeps 5.3.1-5.3.6)
        - **Section 5.4**: Final Decision Rendering and Approval Conditions (substeps 5.4.1-5.4.7)

        The 22 API tools form a sequential dependency chain matching the SOP's procedural flow.

        ## 2. Input Parameters Analysis

        Each API tool requires specific inputs that align with the SOP's data requirements:

        **First-Tier APIs (no dependencies)**:
        - ValidateApplicantIdentity: applicant identity data per Section 5.1.1 (KYC/OFAC verification)
        - ValidateEmploymentAndIncome: employment/income declarations per Section 5.1.2
        - ValidateLiquidAssets: asset declarations per Section 5.1.3
        - PullCreditReports: credit authorization per Section 5.1.4

        **Sequential Dependencies**:
        - CalculateRepresentativeCreditScore depends on PullCreditReports (tri-merge methodology per Section 5.1.5)
        - ExecutePrequalificationAnalysis depends on credit score, income, and debt data (Section 5.1.6-5.1.7)
        - VerifyEmploymentAndIncome requires prequalification_decision as state gate (Section 5.2.1-5.2.2)
        - Subsequent underwriting APIs build on verified data through Section 5.2
        - Appraisal APIs implement Section 5.3 workflow
        - Final decision APIs synthesize all findings per Section 5.4

        ## 3. Constraints and Special Considerations

        **Regulatory Compliance Constraints**:
        - OFAC clearance mandatory before proceeding (USA PATRIOT Act Section 326)
        - Credit consent signed and timestamped required (FCRA permissible purpose)
        - FICO Score Tri-Merge Middle Score Methodology per GSE requirements (Section 3 definition)
        - DTI ≤43% for QM compliance (12 CFR 1026.43), ≤50% absolute maximum
        - LTV thresholds stratified by credit score tier
        - Fair lending monitoring per ECOA/Regulation B

        **Data Precision Requirements**:
        - Currency values as floats (USD)
        - Percentages to two decimal places
        - Interest rates to three decimal places
        - Dates in ISO 8601 format
        - Timestamps in UTC with timezone

        **State-Gating Logic**:
        - Cannot order appraisal for "Not Qualified" applications
        - Cannot verify employment until prequalification complete
        - Cannot render final decision until all underwriting steps complete

        ## 4. SOP Alignment

        The API collection implements the SOP's three interdependent sub-processes:

        **Sub-business A: Application Intake & Pre-qualification**
        - APIs 1-6 implement Section 5.1 (identity, income, assets, credit pull, score calculation, pre-qual analysis)

        **Sub-business B: Credit Underwriting**
        - APIs 7-12 implement Section 5.2 (verification, debt ratios, PD scoring, pricing, compensating factors)

        **Sub-business C: Property Valuation**
        - APIs 13-16 implement Section 5.3 (appraisal order, valuation, AVM validation, LTV calculation)

        **Decision & Compliance**
        - APIs 17-22 implement Section 5.4 (final decision, fair lending, performance tracking)

        ## 5. Edge Cases and Limitations

        **Credit Reporting Edge Cases**:
        - Insufficient credit file from one or more bureaus (handle with only 2 or 1 score)
        - Credit freeze preventing report pull
        - Co-applicant scenarios requiring dual score analysis then lower selection

        **Income Verification Edge Cases**:
        - Self-employed year-over-year income decline >10% (flags but doesn't auto-decline)
        - Multiple income sources requiring separate verification
        - Negative income from Schedule C losses

        **Asset Verification Edge Cases**:
        - Large deposits requiring sourcing (>50% monthly income)
        - Retirement account discount varies by age (25% vs 10%)
        - Gift funds requiring donor verification

        **Appraisal Edge Cases**:
        - AVM variance >±10% triggering second review
        - Property condition C4/C5 requiring repairs or decline
        - No comparable sales available within 12 months/1 mile

        **Compensating Factors Edge Cases**:
        - DTI 43.01-50% requiring minimum 5-7 compensating factor points
        - Points accumulation from multiple sources (reserves, equity, income stability)

        ## 6. Workflow Integration

        The 22 APIs execute in this dependency sequence:

        **Phase 1: Initial Validation (parallel execution possible)**
        1. ValidateApplicantIdentity
        2. ValidateEmploymentAndIncome
        3. ValidateLiquidAssets

        **Phase 2: Credit Analysis (sequential)**
        4. PullCreditReports (depends on #1)
        5. CalculateRepresentativeCreditScore (depends on #4)

        **Phase 3: Pre-qualification (sequential)**
        6. ExecutePrequalificationAnalysis (depends on #2, #3, #5)

        **Phase 4: Detailed Verification (sequential, gated by Phase 3)**
        7. VerifyEmploymentAndIncome (depends on #6)
        8. VerifyAssetsAndReserves (depends on #6, #7)
        9. AnalyzeCreditHistory (depends on #4)

        **Phase 5: Underwriting Calculations (sequential)**
        10. CalculateDebtRatios (depends on #7, #4, #6)
        11. CalculateProbabilityOfDefault (depends on #5, #10, #8, #9)
        12. DetermineFinalInterestRate (depends on #11, #10)
        13. EvaluateCompensatingFactors (depends on #10, #8, #9)

        **Phase 6: Appraisal (sequential, can run parallel to Phase 5)**
        14. OrderPropertyAppraisal (depends on #6)
        15. ConductPropertyValuation (depends on #14)
        16. ValidateAppraisalWithAVM (depends on #15)
        17. CalculateLoanToValueRatio (depends on #16)

        **Phase 7: Final Decision (sequential, requires all previous phases)**
        18. RenderFinalLoanDecision (depends on #10, #11, #12, #13, #15, #17, #9)
        19. ExecuteFairLendingMonitoring (depends on #18)
        20. TrackLoanPerformanceMetrics (depends on #18)

        This workflow ensures deterministic, auditable processing with clear dependencies matching the SOP's logical progression from application intake through final decision rendering.

        </sop_tool_analysis>

        <SCHEMA>
        {
        "toolSpec": {
            "name": "ValidateApplicantIdentity",
            "description": "Performs comprehensive Know Your Customer (KYC) verification by validating government-issued identification documents, checking applicant and co-applicant information against OFAC Specially Designated Nationals lists and other government watchlists, validating Social Security Number patterns and authenticity through SSA Consent Based Verification Service when necessary, and ensuring identity documents match application data fields to prevent identity fraud and comply with USA PATRIOT Act Section 326 Customer Identification Program requirements.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application in the format MLN-YYYY-NNNNNN, used to track the application through all processing stages and establish audit trail linkage."
                },
                "applicant_name": {
                    "type": "string",
                    "description": "Full legal name of the primary applicant as it appears on government-issued identification documents, used for OFAC screening and identity verification matching. Format: First Middle Last."
                },
                "applicant_ssn": {
                    "type": "string",
                    "description": "Social Security Number of the primary applicant in masked format ***-**-NNNN for privacy protection in transit, used for SSN pattern validation, OFAC screening, and SSA Consent Based Verification Service authentication when necessary."
                },
                "applicant_dob": {
                    "type": "string",
                    "description": "Date of birth of the primary applicant in ISO 8601 format YYYY-MM-DD, used for age verification, identity matching against government documents, and OFAC screening accuracy."
                },
                "applicant_address": {
                    "type": "string",
                    "description": "Current residential address of the primary applicant including street number, street name, city, state, and ZIP code, used for identity verification consistency checking and address validation against identification documents."
                },
                "co_applicant_name": {
                    "type": "string",
                    "description": "Full legal name of the co-applicant if applicable, formatted as First Middle Last. Optional field that when populated triggers dual-applicant identity verification and OFAC screening procedures."
                },
                "co_applicant_ssn": {
                    "type": "string",
                    "description": "Social Security Number of the co-applicant in masked format ***-**-NNNN if co-applicant exists. Optional field required when co_applicant_name is provided for complete identity verification of all parties to the loan."
                }
                },
                "required": [
                "application_id",
                "applicant_name",
                "applicant_ssn",
                "applicant_dob",
                "applicant_address"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ValidateEmploymentAndIncome",
            "description": "Analyzes uploaded income documentation including pay stubs, W-2 forms, and tax returns to calculate preliminary gross monthly income, validates employment status and employer information against document data, performs income averaging for self-employed applicants using Schedule C or K-1 distributions with add-backs for non-cash deductions, flags year-over-year income declines exceeding 10% for underwriter attention, and establishes initial income figures for pre-qualification debt-to-income calculations.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this income validation to the specific loan file for audit trail and workflow tracking purposes."
                },
                "employment_status": {
                    "type": "string",
                    "enum": ["Full-Time", "Self-Employed", "Part-Time", "Retired"],
                    "description": "Classification of the applicant's employment arrangement that determines the documentation requirements and income verification methodology. Full-Time requires pay stubs and W-2s, Self-Employed requires tax returns with Schedule C/E/K-1 analysis, Part-Time requires wage verification, Retired requires pension/Social Security documentation."
                },
                "employer_name": {
                    "type": "string",
                    "description": "Name of the employer organization or self-employed business entity as stated by the applicant. Optional for Retired status. Used to verify consistency between application data and uploaded pay stubs or tax documents and to facilitate Verification of Employment contact."
                },
                "employment_years": {
                    "type": "number",
                    "description": "Number of years the applicant has been employed with the current employer or in the same line of work for self-employed applicants, expressed as a decimal (e.g., 5.5 for five years and six months). Optional for Retired status. Used to assess income stability and employment continuity as compensating factors."
                },
                "gross_monthly_income": {
                    "type": "number",
                    "description": "Self-declared gross monthly income in United States Dollars before any tax withholdings or voluntary deductions, representing base salary or wages. This preliminary figure is subject to verification against pay stubs, W-2 forms, or tax returns during the validation process."
                },
                "additional_income_monthly": {
                    "type": "number",
                    "description": "Self-declared additional monthly income from sources other than base employment including bonus compensation, commission income, rental property income, investment dividends, alimony or child support received, or other recurring income sources. Optional field that when populated requires verification and continuance probability assessment."
                },
                "pay_stubs_uploaded": {
                    "type": "boolean",
                    "description": "Boolean indicator confirming whether pay stubs covering the most recent 30 days of pay history have been uploaded to the Loan Origination System. Required to be true for Full-Time and Part-Time employment status to enable income calculation from pay frequency and gross pay per period."
                },
                "w2_forms_uploaded": {
                    "type": "boolean",
                    "description": "Boolean indicator confirming whether IRS Form W-2 Wage and Tax Statements for the most recent two calendar years have been uploaded. Required to be true for Full-Time and Part-Time employment status to validate year-over-year income consistency and enable income averaging."
                },
                "tax_returns_uploaded": {
                    "type": "boolean",
                    "description": "Boolean indicator confirming whether complete IRS Form 1040 individual tax returns with all schedules for the two most recent calendar years have been uploaded. Optional for Full-Time status but required to be true for Self-Employed status to enable Schedule C/E/K-1 analysis and income averaging with non-cash deduction add-backs."
                }
                },
                "required": [
                "application_id",
                "employment_status",
                "gross_monthly_income",
                "pay_stubs_uploaded",
                "w2_forms_uploaded"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ValidateLiquidAssets",
            "description": "Reviews uploaded bank statements covering two consecutive months to verify declared liquid assets and retirement accounts, identifies and flags large deposits exceeding 50% of monthly income or cumulative deposits exceeding 100% of monthly income requiring source documentation, calculates available reserves after accounting for down payment and estimated closing costs, determines reserve months by dividing remaining liquid assets by proposed total housing payment, and applies appropriate discount factors to retirement accounts based on applicant age to reflect tax and penalty implications.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this asset validation to the specific loan file for complete financial profile assembly and audit trail maintenance."
                },
                "liquid_assets": {
                    "type": "number",
                    "description": "Self-declared total dollar amount of liquid assets including checking accounts, savings accounts, money market accounts, and brokerage accounts readily convertible to cash without penalty. Expressed in United States Dollars. Subject to verification against uploaded bank statements showing ending balances for two consecutive months."
                },
                "retirement_accounts": {
                    "type": "number",
                    "description": "Self-declared total dollar amount of retirement savings including 401(k) plans, Individual Retirement Accounts (IRA), SEP-IRAs, and other tax-deferred retirement vehicles. Optional field that when populated is considered for reserve compensating factors but subject to discount factors of 25% for applicants under age 59.5 or 10% for age 59.5 and above to reflect tax liability and potential early withdrawal penalties."
                },
                "bank_statements_uploaded": {
                    "type": "boolean",
                    "description": "Boolean indicator confirming whether bank statements covering two consecutive months for all accounts cited in the liquid_assets declaration have been uploaded to the Loan Origination System. Required to be true to enable verification of ending balances, identification of large deposits, and assessment of asset stability."
                },
                "down_payment_amount": {
                    "type": "number",
                    "description": "Dollar amount of the applicant's equity contribution toward the property purchase, expressed in United States Dollars. Used to calculate remaining liquid assets after down payment is subtracted from verified assets to determine available reserves for ongoing housing payment capacity and financial contingencies."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested by the applicant, expressed in United States Dollars. Used in conjunction with down_payment_amount to estimate closing costs (typically 2-4% of loan amount) which must be deducted from liquid assets when calculating reserve months."
                },
                "applicant_dob": {
                    "type": "string",
                    "description": "Date of birth of the primary applicant in ISO 8601 format YYYY-MM-DD. Required to calculate applicant age for determining appropriate retirement account discount factor: 25% discount if age less than 59.5 years reflecting 10% early withdrawal penalty plus estimated tax liability, or 10% discount if age 59.5 or above reflecting only tax liability without penalty."
                }
                },
                "required": [
                "application_id",
                "liquid_assets",
                "bank_statements_uploaded",
                "down_payment_amount",
                "requested_loan_amount",
                "applicant_dob"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "PullCreditReports",
            "description": "Executes tri-bureau credit report acquisition by transmitting applicant and co-applicant verified identity information to institution's credit bureau API interfaces connected to Equifax, Experian, and TransUnion repositories, retrieves in-file credit reports with FICO scores calculated using mortgage-specific scoring models (FICO Score 2, 5, and 4 respectively), extracts and populates credit score fields for each bureau, analyzes credit report narrative to count derogatory marks including collections, charge-offs, public records, late payments, identifies bankruptcy history and foreclosure history, and calculates total monthly debt obligations by summing all recurring payment obligations shown on tradelines.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking the credit report pull to the specific loan file and establishing the credit_pull_timestamp for regulatory compliance timing calculations under Truth in Lending Act disclosure requirements."
                },
                "verified_applicant_name": {
                    "type": "string",
                    "description": "Identity-verified full legal name of the primary applicant resulting from ValidateApplicantIdentity API execution, ensuring OFAC clearance was obtained prior to credit bureau transmission. Format: First Middle Last matching government identification documents."
                },
                "verified_applicant_ssn": {
                    "type": "string",
                    "description": "Identity-verified Social Security Number of the primary applicant in masked format ***-**-NNNN, confirmed through ValidateApplicantIdentity API to have passed SSN pattern validation and not matched OFAC lists. Used as primary identifier for credit bureau file matching."
                },
                "verified_applicant_dob": {
                    "type": "string",
                    "description": "Identity-verified date of birth of the primary applicant in ISO 8601 format YYYY-MM-DD, confirmed through ValidateApplicantIdentity API. Used in combination with name and SSN for accurate credit file identification and to prevent mismatched credit report retrieval."
                },
                "applicant_address": {
                    "type": "string",
                    "description": "Current residential address of the primary applicant including street number, street name, city, state, and ZIP code. Used by credit bureaus for additional file matching accuracy and to retrieve location-specific credit history when multiple files exist for similar identifying information."
                },
                "co_applicant_name": {
                    "type": "string",
                    "description": "Identity-verified full legal name of the co-applicant if applicable, formatted as First Middle Last. Optional field that when populated triggers parallel tri-bureau credit report acquisition for the co-applicant requiring separate credit consent and identity verification."
                },
                "co_applicant_ssn": {
                    "type": "string",
                    "description": "Identity-verified Social Security Number of the co-applicant in masked format ***-**-NNNN if co-applicant exists. Optional field required when co_applicant_name is provided to retrieve separate credit reports for applying FICO Score Tri-Merge Middle Score Methodology to each individual before selecting the lower qualifying score."
                },
                "credit_consent_signed": {
                    "type": "boolean",
                    "description": "Boolean indicator confirming that the applicant (and co-applicant if applicable) has executed the consumer disclosure and consent form satisfying Fair Credit Reporting Act permissible purpose requirements under 15 USC 1681b. Must be true to proceed with credit report acquisition; false value returns error preventing unauthorized credit pulls."
                },
                "credit_consent_timestamp": {
                    "type": "string",
                    "description": "Date and time when the credit consent was electronically or manually signed by the applicant in ISO 8601 format with UTC timezone (YYYY-MM-DDTHH:MM:SSZ). Used to establish that consent was obtained before credit report transmission and for audit trail documentation of FCRA compliance."
                }
                },
                "required": [
                "application_id",
                "verified_applicant_name",
                "verified_applicant_ssn",
                "verified_applicant_dob",
                "applicant_address",
                "credit_consent_signed",
                "credit_consent_timestamp"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CalculateRepresentativeCreditScore",
            "description": "Applies the FICO Score Tri-Merge Middle Score Methodology by ranking the three bureau credit scores (fico_score_equifax, fico_score_experian, fico_score_transunion) in ascending order and selecting the median value as the representative score, handling cases where only two scores are available by selecting the lower value, or using a single score if only one bureau provides data, and for co-applicant scenarios, determining each individual's representative score then selecting the lower of the two as the qualifying credit score used for all subsequent underwriting policy evaluation, pricing decisions, and mortgage insurance premium calculations.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this credit score calculation to the specific loan file for consistent credit policy application and pricing determination throughout the underwriting workflow."
                },
                "fico_score_equifax": {
                    "type": "integer",
                    "description": "FICO Score obtained from Equifax credit repository using the FICO Score 5 (Equifax Beacon 5.0) mortgage-specific scoring model, ranging from 300 to 850. Optional field that may be null if Equifax returned insufficient credit file; when null, tri-merge methodology proceeds with available scores from other bureaus."
                },
                "fico_score_experian": {
                    "type": "integer",
                    "description": "FICO Score obtained from Experian credit repository using the FICO Score 2 (Experian/Fair Isaac Risk Model v2) mortgage-specific scoring model, ranging from 300 to 850. Optional field that may be null if Experian returned insufficient credit file; when null, tri-merge methodology proceeds with available scores from other bureaus."
                },
                "fico_score_transunion": {
                    "type": "integer",
                    "description": "FICO Score obtained from TransUnion credit repository using the FICO Score 4 (TransUnion FICO Risk Score, Classic 04) mortgage-specific scoring model, ranging from 300 to 850. Optional field that may be null if TransUnion returned insufficient credit file; when null, tri-merge methodology proceeds with available scores from other bureaus."
                },
                "co_applicant_fico_equifax": {
                    "type": "integer",
                    "description": "FICO Score from Equifax for the co-applicant if a co-applicant exists on the loan application. Optional field used in dual-applicant scenarios where methodology first calculates each individual's representative middle score, then selects the lower of the two representative scores as the qualifying score for underwriting and pricing."
                },
                "co_applicant_fico_experian": {
                    "type": "integer",
                    "description": "FICO Score from Experian for the co-applicant if a co-applicant exists on the loan application. Optional field used in dual-applicant scenarios where methodology first calculates each individual's representative middle score, then selects the lower of the two representative scores as the qualifying score for underwriting and pricing."
                },
                "co_applicant_fico_transunion": {
                    "type": "integer",
                    "description": "FICO Score from TransUnion for the co-applicant if a co-applicant exists on the loan application. Optional field used in dual-applicant scenarios where methodology first calculates each individual's representative middle score, then selects the lower of the two representative scores as the qualifying score for underwriting and pricing."
                }
                },
                "required": [
                "application_id"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ExecutePrequalificationAnalysis",
            "description": "Performs systematic pre-qualification screening by comparing representative credit score against institutional minimum threshold of 620 for conventional products, calculates preliminary debt-to-income ratio using estimated housing payment derived from requested loan amount and proxy interest rate based on credit score tier plus total monthly debt obligations divided by gross monthly income, evaluates calculated DTI against policy thresholds to generate prequalification_decision classification of Qualified (DTI ≤43%), Conditionally Qualified (DTI 43.01-50%), or Not Qualified (DTI >50% or FICO <620), calculates maximum loan capacity by solving DTI equation for loan amount that achieves 43% ratio, determines estimated interest rate by applying credit score band and property type pricing adjustments to current market base rates.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this pre-qualification analysis to the loan file and establishing the prequalification_decision state gate that controls progression to detailed verification and underwriting phases."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "The qualifying FICO score calculated using the Tri-Merge Middle Score Methodology from CalculateRepresentativeCreditScore API, representing the median of three bureau scores for single applicants or the lower of two representative scores for co-applicant scenarios. Used to determine minimum credit acceptability (≥620) and credit score tier for proxy interest rate estimation."
                },
                "preliminary_gross_monthly_income": {
                    "type": "number",
                    "description": "Preliminary validated gross monthly income in United States Dollars from ValidateEmploymentAndIncome API, representing base salary or wages before tax withholdings. Used as denominator in debt-to-income ratio calculation and to calculate maximum loan capacity at target DTI threshold."
                },
                "total_monthly_debt_obligations": {
                    "type": "number",
                    "description": "Sum of all recurring monthly debt payment obligations in United States Dollars from PullCreditReports API, including installment loans, revolving credit minimum payments, student loans, alimony, child support, and other contractual obligations. Used as component of DTI numerator along with estimated housing payment."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested by the applicant in United States Dollars. Used to calculate estimated monthly principal and interest payment for preliminary DTI calculation, and compared against calculated maximum loan capacity to assess affordability feasibility."
                },
                "loan_term_months": {
                    "type": "integer",
                    "description": "Number of months over which the loan will amortize, typically 360 for 30-year fixed mortgages or 180 for 15-year fixed mortgages. Used in mortgage payment formula P=L[c(1+c)^n]/[(1+c)^n-1] to calculate estimated monthly principal and interest payment for DTI calculation."
                },
                "estimated_property_value": {
                    "type": "number",
                    "description": "Applicant's good-faith estimate of the subject property's market value in United States Dollars, derived from purchase contract price for purchase transactions or applicant estimate for refinance transactions. Used to calculate preliminary loan-to-value ratio and estimate property taxes (1.2% annually national average) and hazard insurance (0.5% annually) for housing payment calculation."
                },
                "down_payment_amount": {
                    "type": "number",
                    "description": "Dollar amount of the applicant's equity contribution in United States Dollars. Used in conjunction with estimated_property_value to calculate preliminary loan-to-value ratio for determining mortgage insurance requirement and premium estimation in housing payment calculation."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Classification of the subject property's structural type. Single Family = detached single-unit dwelling, Condo = condominium unit in multi-unit project, Townhouse = attached single-unit dwelling sharing walls, Multi-Family = 2-4 unit residential property. Used to apply property type pricing adjustments to estimated interest rate (+0.250% Condo, +0.375% Townhouse, +0.500% to +0.750% Multi-Family)."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Intended occupancy status of the subject property. Primary Residence = borrower will occupy as principal dwelling, Investment Property = property will be rented to tenants, Second Home = vacation or occasional use property. Used to apply occupancy pricing adjustments to estimated interest rate (+0.375% Second Home, +0.750% to +2.000% Investment Property depending on LTV and credit)."
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["Purchase", "Refinance", "Cash-Out Refinance"],
                    "description": "Transaction type classification. Purchase = financing property acquisition, Refinance = replacing existing mortgage with new loan at current rates without cash extraction, Cash-Out Refinance = replacing existing mortgage with larger loan and extracting equity as cash. Used to apply loan purpose pricing adjustments to estimated interest rate (+0.125% Refinance, +0.375% to +0.750% Cash-Out Refinance)."
                }
                },
                "required": [
                "application_id",
                "representative_credit_score",
                "preliminary_gross_monthly_income",
                "total_monthly_debt_obligations",
                "requested_loan_amount",
                "loan_term_months",
                "estimated_property_value",
                "down_payment_amount",
                "property_type",
                "property_occupancy",
                "loan_purpose"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "VerifyEmploymentAndIncome",
            "description": "Conducts comprehensive income verification through Verification of Employment (VOE) telephone calls or written requests to employers using independently verified contact information to confirm current employment status, job title, hire date, base salary or hourly wage, and employment continuity expectations for Full-Time employed applicants, performs detailed tax return analysis for Self-Employed applicants by examining IRS Form 1040 with Schedule C, E, K-1 distributions over two years to calculate adjusted income with add-backs for depreciation and depletion while subtracting one-time income events, applies two-year income averaging methodology, flags year-over-year income declines exceeding 10% requiring documented justification, verifies bonus and commission income through employer confirmation of continuation likelihood.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this detailed income verification to the loan file and establishing verified_gross_monthly_income as the definitive income figure used in all subsequent underwriting calculations including debt-to-income ratio and debt service coverage ratio."
                },
                "prequalification_decision": {
                    "type": "string",
                    "enum": ["Qualified", "Conditionally Qualified", "Not Qualified"],
                    "description": "Pre-qualification classification outcome from ExecutePrequalificationAnalysis API serving as state gate to control workflow progression. Only applications with Qualified or Conditionally Qualified status proceed to detailed verification; Not Qualified applications return error preventing unnecessary verification work for ineligible applications."
                },
                "employment_status": {
                    "type": "string",
                    "enum": ["Full-Time", "Self-Employed", "Part-Time", "Retired"],
                    "description": "Classification of the applicant's employment arrangement determining verification methodology. Full-Time triggers VOE contact to employer and pay stub analysis, Self-Employed triggers comprehensive tax return analysis with Schedule C/E/K-1 review and income add-backs, Part-Time requires wage verification, Retired requires pension/Social Security documentation review."
                },
                "employer_name": {
                    "type": "string",
                    "description": "Name of the employer organization or self-employed business entity. Optional for Retired status. For Full-Time/Part-Time, used to conduct Verification of Employment by independently verifying employer contact information through directory services and calling to confirm employment details. For Self-Employed, matched against business name shown on tax returns."
                },
                "employment_years": {
                    "type": "number",
                    "description": "Number of years employed with current employer or in same line of work for self-employed, expressed as decimal (e.g., 5.5). Optional for Retired status. Used to assess income stability: <2.0 years triggers additional scrutiny, 5.0+ years contributes to compensating factors, verified through VOE confirmation of hire date or tax return history demonstrating business continuity."
                },
                "gross_monthly_income": {
                    "type": "number",
                    "description": "Self-declared gross monthly income in United States Dollars from initial application. Used as comparison baseline for verification; if verified_gross_monthly_income differs by more than 20%, triggers income discrepancy error requiring explanation or application correction."
                },
                "additional_income_monthly": {
                    "type": "number",
                    "description": "Self-declared additional monthly income from bonus, commission, overtime, rental, investment, alimony, or other sources in United States Dollars. Optional field that when populated requires separate verification: bonus/commission through employer confirmation of continuation likelihood and two-year tax return averaging, rental income through lease agreements and Schedule E review, investment income through brokerage statements."
                },
                "pay_stubs_uploaded": {
                    "type": "boolean",
                    "description": "Boolean confirming pay stubs covering most recent 30 days uploaded. For Full-Time/Part-Time, used to calculate verified income by analyzing pay frequency (weekly, bi-weekly, semi-monthly, monthly), gross pay per period, and year-to-date earnings, extracting employer name to match against employer_name field for consistency verification."
                },
                "w2_forms_uploaded": {
                    "type": "boolean",
                    "description": "Boolean confirming IRS Form W-2 for most recent two years uploaded. For Full-Time/Part-Time, used to validate year-over-year income consistency, calculate two-year average if income varies, identify income trending (increasing/stable/declining), and flag year-over-year declines exceeding 10% for underwriter attention."
                },
                "tax_returns_uploaded": {
                    "type": "boolean",
                    "description": "Boolean confirming complete IRS Form 1040 with all schedules for two years uploaded. Required for Self-Employed status. For self-employed income verification, extracts Adjusted Gross Income from Line 11, analyzes Schedule C (sole proprietor), Schedule E (rental/partnership), or K-1 (S-corp/partnership), adds back depreciation and depletion non-cash deductions, subtracts one-time income events, calculates two-year average divided by 12 for monthly figure."
                }
                },
                "required": [
                "application_id",
                "prequalification_decision",
                "employment_status",
                "gross_monthly_income",
                "pay_stubs_uploaded",
                "w2_forms_uploaded"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "VerifyAssetsAndReserves",
            "description": "Executes comprehensive asset verification by examining two consecutive months of bank statements for all checking, savings, and money market accounts to confirm ending balances match or exceed declared liquid_assets, investigates all individual deposits exceeding 50% of monthly income or cumulative deposits exceeding 100% of monthly income to identify potential undisclosed borrowed funds requiring source documentation such as asset transfer statements or gift letters with donor verification, applies appropriate discount factors to retirement_accounts (25% for under age 59.5, 10% for age 59.5 and above) to reflect tax and penalty implications, calculates net available assets after subtracting down_payment_amount and estimated closing costs, determines reserve months by dividing remaining assets by proposed total housing payment.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this comprehensive asset verification to the loan file and establishing verified_liquid_assets_final and reserve_months_calculated as definitive metrics used in compensating factors analysis and underwriting decision logic."
                },
                "prequalification_decision": {
                    "type": "string",
                    "enum": ["Qualified", "Conditionally Qualified", "Not Qualified"],
                    "description": "Pre-qualification classification outcome from ExecutePrequalificationAnalysis API serving as state gate. Only Qualified or Conditionally Qualified applications proceed to detailed asset verification; Not Qualified applications return error preventing unnecessary verification work for applications already determined ineligible."
                },
                "liquid_assets": {
                    "type": "number",
                    "description": "Self-declared total dollar amount of liquid assets in United States Dollars from initial application including checking, savings, money market accounts, and brokerage accounts. Used as comparison baseline against bank statement ending balances; significant variance triggers asset discrepancy investigation or potential fraud alert."
                },
                "retirement_accounts": {
                    "type": "number",
                    "description": "Self-declared total dollar amount of retirement savings in United States Dollars including 401(k), IRA, SEP-IRA, and other tax-deferred vehicles. Optional field that when populated receives discount factor based on applicant age: 25% discount if under 59.5 (reflecting 10% early withdrawal penalty plus ~15% tax), 10% discount if 59.5+ (reflecting only tax liability), used for reserve compensating factors."
                },
                "bank_statements_uploaded": {
                    "type": "boolean",
                    "description": "Boolean confirming bank statements covering two consecutive months for all cited accounts uploaded. Required for asset verification. Used to confirm ending balances aggregate to declared amount, identify large deposits (>50% monthly income individually or >100% cumulatively) requiring source documentation, detect negative balances or non-sufficient funds activity indicating financial instability."
                },
                "down_payment_amount": {
                    "type": "number",
                    "description": "Dollar amount of applicant's equity contribution in United States Dollars. Subtracted from verified liquid assets to calculate remaining assets available as reserves. If verified assets less than down payment plus closing costs, triggers insufficient funds error requiring applicant to increase assets or reduce loan amount/purchase price."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested in United States Dollars. Used to estimate closing costs (2-4% of loan amount including lender fees, third-party fees, prepaid items, recording fees) which are subtracted from verified assets along with down payment to determine net available reserves for ongoing financial capacity assessment."
                },
                "applicant_dob": {
                    "type": "string",
                    "description": "Date of birth of primary applicant in ISO 8601 format YYYY-MM-DD. Used to calculate current age for determining appropriate retirement account discount factor. If age calculation yields less than 59.5 years, applies 25% discount to retirement_accounts; if 59.5 years or older, applies 10% discount reflecting absence of early withdrawal penalty."
                },
                "verified_gross_monthly_income": {
                    "type": "number",
                    "description": "Verified gross monthly income in United States Dollars from VerifyEmploymentAndIncome API representing definitive income figure confirmed through VOE or tax return analysis. Used to establish thresholds for identifying large deposits: individual deposits exceeding 50% of this amount or cumulative deposits exceeding 100% require sourcing documentation to prevent undisclosed borrowed funds."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Classification of subject property structural type. Combined with property_occupancy to determine minimum reserve requirements per institutional policy: Primary Residence Single Family requires 2 months reserves if LTV≤80%, Investment Property requires 12 months regardless of LTV, used to assess reserve adequacy and calculate reserve_compensating_factor_points."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Intended occupancy status of subject property. Combined with property_type and LTV to determine minimum reserve policy thresholds: Primary Residence 2-4 months depending on LTV, Second Home 6 months, Investment Property 12 months. Used to evaluate reserve adequacy for risk mitigation and calculate compensating factor points if reserves exceed minimums."
                }
                },
                "required": [
                "application_id",
                "prequalification_decision",
                "liquid_assets",
                "bank_statements_uploaded",
                "down_payment_amount",
                "requested_loan_amount",
                "applicant_dob",
                "verified_gross_monthly_income",
                "property_type",
                "property_occupancy"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "AnalyzeCreditHistory",
            "description": "Conducts forensic analysis of tri-bureau credit report narrative sections by examining each tradeline's 24-month payment history to identify and count derogatory marks including 30-day late payments (1 point), 60-day late payments (2 points), 90+ day late payments (3 points), collections accounts with distinction between medical collections under $500 (may be disregarded per GSE guidelines) and non-medical collections requiring payment or acceptance justification, charge-offs, public records including tax liens and civil judgments, reviews foreclosure and short sale history to verify compliance with institutional seasoning requirements, validates bankruptcy_history and ensures sufficient time has elapsed since discharge, analyzes credit inquiry patterns in preceding 90 days to identify potential undisclosed debt.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this detailed credit history analysis to the loan file and establishing comprehensive credit quality metrics used in underwriting decision logic, compensating factors evaluation, and fair lending monitoring."
                },
                "credit_report_full_data": {
                    "type": "object",
                    "description": "Complete structured credit report data object from PullCreditReports API containing tradeline payment histories, public records section, collections section, inquiries section, and account details for all three bureaus. Analyzed to extract detailed_derogatory_marks_count by examining each tradeline's 24-month payment history for late payments, reviewing public records for liens/judgments, counting collections with medical vs non-medical distinction, identifying charge-offs, and analyzing inquiry frequency."
                },
                "derogatory_marks_count": {
                    "type": "integer",
                    "description": "Initial derogatory marks count from PullCreditReports API representing preliminary count of adverse items. Used as baseline for detailed forensic analysis that produces refined detailed_derogatory_marks_count with weighted scoring: 30-day late = 1 point, 60-day late = 2 points, 90+ day late = 3 points, collections/charge-offs = varies, providing more granular credit quality assessment."
                },
                "bankruptcy_history": {
                    "type": "boolean",
                    "description": "Boolean indicator from PullCreditReports API confirming presence or absence of bankruptcy filing in credit history. When true, triggers seasoning compliance check: Chapter 7 discharge must be 4+ years ago, Chapter 13 discharge must be 2+ years ago with satisfactory payment history on re-established credit, or 3 years for extenuating circumstances with documentation. Sets bankruptcy_seasoning_compliant Boolean in response."
                }
                },
                "required": [
                "application_id",
                "credit_report_full_data",
                "derogatory_marks_count",
                "bankruptcy_history"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CalculateDebtRatios",
            "description": "Computes precise debt-to-income ratio and debt service coverage ratio using verified financial data by first calculating exact monthly housing payment using mortgage payment formula P=L[c(1+c)^n]/[(1+c)^n-1] where P is principal and interest, L is approved loan amount, c is monthly interest rate, n is loan term in months, then adding property taxes obtained from tax assessor records, hazard insurance quoted by carrier, mortgage insurance premium calculated from MI rate tables based on LTV and credit score if LTV exceeds 80%, and HOA dues from association documents, summing this total housing payment with total_monthly_debt_obligations from credit report, dividing by verified_gross_monthly_income and multiplying by 100 to produce debt_to_income_ratio, and calculating debt_service_coverage_ratio as verified_gross_monthly_income divided by sum of housing payment plus debt obligations.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this debt ratio calculation to the loan file and establishing debt_to_income_ratio and debt_service_coverage_ratio as critical underwriting metrics that determine policy compliance status and gate final approval decision."
                },
                "verified_gross_monthly_income": {
                    "type": "number",
                    "description": "Verified gross monthly income in United States Dollars from VerifyEmploymentAndIncome API, confirmed through VOE for wage earners or tax return analysis for self-employed. Used as denominator in both DTI and DSCR calculations, establishing the income base against which all debt obligations are measured for capacity assessment."
                },
                "total_monthly_debt_obligations": {
                    "type": "number",
                    "description": "Sum of all recurring monthly debt payments in United States Dollars from PullCreditReports API including installment loans (>10 months remaining), revolving credit (minimum payment or 1% of balance, whichever greater), student loans (documented payment or 0.5% of balance if deferred), alimony, child support. Combined with housing payment for DTI numerator and DSCR denominator."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested in United States Dollars. Used as L in mortgage payment formula P=L[c(1+c)^n]/[(1+c)^n-1] to calculate monthly principal and interest payment component of total housing payment. May be adjusted downward during underwriting if calculated DTI exceeds policy thresholds."
                },
                "loan_term_months": {
                    "type": "integer",
                    "description": "Number of months over which loan will amortize, typically 360 (30-year) or 180 (15-year). Used as n in mortgage payment formula to calculate monthly principal and interest. Shorter terms increase monthly payment but reduce total interest paid, potentially impacting DTI compliance."
                },
                "estimated_interest_rate_prequalification": {
                    "type": "number",
                    "description": "Preliminary interest rate as annual percentage to three decimal places from ExecutePrequalificationAnalysis API. Converted to monthly rate c by dividing by 12 and used in payment formula. Will be refined to final_risk_adjusted_interest_rate but provides basis for initial housing payment calculation."
                },
                "property_address": {
                    "type": "string",
                    "description": "Complete street address, city, state, ZIP code of subject property. Used to retrieve actual property tax amount from county tax assessor records for precise monthly tax calculation, replacing estimated 1.2% national average with property-specific data for accurate housing payment and DTI calculation."
                },
                "estimated_property_value": {
                    "type": "number",
                    "description": "Property value in United States Dollars, from purchase contract price for purchases or appraisal for refinances. Used to estimate hazard insurance premium (typically 0.5% annually divided by 12) until actual insurance quote obtained, and combined with loan amount to determine if mortgage insurance required for LTV>80% and calculate MI premium."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "Qualifying FICO score from CalculateRepresentativeCreditScore API. Used in conjunction with LTV to determine mortgage insurance premium rate from MI rate tables: rates range from 0.30% annually (85% LTV, 760+ FICO) to 1.50% annually (95% LTV, 680 FICO), impacting monthly housing payment and therefore DTI calculation."
                },
                "preliminary_ltv_ratio": {
                    "type": "number",
                    "description": "Preliminary loan-to-value ratio as percentage to two decimal places from ExecutePrequalificationAnalysis API. Determines mortgage insurance requirement: LTV≤80% no MI required, LTV>80% requires MI with premium based on specific LTV percentage and credit score tier, significantly impacting monthly housing payment and DTI."
                }
                },
                "required": [
                "application_id",
                "verified_gross_monthly_income",
                "total_monthly_debt_obligations",
                "requested_loan_amount",
                "loan_term_months",
                "estimated_interest_rate_prequalification",
                "property_address",
                "estimated_property_value",
                "representative_credit_score",
                "preliminary_ltv_ratio"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CalculateProbabilityOfDefault",
            "description": "Executes institution's proprietary multivariate logistic regression model to quantify default risk by computing probability_of_default_score as PD = 1/(1+e^-z) where z = β₀ + β₁(FICO) + β₂(DTI) + β₃(LTV) + β₄(Reserves) + β₅(Employment_Years) + β₆(Property_Type_Indicator) + β₇(Occupancy_Indicator) + β₈(Loan_Purpose_Indicator) + β₉(Derogatory_Marks) with institution-calibrated beta coefficients derived from historical loan performance data, maps the PD score to credit_score_grade using thresholds of ≤2.0% for Grade A, 2.01-4.0% for Grade B, 4.01-7.0% for Grade C, and >7.0% for Grade D requiring special review.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this probability of default calculation to the loan file and establishing credit_score_grade as primary risk classification used for pricing determination, capital allocation, and decisioning logic throughout underwriting workflow."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "Qualifying FICO score from CalculateRepresentativeCreditScore API, ranging 300-850. Used as β₁(FICO) term in logistic regression where coefficient is typically negative (e.g., -0.012) indicating higher scores reduce default probability. Primary driver of PD score with approximately 35-40% weight in model."
                },
                "debt_to_income_ratio": {
                    "type": "number",
                    "description": "Calculated DTI as percentage to two decimal places from CalculateDebtRatios API. Used as β₂(DTI) term in logistic regression where coefficient is positive (e.g., +0.08) indicating higher DTI increases default probability. Measures payment burden relative to income, contributing approximately 20-25% weight to PD score."
                },
                "preliminary_ltv_ratio": {
                    "type": "number",
                    "description": "Loan-to-value ratio as percentage to two decimal places. Used as β₃(LTV) term in logistic regression where coefficient is positive (e.g., +0.05) indicating higher LTV increases default probability due to reduced equity cushion. Contributes approximately 15-20% weight to PD score, critical for loss severity in default scenarios."
                },
                "reserve_months_calculated": {
                    "type": "number",
                    "description": "Number of months of total housing payment covered by liquid asset reserves after down payment and closing costs, from VerifyAssetsAndReserves API. Used as β₄(Reserves) term where coefficient is negative (e.g., -0.15) indicating more reserves reduce default probability. Measures financial resilience to income disruption."
                },
                "employment_years": {
                    "type": "number",
                    "description": "Years employed with current employer or in same field, as decimal (e.g., 5.5). Used as β₅(Employment_Years) term where coefficient is negative (e.g., -0.05) indicating longer employment tenure reduces default probability through income stability demonstration. Contributes approximately 5-10% weight."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification. Converted to dummy variable for β₆(Property_Type_Indicator) term where Multi-Family receives coefficient (e.g., +0.3) and others default to 0, reflecting elevated default risk and marketability challenges for multi-unit properties in distressed scenarios."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Property occupancy status. Converted to dummy variable for β₇(Occupancy_Indicator) term where Investment Property receives coefficient (e.g., +0.8) and others default to 0, reflecting significantly higher default probability for non-owner-occupied properties where borrower prioritizes primary residence payment over investment property during financial stress."
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["Purchase", "Refinance", "Cash-Out Refinance"],
                    "description": "Transaction type classification. Converted to dummy variable for β₈(Loan_Purpose_Indicator) term where Cash-Out Refinance receives coefficient (e.g., +0.4) and others default to 0, reflecting elevated default risk from equity extraction reducing borrower's financial cushion and potentially funding consumption rather than wealth building."
                },
                "detailed_derogatory_marks_count": {
                    "type": "integer",
                    "description": "Weighted count of adverse credit items from AnalyzeCreditHistory API where 30-day late=1 point, 60-day late=2 points, 90+ day late=3 points, collections/charge-offs vary. Used as β₉(Derogatory_Marks) term where coefficient is positive (e.g., +0.2) per point, indicating credit blemishes increase default probability through demonstrated payment management challenges."
                }
                },
                "required": [
                "application_id",
                "representative_credit_score",
                "debt_to_income_ratio",
                "preliminary_ltv_ratio",
                "reserve_months_calculated",
                "employment_years",
                "property_type",
                "property_occupancy",
                "loan_purpose",
                "detailed_derogatory_marks_count"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "DetermineFinalInterestRate",
            "description": "Applies multi-dimensional risk-based pricing matrix to calculate final_risk_adjusted_interest_rate by retrieving base interest rate indexed to current 30-year Freddie Mac Primary Mortgage Market Survey rate or institution's cost of funds benchmark, then applying additive pricing adjustments across multiple risk dimensions including credit score adjustments ranging from -0.500% for FICO ≥780 to +2.000% for FICO 620-639, LTV ratio adjustments ranging from +0.000% for LTV ≤60% to +1.000% for LTV 90.01-95%, property type adjustments, occupancy adjustments, loan purpose adjustments, and DTI adjustments, summing all applicable adjustments algebraically to produce final interest rate expressed to three decimal places.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this final interest rate determination to the loan file and establishing final_risk_adjusted_interest_rate as definitive pricing used for loan documents, APR calculation, and payment schedule generation."
                },
                "credit_score_grade": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "Risk classification from CalculateProbabilityOfDefault API mapped from PD score: A=≤2.0% PD, B=2.01-4.0%, C=4.01-7.0%, D=>7.0%. Used to determine credit score tier for pricing adjustments, with Grade D typically requiring decline or special approval before pricing is applied."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "Qualifying FICO score from CalculateRepresentativeCreditScore API, ranging 300-850. Used to apply credit score pricing adjustments: -0.500% for ≥780, -0.250% for 760-779, 0.000% for 740-759 (base), +0.125% for 720-739, +0.250% for 700-719, +0.500% for 680-699, +0.875% for 660-679, +1.375% for 640-659, +2.000% for 620-639."
                },
                "preliminary_ltv_ratio": {
                    "type": "number",
                    "description": "Loan-to-value ratio as percentage to two decimal places. Used to apply LTV pricing adjustments: +0.000% for ≤60%, +0.125% for 60.01-70%, +0.250% for 70.01-80%, +0.375% for 80.01-85%, +0.625% for 85.01-90%, +1.000% for 90.01-95%, reflecting elevated collateral risk and loss severity at higher LTV levels."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification. Used to apply property type pricing adjustments: +0.000% for Single Family (base), +0.250% for Condo, +0.375% for Townhouse, +0.500% for two-unit Multi-Family, +1.000% for three-to-four unit Multi-Family, reflecting marketability and default risk variations."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Property occupancy status. Used to apply occupancy pricing adjustments: +0.000% for Primary Residence (base), +0.375% for Second Home, +0.750% for Investment Property single-family, +1.500% for Investment Property multi-family, reflecting elevated default probability for non-owner-occupied properties."
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["Purchase", "Refinance", "Cash-Out Refinance"],
                    "description": "Transaction type classification. Used to apply loan purpose pricing adjustments: +0.000% for Purchase (base), +0.125% for Rate-Term Refinance, +0.375% for Cash-Out Refinance with LTV≤70%, +0.750% for Cash-Out Refinance with LTV>70%, reflecting risk of equity extraction and potential consumption spending."
                },
                "debt_to_income_ratio": {
                    "type": "number",
                    "description": "Calculated DTI as percentage to two decimal places from CalculateDebtRatios API. Used to apply DTI pricing adjustments: +0.000% for ≤36%, +0.125% for 36.01-40%, +0.250% for 40.01-43%, +0.500% for 43.01-45% (if permitted by compensating factors), reflecting payment burden impact on default probability."
                },
                "loan_term_months": {
                    "type": "integer",
                    "description": "Number of months over which loan will amortize, typically 360 (30-year) or 180 (15-year). Used to retrieve appropriate pricing matrix as 15-year fixed mortgages typically price 0.250-0.500% lower than 30-year due to reduced interest rate risk and faster principal amortization benefiting lender."
                }
                },
                "required": [
                "application_id",
                "credit_score_grade",
                "representative_credit_score",
                "preliminary_ltv_ratio",
                "property_type",
                "property_occupancy",
                "loan_purpose",
                "debt_to_income_ratio",
                "loan_term_months"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "EvaluateCompensatingFactors",
            "description": "Conducts systematic compensating factors analysis for applications with elevated risk indicators by evaluating and scoring positive offsetting characteristics including reserve months (3 points for >12 months, 2 points for 6-12 months, 1 point for 3-6 months), housing payment history assessment (2 points for 12+ months of equal or higher payment without 30-day lates), residual income analysis (2 points for residual exceeding 120% of regional HUD median income after all obligations), LTV position evaluation (2 points for LTV ≤70% representing substantial equity), employment stability assessment (1 point for employment_years >5.0 in same field), and credit depth review (1 point for 3+ revolving accounts with limits >$5,000 and utilization <30%).",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this compensating factors evaluation to the loan file and establishing compensating_factors_present Boolean and total_compensating_factor_points as critical metrics for exception approval eligibility when DTI exceeds 43% or other marginal risk indicators exist."
                },
                "debt_to_income_ratio": {
                    "type": "number",
                    "description": "Calculated DTI as percentage to two decimal places from CalculateDebtRatios API. Used to determine if compensating factors analysis is required: DTI≤43% does not require analysis, DTI 43.01-45% requires 5+ points, DTI 45.01-50% requires 7+ points. Triggers mandatory compensating factor evaluation for exception approval consideration."
                },
                "reserve_months_calculated": {
                    "type": "number",
                    "description": "Months of housing payment covered by liquid assets after down payment and closing costs from VerifyAssetsAndReserves API. Scored as: 3 points if >12 months demonstrating strong financial cushion, 2 points if 6-12 months showing adequate reserves, 1 point if 3-6 months indicating minimal reserves, 0 points if <3 months."
                },
                "payment_history_quality": {
                    "type": "string",
                    "enum": ["Excellent", "Good", "Fair", "Poor"],
                    "description": "Credit payment history assessment from AnalyzeCreditHistory API evaluating tradeline payment patterns over 24 months. Used to award housing payment history compensating factor: 2 points if Excellent/Good with no 30-day lates in 12 months and current housing payment equal to or greater than proposed payment, 0 points otherwise."
                },
                "verified_gross_monthly_income": {
                    "type": "number",
                    "description": "Verified gross monthly income in United States Dollars from VerifyEmploymentAndIncome API. Used as component in residual income calculation: income minus all debt obligations including proposed housing payment, compared to 120% of regional HUD median income adjusted for family size to award 2 compensating factor points if exceeds threshold."
                },
                "monthly_housing_payment_total": {
                    "type": "number",
                    "description": "Total monthly housing payment in United States Dollars from CalculateDebtRatios API including principal, interest, property taxes, insurance, HOA, MI. Used in residual income calculation as deduction from gross income, and used as divisor for reserve months calculation to standardize reserve adequacy across varying payment levels."
                },
                "total_monthly_debt_obligations": {
                    "type": "number",
                    "description": "Sum of all recurring monthly debt payments in United States Dollars from PullCreditReports API. Used in residual income calculation: verified_gross_monthly_income minus monthly_housing_payment_total minus total_monthly_debt_obligations equals residual, which must exceed 120% of HUD median to earn 2 compensating factor points."
                },
                "preliminary_ltv_ratio": {
                    "type": "number",
                    "description": "Loan-to-value ratio as percentage to two decimal places. Used to award LTV position compensating factor: 2 points if LTV≤70% demonstrating substantial equity cushion (30%+ down payment), providing significant loss mitigation in default scenario and indicating borrower's strong financial position and commitment, 0 points if LTV>70%."
                },
                "employment_years": {
                    "type": "number",
                    "description": "Years employed with current employer or in same line of work as decimal. Used to award employment stability compensating factor: 1 point if employment_years>5.0 demonstrating long-term job stability and income continuity likelihood even if not with same employer, provided work is in consistent field, 0 points if ≤5.0 years."
                },
                "credit_depth_rating": {
                    "type": "string",
                    "enum": ["Deep", "Moderate", "Thin", "Insufficient"],
                    "description": "Credit depth assessment from AnalyzeCreditHistory API evaluating breadth and management of credit accounts. Used to award credit depth compensating factor: 1 point if Deep rating with 3+ active revolving accounts, each with credit limit >$5,000 and utilization ratio <30%, indicating sophisticated credit management, 0 points otherwise."
                },
                "applicant_address": {
                    "type": "string",
                    "description": "Current residential address of primary applicant including city, state, ZIP code. Used to geocode to HUD metropolitan statistical area or county to retrieve regional median income data for family size from HUD published tables, establishing 120% threshold for residual income compensating factor evaluation."
                }
                },
                "required": [
                "application_id",
                "debt_to_income_ratio",
                "reserve_months_calculated",
                "payment_history_quality",
                "verified_gross_monthly_income",
                "monthly_housing_payment_total",
                "total_monthly_debt_obligations",
                "preliminary_ltv_ratio",
                "employment_years",
                "credit_depth_rating",
                "applicant_address"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "OrderPropertyAppraisal",
            "description": "Initiates property valuation process by transmitting appraisal order to independent Appraisal Management Company or internal appraisal desk with comprehensive property information including property_address, property_type classification, property_occupancy status, estimated_property_value from purchase contract or applicant estimate, specific appraisal form requirements (URAR Form 1004 for single-family, Form 1073 for condos, Form 1025 for 2-4 units), recording appraisal_order_timestamp to initiate 10-business-day completion expectation per service level agreement.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this appraisal order to the loan file and establishing appraisal_order_timestamp that starts the 10-business-day completion expectation and gates the application status transition from Pre-qualification to Appraisal Pending in the loan origination workflow."
                },
                "prequalification_decision": {
                    "type": "string",
                    "enum": ["Qualified", "Conditionally Qualified", "Not Qualified"],
                    "description": "Pre-qualification classification from ExecutePrequalificationAnalysis API serving as state gate. Only Qualified or Conditionally Qualified applications proceed to appraisal order; Not Qualified applications return error preventing unnecessary appraisal expense for ineligible applications, protecting both institution cost and applicant expense."
                },
                "property_address": {
                    "type": "string",
                    "description": "Complete street address, city, state, ZIP code of subject property requiring valuation. Used by Appraisal Management Company for appraiser assignment based on geographic competency (minimum 5 appraisals in subject ZIP or adjacent ZIPs in preceding 12 months), for comparable sales search radius determination, and for property record retrieval from county assessor database."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification determining appraisal form requirement and appraiser certification level. Single Family/Townhouse requires URAR Form 1004, Condo requires Form 1073 with project review, Multi-Family (2-4 units) requires Form 1025 with income approach. Properties >$400K or complex characteristics require certified appraiser versus licensed appraiser."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Intended occupancy status of subject property. Provided to appraiser for report narrative and to ensure appraiser understands property use context for highest and best use analysis. Investment Property designation triggers income approach requirement for 2-4 unit properties to evaluate rental income potential supporting debt service capacity."
                },
                "estimated_property_value": {
                    "type": "number",
                    "description": "Purchase contract price for purchase transactions or applicant estimate for refinance transactions, in United States Dollars. Provided to appraiser as reference point but should not influence appraiser's independent opinion of value per Dodd-Frank appraisal independence rules. Used by AMC for appraiser fee quotation based on complexity and value tier."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested in United States Dollars. Combined with estimated_property_value to calculate expected LTV for AMC's risk assessment and to determine if Enhanced Appraisal Review is required for high LTV loans (>90%) or high balance amounts (>$1M) per institutional quality control protocols."
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["Purchase", "Refinance", "Cash-Out Refinance"],
                    "description": "Transaction type classification. Purchase requires contract price as value baseline, Refinance requires current market value determination without purchase price anchor, Cash-Out Refinance requires careful comparable selection to ensure recent arms-length sales used rather than potential inflated estimates. Communicated to appraiser for proper assignment approach."
                }
                },
                "required": [
                "application_id",
                "prequalification_decision",
                "property_address",
                "property_type",
                "property_occupancy",
                "estimated_property_value",
                "requested_loan_amount",
                "loan_purpose"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ConductPropertyValuation",
            "description": "Executes comprehensive property appraisal by licensed appraiser conducting site inspection to measure property_square_footage using external perimeter methodology, assessing property_condition using standardized C1-C5 rating scale, photographing interior rooms and exterior elevations for report documentation, identifying and analyzing minimum three closed sales of comparable properties within 12 months and 1 mile radius matching subject property characteristics, constructing adjustment grid quantifying dollar differences between subject and each comparable, reconciling adjusted comparable values into final opinion of market value, recording site_inspection_timestamp and comparable_properties_count, completing URAR Form 1004 with all required sections.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this property valuation to the loan file and establishing final_appraised_value as definitive collateral value used for LTV calculation, loan amount validation, and collateral risk assessment throughout underwriting and portfolio management lifecycle."
                },
                "appraisal_order_number": {
                    "type": "string",
                    "description": "Unique order number assigned by OrderPropertyAppraisal API linking this valuation execution to the specific appraisal engagement. Used to retrieve appraisal order details, confirm appraiser assignment, track status progression from ordered to scheduled to inspected to completed, and maintain chain of custody for regulatory examination audit trail."
                },
                "property_address": {
                    "type": "string",
                    "description": "Complete street address, city, state, ZIP code of subject property being appraised. Used by appraiser to locate property for site inspection, search MLS and public records for comparable sales within 1-mile radius, retrieve tax assessor records for lot size and building characteristics verification, and establish neighborhood boundaries for location adjustments."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification guiding appraiser's comparable selection criteria. Single Family requires detached single-unit comps, Condo requires unit comps in similar projects with HOA review, Townhouse requires attached comps with similar configuration, Multi-Family requires 2-4 unit comps with income approach analysis for rent potential and gross rent multipliers."
                },
                "estimated_property_value": {
                    "type": "number",
                    "description": "Purchase contract price or applicant estimate in United States Dollars, provided for appraiser reference only. Should not anchor or influence appraiser's independent opinion of value per Dodd-Frank independence rules. Used by appraiser to select comparable sales in similar price range and validate reasonableness of market activity at subject's price point."
                }
                },
                "required": [
                "application_id",
                "appraisal_order_number",
                "property_address",
                "property_type",
                "estimated_property_value"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ValidateAppraisalWithAVM",
            "description": "Executes automated appraisal quality control by submitting property_address to institution's Automated Valuation Model platform which returns avm_estimated_value derived from statistical analysis of historical sales data, property characteristics from public records, hedonic pricing models, and market condition indices along with confidence score and forecast standard deviation, calculates appraisal_avm_variance_percent using formula [(final_appraised_value - avm_estimated_value) / avm_estimated_value] × 100, compares absolute variance value against 10.00% tolerance threshold, sets second_review_triggered Boolean to true if variance exceeds ±10.00%.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this AVM validation to the loan file and establishing second_review_triggered Boolean that gates workflow progression: if false, appraisal accepted and proceeds to LTV calculation; if true, secondary field review appraisal ordered requiring reconciliation before proceeding."
                },
                "property_address": {
                    "type": "string",
                    "description": "Complete street address, city, state, ZIP code of subject property being validated. Used to query AVM platform (CoreLogic HouseCanopy, Freddie Mac Home Value Explorer, or proprietary model) for electronic submission to retrieve statistically derived market value estimate, confidence score, and forecast standard deviation from AVM's historical sales database and hedonic pricing algorithms."
                },
                "final_appraised_value": {
                    "type": "number",
                    "description": "Final opinion of market value in United States Dollars from ConductPropertyValuation API, representing appraiser's reconciled value from comparable sales analysis documented in URAR Form 1004. Used as numerator component in variance calculation to determine deviation magnitude and direction from AVM estimate for quality control flagging."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification used to confirm AVM platform supports the property type for accurate valuation. Some AVM models have limited data for Condo or Multi-Family properties resulting in lower confidence scores or inability to provide estimate, triggering manual review protocols rather than automated variance comparison."
                }
                },
                "required": [
                "application_id",
                "property_address",
                "final_appraised_value",
                "property_type"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CalculateLoanToValueRatio",
            "description": "Computes definitive loan-to-value ratio using validated final_appraised_value and requested_loan_amount by applying formula LTV = (requested_loan_amount / MIN(final_appraised_value, purchase_price)) × 100 for purchase transactions where the lesser of appraised value or contract price establishes collateral value per prudential lending standards, or LTV = (requested_loan_amount / final_appraised_value) × 100 for refinance transactions where no purchase price exists, expressing result as percentage with precision to two decimal places, comparing calculated loan_to_value_ratio against institutional policy thresholds stratified by credit score tier, property type, and occupancy status.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this LTV calculation to the loan file and establishing loan_to_value_ratio as critical collateral risk metric determining mortgage insurance requirement, loan amount adjustment necessity, and policy compliance for proceeding to final decision rendering."
                },
                "final_appraised_value": {
                    "type": "number",
                    "description": "Validated final opinion of market value in United States Dollars from ValidateAppraisalWithAVM API, having passed AVM variance tolerance check or completed second review reconciliation if variance exceeded ±10%. Used as denominator in LTV formula or as MIN comparison with purchase_price for purchase transactions to establish conservative collateral value."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Dollar amount of mortgage financing requested by applicant in United States Dollars. Used as numerator in LTV formula. If calculated LTV exceeds policy maximum for applicant's credit tier, this amount must be reduced or applicant must increase down payment to achieve compliant LTV, triggering loan_amount_adjustment_required Boolean response."
                },
                "estimated_property_value": {
                    "type": "number",
                    "description": "Purchase contract price for purchase transactions in United States Dollars, or applicant estimate for refinance transactions. For purchases, used in MIN(final_appraised_value, purchase_price) comparison to establish conservative collateral value preventing LTV inflation if appraisal exceeds contract. For refinances, serves as reference but final_appraised_value is sole basis."
                },
                "loan_purpose": {
                    "type": "string",
                    "enum": ["Purchase", "Refinance", "Cash-Out Refinance"],
                    "description": "Transaction type classification determining LTV calculation methodology. Purchase uses MIN(appraised_value, contract_price), Refinance uses appraised_value only, Cash-Out Refinance uses appraised_value with additional scrutiny as higher LTV thresholds apply and institutional policy may impose maximum 80% LTV for cash-out transactions regardless of credit score."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "Qualifying FICO score from CalculateRepresentativeCreditScore API. Used to determine credit-score-stratified maximum LTV policy threshold: 740+ FICO may permit 95% LTV, 700-739 may permit 90% LTV, 680-699 may permit 85% LTV, 620-679 may permit 80% LTV, with thresholds varying by property type and occupancy status."
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Single Family", "Condo", "Townhouse", "Multi-Family"],
                    "description": "Property structural classification affecting maximum LTV policy threshold. Single Family typically permits highest LTV, Condo may have 5% LTV reduction if project is non-warrantable, Townhouse similar to Single Family, Multi-Family typically limited to 80% LTV maximum regardless of credit score due to marketability and default risk considerations."
                },
                "property_occupancy": {
                    "type": "string",
                    "enum": ["Primary Residence", "Investment Property", "Second Home"],
                    "description": "Property occupancy status significantly affecting maximum LTV policy. Primary Residence permits highest LTV based on credit tier, Second Home typically limited to 90% LTV maximum, Investment Property typically limited to 85% LTV for single-family or 75% LTV for multi-family regardless of credit score due to elevated default probability."
                }
                },
                "required": [
                "application_id",
                "final_appraised_value",
                "requested_loan_amount",
                "estimated_property_value",
                "loan_purpose",
                "representative_credit_score",
                "property_type",
                "property_occupancy"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "RenderFinalLoanDecision",
            "description": "Synthesizes all underwriting findings to render ultimate credit decision by executing comprehensive decision tree logic that evaluates debt_to_income_ratio against 43.00% QM threshold and 50.00% absolute maximum, loan_to_value_ratio against credit-score-stratified policy thresholds, credit_score_grade for A/B acceptability versus C requiring compensating factors or D requiring decline, property_condition for C3 or better acceptability versus C4/C5 requiring repairs or decline, probability_of_default_score for acceptable risk levels, compensating_factors_present Boolean for marginal DTI or LTV cases, and bankruptcy/foreclosure seasoning compliance.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this final decision to the loan file and establishing final_decision, final_decision_timestamp, and approved loan terms as ultimate underwriting outcomes that control progression to closing preparation or adverse action notice generation per ECOA requirements."
                },
                "debt_to_income_ratio": {
                    "type": "number",
                    "description": "Calculated DTI as percentage to two decimal places from CalculateDebtRatios API. Primary decision criterion: DTI≤43.00% meets QM standard and proceeds to approval consideration, DTI 43.01-50.00% requires compensating factors for exception approval, DTI>50.00% results in decline absent extraordinary circumstances requiring Chief Credit Officer approval."
                },
                "debt_service_coverage_ratio": {
                    "type": "number",
                    "description": "Calculated DSCR to two decimal places from CalculateDebtRatios API. Secondary capacity metric: DSCR≥1.15 indicates adequate margin for approval, DSCR 1.00-1.14 requires compensating factors analysis, DSCR<1.00 indicates negative cash flow and typically results in decline as borrower lacks capacity to meet all obligations."
                },
                "loan_to_value_ratio": {
                    "type": "number",
                    "description": "Calculated LTV as percentage to two decimal places from CalculateLoanToValueRatio API. Critical collateral risk metric compared against credit-score-stratified policy maximums: exceeding maximum LTV for applicant's credit tier results in loan amount reduction requirement or decline if applicant cannot increase down payment to achieve compliance."
                },
                "probability_of_default_score": {
                    "type": "number",
                    "description": "PD score as percentage to two decimal places from CalculateProbabilityOfDefault API representing 36-month default likelihood. Used to validate credit_score_grade assignment and identify applications with PD>7.0% requiring special Credit Committee review before approval, or PD>10.0% typically resulting in automatic decline due to unacceptable risk."
                },
                "credit_score_grade": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "Risk classification from CalculateProbabilityOfDefault API. Grade A/B applications proceed to approval if other metrics comply, Grade C requires compensating factors documentation and elevated pricing adjustments, Grade D requires decline or escalation to Credit Committee for extraordinary exception with comprehensive risk justification."
                },
                "final_risk_adjusted_interest_rate": {
                    "type": "number",
                    "description": "Interest rate as annual percentage to three decimal places from DetermineFinalInterestRate API incorporating all risk-based pricing adjustments. Becomes approved_interest_rate if application approved, used to calculate final monthly payment for Closing Disclosure, and establishes pricing for rate lock and loan documents."
                },
                "compensating_factors_present": {
                    "type": "boolean",
                    "description": "Boolean from EvaluateCompensatingFactors API indicating whether sufficient positive offsetting characteristics exist. Critical for marginal applications: DTI 43.01-45% requires compensating_factors_present=true with 5+ points, DTI 45.01-50% requires 7+ points, Grade C credit requires compensating factors documentation, without adequate factors marginal applications decline."
                },
                "property_condition": {
                    "type": "string",
                    "enum": ["C1", "C2", "C3", "C4", "C5"],
                    "description": "Condition rating from ConductPropertyValuation API on C1-C5 scale. C1/C2/C3 (new to normal wear) acceptable for standard approval, C4 (deferred maintenance) requires mandatory repair completion prior to closing with escrow holdback or repair completion certification, C5 (substantial deficiencies) results in decline until comprehensive rehabilitation completed."
                },
                "bankruptcy_seasoning_compliant": {
                    "type": "boolean",
                    "description": "Boolean from AnalyzeCreditHistory API indicating bankruptcy discharge timing compliance. Must be true for approval: Chapter 7 requires 4+ years from discharge, Chapter 13 requires 2+ years from discharge with satisfactory payment history, or 3 years for either with documented extenuating circumstances (job loss, medical, death of wage earner). False results in decline."
                },
                "foreclosure_history_compliant": {
                    "type": "boolean",
                    "description": "Boolean from AnalyzeCreditHistory API indicating foreclosure/short sale timing compliance. Must be true for approval: foreclosure/deed-in-lieu/short sale requires 7 years elapsed, or 3 years with documented extenuating circumstances and re-established satisfactory credit. False results in decline due to elevated re-default risk demonstrated by prior mortgage default."
                },
                "verified_gross_monthly_income": {
                    "type": "number",
                    "description": "Verified income in United States Dollars from VerifyEmploymentAndIncome API. Used to populate approved loan documents if approved, validate that income supports approved_loan_amount at calculated DTI, and confirm no material income changes occurred since verification that would invalidate capacity assessment requiring re-verification."
                },
                "requested_loan_amount": {
                    "type": "number",
                    "description": "Original loan amount requested in United States Dollars. Compared against maximum supportable amount based on DTI and LTV constraints: if requested amount exceeds maximum, approved_loan_amount set to lower figure that achieves policy compliance, requiring applicant acceptance of reduced loan amount or increased down payment before proceeding to closing."
                },
                "loan_term_months": {
                    "type": "integer",
                    "description": "Requested amortization term in months, typically 360 or 180. Becomes approved_loan_term_months if application approved, or may be modified (e.g., from 360 to 180) to improve DTI compliance through higher monthly payment if applicant capacity supports shorter term, though term modifications require explicit applicant consent."
                },
                "application_timestamp": {
                    "type": "string",
                    "description": "Date and time when application was received in ISO 8601 format with UTC timezone. Used with final_decision_timestamp to calculate turnaround_time_days for operational performance measurement against institutional target of ≤25 days for W-2 wage earners or ≤35 days for self-employed, identifying process improvement opportunities if exceeded."
                }
                },
                "required": [
                "application_id",
                "debt_to_income_ratio",
                "debt_service_coverage_ratio",
                "loan_to_value_ratio",
                "probability_of_default_score",
                "credit_score_grade",
                "final_risk_adjusted_interest_rate",
                "compensating_factors_present",
                "property_condition",
                "bankruptcy_seasoning_compliant",
                "foreclosure_history_compliant",
                "verified_gross_monthly_income",
                "requested_loan_amount",
                "loan_term_months",
                "application_timestamp"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ExecuteFairLendingMonitoring",
            "description": "Conducts automated fair lending compliance surveillance by capturing demographic information collected per Home Mortgage Disclosure Act requirements including applicant ethnicity, race, sex, and Census tract geocode derived from property_address, performing statistical analysis comparing approval rates across demographic cohorts after controlling for legitimate risk factors (credit score, DTI, LTV) to identify statistically significant disparities indicating potential disparate treatment or disparate impact violations of Equal Credit Opportunity Act, analyzing final_risk_adjusted_interest_rate spreads across demographic groups to detect pricing discrimination after risk adjustment, evaluating conditions_count patterns to identify inconsistent application of underwriting standards across protected classes.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking this fair lending surveillance to the loan file and establishing fair_lending_audit_flag Boolean that triggers detailed quality control file review by Fair Lending specialists if algorithmic surveillance detects potential discriminatory treatment or pricing anomalies requiring investigation."
                },
                "final_decision": {
                    "type": "string",
                    "enum": ["Approved", "Approved with Conditions", "Declined"],
                    "description": "Ultimate credit decision outcome from RenderFinalLoanDecision API. Used in statistical analysis comparing approval rates across demographic groups after controlling for risk factors: significant disparities in approval rates for protected class applicants versus control group with similar credit profiles triggers fair_lending_audit_flag for detailed file review and justification documentation."
                },
                "approved_interest_rate": {
                    "type": "number",
                    "description": "Final interest rate as annual percentage to three decimal places for approved loans. Optional field present only for Approved/Approved with Conditions decisions. Used to analyze pricing dispersion across demographic cohorts after controlling for credit score, LTV, DTI, property type: statistically significant rate spreads for protected classes trigger pricing_variance_alert requiring Fair Lending review."
                },
                "conditions_count": {
                    "type": "integer",
                    "description": "Number of outstanding conditions imposed for approved loans. Optional field present only for Approved with Conditions. Used to detect inconsistent application of underwriting standards: if protected class applicants systematically receive higher conditions_count than control group with equivalent risk profiles, triggers decisioning_consistency_alert requiring process review and underwriter training."
                },
                "representative_credit_score": {
                    "type": "integer",
                    "description": "Qualifying FICO score from CalculateRepresentativeCreditScore API. Used as primary control variable in statistical analysis isolating demographic effects: approval rate and pricing analyses stratify by credit score bands (780+, 740-779, 700-739, 680-699, 660-679, 640-659, 620-639) to ensure comparisons made within similar risk cohorts preventing false positive discrimination alerts."
                },
                "debt_to_income_ratio": {
                    "type": "number",
                    "description": "Calculated DTI as percentage to two decimal places from CalculateDebtRatios API. Used as secondary control variable in statistical models: approval rate regressions control for DTI bands (≤36%, 36.01-40%, 40.01-43%, 43.01-45%, 45.01-50%) ensuring demographic disparities detected represent true disparate treatment rather than legitimate risk-based differences in capacity."
                },
                "loan_to_value_ratio": {
                    "type": "number",
                    "description": "Calculated LTV as percentage to two decimal places from CalculateLoanToValueRatio API. Used as tertiary control variable isolating collateral risk: approval rate and pricing analyses stratify by LTV bands (≤60%, 60.01-70%, 70.01-80%, 80.01-85%, 85.01-90%, 90.01-95%) ensuring demographic comparisons made within similar collateral risk profiles."
                },
                "property_address": {
                    "type": "string",
                    "description": "Complete street address, city, state, ZIP code of subject property. Used to geocode to Census tract for HMDA reporting: Census tract geocode combined with applicant demographic data enables aggregate statistical analysis by geographic area to detect redlining patterns or geographic discrimination requiring Fair Lending investigation and potential Community Reinvestment Act implications."
                }
                },
                "required": [
                "application_id",
                "final_decision",
                "representative_credit_score",
                "debt_to_income_ratio",
                "loan_to_value_ratio",
                "property_address"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "TrackLoanPerformanceMetrics",
            "description": "Calculates operational and business performance metrics by computing turnaround_time_days as calendar days between application_timestamp and final_decision_timestamp to measure process efficiency against institutional target of ≤25 days for W-2 wage earner applications and ≤35 days for self-employed applications, records first_pass_yield Boolean based on conditions_count equaling zero (true for clean approval, false if conditions exist) to measure operational quality and file preparation completeness, tracks funded_status Boolean indicating whether approved loan ultimately closed and was disbursed, records funding_date when loan funds were advanced.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique identifier for the mortgage loan application, linking these performance metrics to the loan file for operational analytics dashboard population, executive reporting on process efficiency and quality, and continuous improvement initiatives targeting cycle time reduction and first-pass yield optimization."
                },
                "application_timestamp": {
                    "type": "string",
                    "description": "Date and time when application was initially received in ISO 8601 format with UTC timezone (YYYY-MM-DDTHH:MM:SSZ). Used as start point for turnaround_time_days calculation by subtracting from final_decision_timestamp and converting elapsed duration to calendar days integer for comparison against institutional performance targets."
                },
                "final_decision": {
                    "type": "string",
                    "enum": ["Approved", "Approved with Conditions", "Declined"],
                    "description": "Ultimate credit decision outcome from RenderFinalLoanDecision API. Used to filter performance metrics: turnaround_time_days relevant for all decisions, first_pass_yield only meaningful for Approved decisions, funded_status only applicable for Approved/Approved with Conditions decisions as Declined applications cannot fund."
                },
                "final_decision_timestamp": {
                    "type": "string",
                    "description": "Date and time when final underwriting decision was rendered in ISO 8601 format with UTC timezone. Used as end point for turnaround_time_days calculation: calendar days elapsed from application_timestamp to final_decision_timestamp measuring complete origination cycle efficiency, with outliers requiring root cause analysis."
                },
                "conditions_count": {
                    "type": "integer",
                    "description": "Number of outstanding conditions imposed for approved loans. Used to determine first_pass_yield Boolean: conditions_count=0 sets first_pass_yield=true indicating clean approval requiring no additional documentation or clarification, optimal operational outcome; conditions_count>0 sets first_pass_yield=false indicating rework required, targeting process improvement to increase first-pass rate."
                },
                "approved_loan_amount": {
                    "type": "number",
                    "description": "Approved loan amount in United States Dollars for approved applications. Optional field present only for Approved/Approved with Conditions decisions. Used for funded volume tracking, pull-through rate analysis (funded applications ÷ approved applications), and revenue recognition calculations based on interest income projections from approved amounts that successfully close."
                }
                },
                "required": [
                "application_id",
                "application_timestamp",
                "final_decision",
                "final_decision_timestamp",
                "conditions_count"
                ]
            }
            }
        }
        }
        </SCHEMA>'''
    sample_data = '''
        <data>
        ['application_id:MLN-2025-048732', 'application_timestamp:2025-04-18T10:15:30Z', 'loan_officer_id:LO-8492', 'applicant_name:John Michael Smith', 'applicant_ssn:***-**-4829', 'applicant_dob:1985-06-15', 'applicant_address:1234 Oak Street, Springfield, IL 62704', 'co_applicant_name:Jane Elizabeth Smith', 'co_applicant_ssn:***-**-7391', 'employment_status:Full-Time', 'employer_name:Acme Corporation', 'employment_years:5.5', 'gross_monthly_income:8500.00', 'additional_income_monthly:1200.00', 'liquid_assets:75000.00', 'retirement_accounts:150000.00', 'requested_loan_amount:380000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:5678 Elm Avenue, Springfield, IL 62705', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:450000.00', 'down_payment_amount:90000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-18T10:22:45Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-20T14:30:00Z', 'credit_pull_timestamp:2025-04-18T10:25:00Z', 'fico_score_equifax:742', 'fico_score_experian:738', 'fico_score_transunion:745', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2800.00', 'appraisal_order_timestamp:2025-04-21T09:00:00Z', 'appraiser_id:APR-3291', 'appraisal_company:National Valuation Services', 'site_inspection_timestamp:2025-04-23T13:00:00Z', 'property_square_footage:2400', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-6174', 'processor_id:LP-2938', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-049103', 'application_timestamp:2025-04-19T09:30:15Z', 'loan_officer_id:LO-7234', 'applicant_name:Maria Garcia Rodriguez', 'applicant_ssn:***-**-6721', 'applicant_dob:1982-03-22', 'applicant_address:890 Pine Road, Austin, TX 78701', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Self-Employed', 'employer_name:MG Consulting LLC', 'employment_years:8.0', 'gross_monthly_income:11200.00', 'additional_income_monthly:2400.00', 'liquid_assets:125000.00', 'retirement_accounts:220000.00', 'requested_loan_amount:525000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:3456 Maple Drive, Austin, TX 78702', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:625000.00', 'down_payment_amount:125000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-19T09:45:30Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-04-22T11:15:00Z', 'credit_pull_timestamp:2025-04-19T09:50:00Z', 'fico_score_equifax:798', 'fico_score_experian:805', 'fico_score_transunion:792', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:1850.00', 'appraisal_order_timestamp:2025-04-23T10:30:00Z', 'appraiser_id:APR-4512', 'appraisal_company:Premier Appraisal Group', 'site_inspection_timestamp:2025-04-26T14:00:00Z', 'property_square_footage:3200', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-8823', 'processor_id:LP-4621', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-049287', 'application_timestamp:2025-04-20T14:20:00Z', 'loan_officer_id:LO-5639', 'applicant_name:Robert James Thompson', 'applicant_ssn:***-**-9182', 'applicant_dob:1978-11-08', 'applicant_address:2345 Cedar Lane, Denver, CO 80202', 'co_applicant_name:Susan Marie Thompson', 'co_applicant_ssn:***-**-3047', 'employment_status:Full-Time', 'employer_name:TechCorp Industries', 'employment_years:12.0', 'gross_monthly_income:14500.00', 'additional_income_monthly:800.00', 'liquid_assets:185000.00', 'retirement_accounts:425000.00', 'requested_loan_amount:640000.00', 'loan_purpose:Cash-Out Refinance', 'loan_term_months:360', 'property_address:2345 Cedar Lane, Denver, CO 80202', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:800000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-20T14:35:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-23T09:45:00Z', 'credit_pull_timestamp:2025-04-20T14:40:00Z', 'fico_score_equifax:712', 'fico_score_experian:718', 'fico_score_transunion:709', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:4200.00', 'appraisal_order_timestamp:2025-04-24T08:00:00Z', 'appraiser_id:APR-2876', 'appraisal_company:Rocky Mountain Valuations', 'site_inspection_timestamp:2025-04-27T10:30:00Z', 'property_square_footage:3800', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-5491', 'processor_id:LP-7832', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-049421', 'application_timestamp:2025-04-21T11:00:00Z', 'loan_officer_id:LO-9123', 'applicant_name:David Lee Chen', 'applicant_ssn:***-**-5438', 'applicant_dob:1990-07-14', 'applicant_address:6789 Birch Avenue, Seattle, WA 98101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Amazon Web Services', 'employment_years:3.5', 'gross_monthly_income:9800.00', 'additional_income_monthly:500.00', 'liquid_assets:45000.00', 'retirement_accounts:68000.00', 'requested_loan_amount:420000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:1122 Spruce Court, Seattle, WA 98102', 'property_type:Condo', 'property_occupancy:Primary Residence', 'estimated_property_value:480000.00', 'down_payment_amount:60000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-21T11:15:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-24T13:20:00Z', 'credit_pull_timestamp:2025-04-21T11:20:00Z', 'fico_score_equifax:685', 'fico_score_experian:692', 'fico_score_transunion:688', 'derogatory_marks_count:3', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3600.00', 'appraisal_order_timestamp:2025-04-25T09:30:00Z', 'appraiser_id:APR-6701', 'appraisal_company:Pacific Northwest Appraisals', 'site_inspection_timestamp:2025-04-28T15:00:00Z', 'property_square_footage:1450', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-3214', 'processor_id:LP-5987', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-049634', 'application_timestamp:2025-04-22T08:45:00Z', 'loan_officer_id:LO-4728', 'applicant_name:Jennifer Ann Wilson', 'applicant_ssn:***-**-2914', 'applicant_dob:1987-01-29', 'applicant_address:4567 Willow Street, Phoenix, AZ 85001', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Banner Health System', 'employment_years:6.0', 'gross_monthly_income:7200.00', 'additional_income_monthly:0.00', 'liquid_assets:28000.00', 'retirement_accounts:82000.00', 'requested_loan_amount:295000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:7890 Aspen Way, Phoenix, AZ 85003', 'property_type:Townhouse', 'property_occupancy:Primary Residence', 'estimated_property_value:310000.00', 'down_payment_amount:31000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-22T09:00:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-25T10:30:00Z', 'credit_pull_timestamp:2025-04-22T09:05:00Z', 'fico_score_equifax:728', 'fico_score_experian:735', 'fico_score_transunion:731', 'derogatory_marks_count:1', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2400.00', 'appraisal_order_timestamp:2025-04-26T11:00:00Z', 'appraiser_id:APR-8934', 'appraisal_company:Desert Valley Appraisals', 'site_inspection_timestamp:2025-04-29T09:30:00Z', 'property_square_footage:1850', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-7652', 'processor_id:LP-3401', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-049852', 'application_timestamp:2025-04-23T13:30:00Z', 'loan_officer_id:LO-6182', 'applicant_name:Michael Patrick O Brien', 'applicant_ssn:***-**-7256', 'applicant_dob:1975-09-03', 'applicant_address:3210 Redwood Drive, Portland, OR 97201', 'co_applicant_name:Katherine Lynn O Brien', 'co_applicant_ssn:***-**-4893', 'employment_status:Self-Employed', 'employer_name:O Brien Architecture PLLC', 'employment_years:15.0', 'gross_monthly_income:13800.00', 'additional_income_monthly:3200.00', 'liquid_assets:210000.00', 'retirement_accounts:580000.00', 'requested_loan_amount:720000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:5432 Fir Lane, Portland, OR 97202', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:900000.00', 'down_payment_amount:225000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-23T13:45:00Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-04-26T15:00:00Z', 'credit_pull_timestamp:2025-04-23T13:50:00Z', 'fico_score_equifax:815', 'fico_score_experian:808', 'fico_score_transunion:822', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2100.00', 'appraisal_order_timestamp:2025-04-27T08:30:00Z', 'appraiser_id:APR-1529', 'appraisal_company:Northwest Property Valuations', 'site_inspection_timestamp:2025-04-30T11:00:00Z', 'property_square_footage:4200', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-9245', 'processor_id:LP-6738', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-050078', 'application_timestamp:2025-04-24T10:00:00Z', 'loan_officer_id:LO-8371', 'applicant_name:Angela Marie Washington', 'applicant_ssn:***-**-3678', 'applicant_dob:1992-05-18', 'applicant_address:8901 Magnolia Boulevard, Atlanta, GA 30301', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Delta Airlines Inc', 'employment_years:4.0', 'gross_monthly_income:6800.00', 'additional_income_monthly:1100.00', 'liquid_assets:35000.00', 'retirement_accounts:47000.00', 'requested_loan_amount:285000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:2468 Peachtree Circle, Atlanta, GA 30302', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:300000.00', 'down_payment_amount:30000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-24T10:15:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-27T11:45:00Z', 'credit_pull_timestamp:2025-04-24T10:20:00Z', 'fico_score_equifax:652', 'fico_score_experian:648', 'fico_score_transunion:655', 'derogatory_marks_count:4', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2900.00', 'appraisal_order_timestamp:2025-04-28T09:00:00Z', 'appraiser_id:APR-5647', 'appraisal_company:Southern Regional Appraisals', 'site_inspection_timestamp:2025-05-01T14:30:00Z', 'property_square_footage:1650', 'property_condition:Fair', 'comparable_properties_count:3', 'underwriter_id:UW-4823', 'processor_id:LP-9156', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-050241', 'application_timestamp:2025-04-25T15:20:00Z', 'loan_officer_id:LO-2947', 'applicant_name:Thomas Edward Martinez', 'applicant_ssn:***-**-8429', 'applicant_dob:1980-12-11', 'applicant_address:1357 Sycamore Street, Miami, FL 33101', 'co_applicant_name:Carmen Isabel Martinez', 'co_applicant_ssn:***-**-5762', 'employment_status:Full-Time', 'employer_name:Baptist Health South Florida', 'employment_years:9.5', 'gross_monthly_income:10500.00', 'additional_income_monthly:900.00', 'liquid_assets:62000.00', 'retirement_accounts:145000.00', 'requested_loan_amount:465000.00', 'loan_purpose:Refinance', 'loan_term_months:360', 'property_address:1357 Sycamore Street, Miami, FL 33101', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:550000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-25T15:35:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-28T14:00:00Z', 'credit_pull_timestamp:2025-04-25T15:40:00Z', 'fico_score_equifax:758', 'fico_score_experian:762', 'fico_score_transunion:755', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3100.00', 'appraisal_order_timestamp:2025-04-29T10:30:00Z', 'appraiser_id:APR-7283', 'appraisal_company:Sunshine State Valuations', 'site_inspection_timestamp:2025-05-02T13:00:00Z', 'property_square_footage:2650', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-6417', 'processor_id:LP-8029', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-050463', 'application_timestamp:2025-04-26T09:15:00Z', 'loan_officer_id:LO-5184', 'applicant_name:Sarah Elizabeth Brown', 'applicant_ssn:***-**-9341', 'applicant_dob:1988-08-25', 'applicant_address:9876 Oakridge Avenue, Boston, MA 02101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Massachusetts General Hospital', 'employment_years:7.0', 'gross_monthly_income:12200.00', 'additional_income_monthly:1500.00', 'liquid_assets:95000.00', 'retirement_accounts:178000.00', 'requested_loan_amount:580000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:6543 Commonwealth Drive, Boston, MA 02102', 'property_type:Condo', 'property_occupancy:Primary Residence', 'estimated_property_value:650000.00', 'down_payment_amount:97500.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-26T09:30:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-29T10:15:00Z', 'credit_pull_timestamp:2025-04-26T09:35:00Z', 'fico_score_equifax:776', 'fico_score_experian:782', 'fico_score_transunion:779', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2700.00', 'appraisal_order_timestamp:2025-04-30T08:00:00Z', 'appraiser_id:APR-3928', 'appraisal_company:New England Property Assessors', 'site_inspection_timestamp:2025-05-03T10:00:00Z', 'property_square_footage:1950', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-2735', 'processor_id:LP-4892', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-050697', 'application_timestamp:2025-04-27T12:45:00Z', 'loan_officer_id:LO-7562', 'applicant_name:Kevin James Anderson', 'applicant_ssn:***-**-1825', 'applicant_dob:1983-02-17', 'applicant_address:7531 Elmwood Terrace, Chicago, IL 60601', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:United Airlines Holdings', 'employment_years:2.0', 'gross_monthly_income:8900.00', 'additional_income_monthly:600.00', 'liquid_assets:42000.00', 'retirement_accounts:55000.00', 'requested_loan_amount:340000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:2109 Lakeview Avenue, Chicago, IL 60602', 'property_type:Condo', 'property_occupancy:Investment Property', 'estimated_property_value:380000.00', 'down_payment_amount:95000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-27T13:00:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-30T11:30:00Z', 'credit_pull_timestamp:2025-04-27T13:05:00Z', 'fico_score_equifax:698', 'fico_score_experian:705', 'fico_score_transunion:702', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3200.00', 'appraisal_order_timestamp:2025-05-01T09:00:00Z', 'appraiser_id:APR-4651', 'appraisal_company:Midwest Appraisal Services', 'site_inspection_timestamp:2025-05-04T14:00:00Z', 'property_square_footage:1550', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-8164', 'processor_id:LP-5237', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-050829', 'application_timestamp:2025-04-28T08:30:00Z', 'loan_officer_id:LO-3698', 'applicant_name:Lisa Ann Johnson', 'applicant_ssn:***-**-4972', 'applicant_dob:1979-06-30', 'applicant_address:4682 Poplar Street, Nashville, TN 37201', 'co_applicant_name:Mark Steven Johnson', 'co_applicant_ssn:***-**-8135', 'employment_status:Self-Employed', 'employer_name:Johnson Tax Services LLC', 'employment_years:11.0', 'gross_monthly_income:9700.00', 'additional_income_monthly:1800.00', 'liquid_assets:82000.00', 'retirement_accounts:195000.00', 'requested_loan_amount:395000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:8024 Hickory Lane, Nashville, TN 37202', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:460000.00', 'down_payment_amount:92000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-28T08:45:00Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-05-01T13:00:00Z', 'credit_pull_timestamp:2025-04-28T08:50:00Z', 'fico_score_equifax:744', 'fico_score_experian:738', 'fico_score_transunion:751', 'derogatory_marks_count:1', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2650.00', 'appraisal_order_timestamp:2025-05-02T10:00:00Z', 'appraiser_id:APR-9217', 'appraisal_company:Music City Valuations', 'site_inspection_timestamp:2025-05-05T11:30:00Z', 'property_square_footage:2350', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-5946', 'processor_id:LP-7418', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-051043', 'application_timestamp:2025-04-29T14:10:00Z', 'loan_officer_id:LO-8925', 'applicant_name:Christopher Alan Davis', 'applicant_ssn:***-**-6284', 'applicant_dob:1986-04-09', 'applicant_address:3197 Walnut Grove, San Diego, CA 92101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Qualcomm Incorporated', 'employment_years:8.5', 'gross_monthly_income:11800.00', 'additional_income_monthly:700.00', 'liquid_assets:118000.00', 'retirement_accounts:245000.00', 'requested_loan_amount:685000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:5876 Pacific View Drive, San Diego, CA 92102', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:760000.00', 'down_payment_amount:114000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-29T14:25:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-02T09:45:00Z', 'credit_pull_timestamp:2025-04-29T14:30:00Z', 'fico_score_equifax:787', 'fico_score_experian:793', 'fico_score_transunion:790', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3400.00', 'appraisal_order_timestamp:2025-05-03T08:30:00Z', 'appraiser_id:APR-6832', 'appraisal_company:Southern California Appraisers', 'site_inspection_timestamp:2025-05-06T10:00:00Z', 'property_square_footage:3100', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-4571', 'processor_id:LP-9283', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-051278', 'application_timestamp:2025-04-30T11:20:00Z', 'loan_officer_id:LO-4317', 'applicant_name:Patricia Marie Taylor', 'applicant_ssn:***-**-3519', 'applicant_dob:1991-10-12', 'applicant_address:6254 Chestnut Avenue, Philadelphia, PA 19101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Comcast Corporation', 'employment_years:5.0', 'gross_monthly_income:7800.00', 'additional_income_monthly:400.00', 'liquid_assets:38000.00', 'retirement_accounts:72000.00', 'requested_loan_amount:315000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:1793 Liberty Court, Philadelphia, PA 19102', 'property_type:Townhouse', 'property_occupancy:Primary Residence', 'estimated_property_value:350000.00', 'down_payment_amount:52500.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-30T11:35:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-03T14:20:00Z', 'credit_pull_timestamp:2025-04-30T11:40:00Z', 'fico_score_equifax:671', 'fico_score_experian:678', 'fico_score_transunion:674', 'derogatory_marks_count:3', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2950.00', 'appraisal_order_timestamp:2025-05-04T09:30:00Z', 'appraiser_id:APR-5194', 'appraisal_company:Liberty Bell Appraisals', 'site_inspection_timestamp:2025-05-07T13:00:00Z', 'property_square_footage:1750', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-7382', 'processor_id:LP-2956', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-051456', 'application_timestamp:2025-05-01T09:00:00Z', 'loan_officer_id:LO-6729', 'applicant_name:Daniel Robert Miller', 'applicant_ssn:***-**-7941', 'applicant_dob:1984-01-20', 'applicant_address:9142 Riverside Drive, Minneapolis, MN 55401', 'co_applicant_name:Emily Rose Miller', 'co_applicant_ssn:***-**-4263', 'employment_status:Full-Time', 'employer_name:Target Corporation', 'employment_years:10.0', 'gross_monthly_income:9500.00', 'additional_income_monthly:1000.00', 'liquid_assets:72000.00', 'retirement_accounts:165000.00', 'requested_loan_amount:420000.00', 'loan_purpose:Refinance', 'loan_term_months:180', 'property_address:9142 Riverside Drive, Minneapolis, MN 55401', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:525000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-01T09:15:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-04T10:30:00Z', 'credit_pull_timestamp:2025-05-01T09:20:00Z', 'fico_score_equifax:766', 'fico_score_experian:772', 'fico_score_transunion:769', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2500.00', 'appraisal_order_timestamp:2025-05-05T08:00:00Z', 'appraiser_id:APR-8376', 'appraisal_company:North Star Valuations', 'site_inspection_timestamp:2025-05-08T11:00:00Z', 'property_square_footage:2800', 'property_condition:Excellent', 'comparable_properties_count:3', 'underwriter_id:UW-3619', 'processor_id:LP-6842', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-051683', 'application_timestamp:2025-05-02T13:40:00Z', 'loan_officer_id:LO-2854', 'applicant_name:Brian Christopher Moore', 'applicant_ssn:***-**-5628', 'applicant_dob:1989-07-05', 'applicant_address:5318 Mountain View Road, Salt Lake City, UT 84101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Intermountain Healthcare', 'employment_years:6.5', 'gross_monthly_income:8200.00', 'additional_income_monthly:500.00', 'liquid_assets:48000.00', 'retirement_accounts:95000.00', 'requested_loan_amount:365000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:7629 Wasatch Boulevard, Salt Lake City, UT 84102', 'property_type:Single Family', 'property_occupancy:Second Home', 'estimated_property_value:425000.00', 'down_payment_amount:85000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-02T13:55:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-05T11:20:00Z', 'credit_pull_timestamp:2025-05-02T14:00:00Z', 'fico_score_equifax:721', 'fico_score_experian:728', 'fico_score_transunion:724', 'derogatory_marks_count:1', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2800.00', 'appraisal_order_timestamp:2025-05-06T09:00:00Z', 'appraiser_id:APR-4925', 'appraisal_company:Mountain West Appraisals', 'site_inspection_timestamp:2025-05-09T10:30:00Z', 'property_square_footage:2250', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-8257', 'processor_id:LP-5124', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-051897', 'application_timestamp:2025-05-03T10:25:00Z', 'loan_officer_id:LO-9436', 'applicant_name:Rebecca Lynn Harris', 'applicant_ssn:***-**-2174', 'applicant_dob:1993-11-28', 'applicant_address:4821 Palmetto Drive, Charlotte, NC 28201', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Bank of America', 'employment_years:4.5', 'gross_monthly_income:7400.00', 'additional_income_monthly:300.00', 'liquid_assets:32000.00', 'retirement_accounts:58000.00', 'requested_loan_amount:280000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:9517 Carolina Avenue, Charlotte, NC 28202', 'property_type:Condo', 'property_occupancy:Primary Residence', 'estimated_property_value:295000.00', 'down_payment_amount:29500.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-03T10:40:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-06T13:00:00Z', 'credit_pull_timestamp:2025-05-03T10:45:00Z', 'fico_score_equifax:641', 'fico_score_experian:638', 'fico_score_transunion:644', 'derogatory_marks_count:5', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3100.00', 'appraisal_order_timestamp:2025-05-07T10:00:00Z', 'appraiser_id:APR-7148', 'appraisal_company:Queen City Valuations', 'site_inspection_timestamp:2025-05-10T14:00:00Z', 'property_square_footage:1350', 'property_condition:Fair', 'comparable_properties_count:3', 'underwriter_id:UW-6923', 'processor_id:LP-3785', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-052114', 'application_timestamp:2025-05-04T15:15:00Z', 'loan_officer_id:LO-5762', 'applicant_name:Gregory Paul White', 'applicant_ssn:***-**-8396', 'applicant_dob:1981-04-16', 'applicant_address:8374 Riverfront Plaza, St Louis, MO 63101', 'co_applicant_name:Amanda Jane White', 'co_applicant_ssn:***-**-1529', 'employment_status:Self-Employed', 'employer_name:White Consulting Group LLC', 'employment_years:13.0', 'gross_monthly_income:15200.00', 'additional_income_monthly:2800.00', 'liquid_assets:245000.00', 'retirement_accounts:630000.00', 'requested_loan_amount:820000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:2846 Gateway Terrace, St Louis, MO 63102', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:1025000.00', 'down_payment_amount:256250.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-04T15:30:00Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-05-07T14:45:00Z', 'credit_pull_timestamp:2025-05-04T15:35:00Z', 'fico_score_equifax:832', 'fico_score_experian:825', 'fico_score_transunion:838', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2350.00', 'appraisal_order_timestamp:2025-05-08T08:30:00Z', 'appraiser_id:APR-3681', 'appraisal_company:Gateway Arch Appraisals', 'site_inspection_timestamp:2025-05-11T10:00:00Z', 'property_square_footage:4800', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-9574', 'processor_id:LP-8216', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-052348', 'application_timestamp:2025-05-05T08:50:00Z', 'loan_officer_id:LO-4183', 'applicant_name:Nicole Marie Clark', 'applicant_ssn:***-**-6742', 'applicant_dob:1994-09-02', 'applicant_address:6195 Highland Park, Pittsburgh, PA 15201', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:UPMC Health System', 'employment_years:3.0', 'gross_monthly_income:6500.00', 'additional_income_monthly:200.00', 'liquid_assets:22000.00', 'retirement_accounts:35000.00', 'requested_loan_amount:245000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:4729 Steel City Lane, Pittsburgh, PA 15202', 'property_type:Townhouse', 'property_occupancy:Primary Residence', 'estimated_property_value:258000.00', 'down_payment_amount:25800.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-05T09:05:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-08T11:30:00Z', 'credit_pull_timestamp:2025-05-05T09:10:00Z', 'fico_score_equifax:612', 'fico_score_experian:608', 'fico_score_transunion:615', 'derogatory_marks_count:6', 'bankruptcy_history:True', 'total_monthly_debt_obligations:2700.00', 'appraisal_order_timestamp:2025-05-09T09:00:00Z', 'appraiser_id:APR-5293', 'appraisal_company:Three Rivers Appraisals', 'site_inspection_timestamp:2025-05-12T13:00:00Z', 'property_square_footage:1450', 'property_condition:Fair', 'comparable_properties_count:3', 'underwriter_id:UW-7485', 'processor_id:LP-2619', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-052571', 'application_timestamp:2025-05-06T12:30:00Z', 'loan_officer_id:LO-8649', 'applicant_name:Andrew Thomas Lewis', 'applicant_ssn:***-**-9517', 'applicant_dob:1985-12-19', 'applicant_address:9638 Desert Bloom Road, Las Vegas, NV 89101', 'co_applicant_name:Michelle Lynn Lewis', 'co_applicant_ssn:***-**-3842', 'employment_status:Full-Time', 'employer_name:MGM Resorts International', 'employment_years:7.0', 'gross_monthly_income:10100.00', 'additional_income_monthly:1300.00', 'liquid_assets:88000.00', 'retirement_accounts:135000.00', 'requested_loan_amount:485000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:5213 Vegas Valley Drive, Las Vegas, NV 89102', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:550000.00', 'down_payment_amount:82500.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-06T12:45:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-09T10:15:00Z', 'credit_pull_timestamp:2025-05-06T12:50:00Z', 'fico_score_equifax:753', 'fico_score_experian:747', 'fico_score_transunion:760', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3050.00', 'appraisal_order_timestamp:2025-05-10T08:00:00Z', 'appraiser_id:APR-6418', 'appraisal_company:Silver State Valuations', 'site_inspection_timestamp:2025-05-13T11:00:00Z', 'property_square_footage:2750', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-5198', 'processor_id:LP-9437', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-052796', 'application_timestamp:2025-05-07T14:00:00Z', 'loan_officer_id:LO-3527', 'applicant_name:Stephanie Rose Walker', 'applicant_ssn:***-**-4186', 'applicant_dob:1990-03-07', 'applicant_address:7412 Bayshore Boulevard, Tampa, FL 33601', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Raymond James Financial', 'employment_years:5.5', 'gross_monthly_income:9200.00', 'additional_income_monthly:800.00', 'liquid_assets:65000.00', 'retirement_accounts:112000.00', 'requested_loan_amount:390000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:3857 Gulf Coast Drive, Tampa, FL 33602', 'property_type:Condo', 'property_occupancy:Primary Residence', 'estimated_property_value:435000.00', 'down_payment_amount:65250.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-07T14:15:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-10T12:30:00Z', 'credit_pull_timestamp:2025-05-07T14:20:00Z', 'fico_score_equifax:734', 'fico_score_experian:741', 'fico_score_transunion:737', 'derogatory_marks_count:1', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2850.00', 'appraisal_order_timestamp:2025-05-11T09:00:00Z', 'appraiser_id:APR-9527', 'appraisal_company:Tampa Bay Appraisals', 'site_inspection_timestamp:2025-05-14T10:30:00Z', 'property_square_footage:1850', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-4362', 'processor_id:LP-7195', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-053019', 'application_timestamp:2025-05-08T09:45:00Z', 'loan_officer_id:LO-7894', 'applicant_name:Jonathan David Hall', 'applicant_ssn:***-**-5924', 'applicant_dob:1987-08-11', 'applicant_address:5927 Northside Parkway, Indianapolis, IN 46201', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Eli Lilly and Company', 'employment_years:9.0', 'gross_monthly_income:11400.00', 'additional_income_monthly:600.00', 'liquid_assets:102000.00', 'retirement_accounts:225000.00', 'requested_loan_amount:495000.00', 'loan_purpose:Cash-Out Refinance', 'loan_term_months:360', 'property_address:5927 Northside Parkway, Indianapolis, IN 46201', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:660000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-08T10:00:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-11T13:15:00Z', 'credit_pull_timestamp:2025-05-08T10:05:00Z', 'fico_score_equifax:719', 'fico_score_experian:726', 'fico_score_transunion:722', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3700.00', 'appraisal_order_timestamp:2025-05-12T08:30:00Z', 'appraiser_id:APR-2746', 'appraisal_company:Hoosier State Appraisals', 'site_inspection_timestamp:2025-05-15T09:00:00Z', 'property_square_footage:3200', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-8641', 'processor_id:LP-4523', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        ['application_id:MLN-2025-053247', 'application_timestamp:2025-05-09T11:10:00Z', 'loan_officer_id:LO-6215', 'applicant_name:Rachel Marie Young', 'applicant_ssn:***-**-7318', 'applicant_dob:1995-06-23', 'applicant_address:8146 Mountain Ridge, Raleigh, NC 27601', 'co_applicant_name:Joshua Allen Young', 'co_applicant_ssn:***-**-2954', 'employment_status:Full-Time', 'employer_name:Cisco Systems', 'employment_years:2.5', 'gross_monthly_income:8600.00', 'additional_income_monthly:400.00', 'liquid_assets:41000.00', 'retirement_accounts:62000.00', 'requested_loan_amount:335000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:4295 Research Park Drive, Raleigh, NC 27602', 'property_type:Townhouse', 'property_occupancy:Primary Residence', 'estimated_property_value:375000.00', 'down_payment_amount:56250.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-09T11:25:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-12T14:00:00Z', 'credit_pull_timestamp:2025-05-09T11:30:00Z', 'fico_score_equifax:695', 'fico_score_experian:702', 'fico_score_transunion:698', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3150.00', 'appraisal_order_timestamp:2025-05-13T09:30:00Z', 'appraiser_id:APR-8164', 'appraisal_company:Triangle Area Valuations', 'site_inspection_timestamp:2025-05-16T11:00:00Z', 'property_square_footage:1900', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-3956', 'processor_id:LP-6287', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:']
        </data>'''
    yaml_file = "Loans/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_API_Function_Code.yaml"
    system_prompt = "You are an SOP-related API tool Python code implementation generator. The functions you generate, strictly based on the finalized JSON template, must comply with Python syntax rules, and input parameters must be either already present in the sample_data or obtainable through calculations from other API functions. **You must generate all the code at once, without interruption.** The code you generate must also properly maintain and record all data entries contained within <data></data> and allow reading and writing by other APIs. Additionally, design a function to retrieve the corresponding data entry information from the data you maintain based on application_id. Fields that already have values in the data are user's original data; empty fields are data that need to be obtained through API processing, and once obtained via an API, they should be written back to the corresponding entry in the data you maintain, completing the update. The code you generate must consist entirely of deterministic calculations; the use of random numbers, such as random, is strictly prohibited."

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, api, sample_data)

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
