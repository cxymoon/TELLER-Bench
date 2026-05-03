import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():
    # === 输入参数 ===
    system_prompt = "I will provide you with a detailed SOP process, data schema, data dataset, and defined function APIs. Based on these, you need to generate a piece of Python code that can process each entry in the dataset using the SOP API functions in a calling order that complies with both the actual situation of the data entry and the SOP workflow, and output the result of the last called API. You must execute each API exactly as specified in your defined sequence; modifying or simplifying the original API logic is strictly prohibited. It is acceptable for the output to be lengthy. Your generated code must be able to process all data in the same format as the <schema></schema> and example <data></data>. Your final standard output should have each ID on a separate line, followed by each output field name and its value. For example: 'application_id:MLN-2025-048732', 'xxx:yyy', ....When calculating the standard results, pay attention to boundary cases such as division by zero."

    # 初始化客户端
    client = OpenAI(
        api_key="your_api_key",  # 替换为你的API Key
        base_url="https://..."  # API中转地址
    )
    # === 读取YAML并填充 ===
    user_prompt = '''
        I will provide you with a detailed SOP process, data schema, data dataset, and defined function APIs. Based on these, you need to generate a piece of Python code that can process each entry in the dataset using the SOP API functions in a calling order that complies with both the actual situation of the data entry and the SOP workflow, and output the result of the last called API.
        ***The SOP process is as follows***:
        <SOP>

        1. Purpose

        This Standard Operating Procedure establishes the comprehensive, deterministic framework governing the origination, credit assessment, collateral valuation, and decisioning of retail residential mortgage loan applications within the consumer lending division of the financial institution, ensuring full compliance with applicable federal regulations including but not limited to the Truth in Lending Act as implemented through Regulation Z, the Equal Credit Opportunity Act as implemented through Regulation B, the Home Mortgage Disclosure Act as implemented through Regulation C, and the Dodd-Frank Wall Street Reform and Consumer Protection Act's Ability-to-Repay and Qualified Mortgage provisions as codified in 12 CFR 1026.43, while simultaneously achieving the institution's risk management objectives through the systematic application of credit underwriting standards, collateral valuation protocols, and risk-based pricing methodologies that yield specific, non-discretionary numerical outputs including but not limited to maximum pre-qualified loan amounts, debt-to-income ratios, loan-to-value ratios, probability of default scores, credit score grades, final risk-adjusted interest rates, and ultimate binary approval or decline decisions with associated quantitative justifications. The procedure integrates three interdependent sub-processes—Mortgage Application Intake and Pre-qualification, Credit Underwriting, and Property Valuation—into a unified workflow that transforms applicant-provided data and third-party verification information into actionable lending decisions supported by auditable calculations and documented risk assessments suitable for secondary market sale, portfolio retention, or regulatory examination.

        2. Scope

        This Standard Operating Procedure applies to all first-lien residential mortgage loan applications submitted by individual retail customers seeking financing for the purchase, refinance, or cash-out refinance of one-to-four unit residential properties located within the institution's approved lending footprint, encompassing applications for conforming loan amounts as defined by the Federal Housing Finance Agency's annual adjustment of baseline and high-cost area limits, as well as non-conforming jumbo loan amounts up to the institution's maximum exposure threshold of five million United States Dollars, and governs the activities of all personnel involved in the mortgage origination process including Loan Officers (Relationship Managers) responsible for customer interaction and initial data collection, Loan Processors responsible for document compilation and file preparation, Credit Underwriters responsible for financial analysis and credit decisioning, Appraisers and Appraisal Management Company coordinators responsible for collateral valuation, and Quality Control personnel responsible for pre-funding and post-closing file reviews. The procedure encompasses applications processed through the institution's Loan Origination System and evaluated through both automated underwriting engines and manual underwriting protocols, covering conventional mortgage products eligible for sale to Government-Sponsored Enterprises as well as portfolio products retained on the institution's balance sheet, but explicitly excludes government-insured or government-guaranteed loan products such as Federal Housing Administration, Department of Veterans Affairs, or United States Department of Agriculture Rural Development mortgages, which are governed by separate standard operating procedures reflecting agency-specific underwriting and processing requirements, and further excludes commercial real estate loans, construction-to-permanent financing requiring builder certifications and multiple disbursement schedules, reverse mortgage products subject to Home Equity Conversion Mortgage protocols, and shared appreciation or shared equity mortgage structures.

        3. Definitions

        **Automated Underwriting System (AUS)** refers to the proprietary algorithmic decision engines developed and maintained by Government-Sponsored Enterprises, specifically Fannie Mae's Desktop Underwriter and Freddie Mac's Loan Product Advisor, which ingest standardized loan application data conforming to the Uniform Residential Loan Application format and output a recommendation classification of "Approve/Eligible," "Approve/Ineligible," "Refer with Caution," "Refer," or "Decline/Ineligible" based on multivariate statistical models incorporating credit history, income stability, asset reserves, employment continuity, and property characteristics, along with a findings report specifying required documentation and verification procedures necessary to substantiate the recommendation.

        **Automated Valuation Model (AVM)** constitutes a computerized statistical technique that estimates the market value of a specific residential property by analyzing historical sales data, property characteristics from public records, tax assessor information, and proprietary hedonic pricing models incorporating location-specific appreciation trends, comparable property adjustments, and market condition indices, generating a point estimate of value along with a confidence score and forecast standard deviation that quantifies the statistical reliability of the valuation, typically expressed as a percentage variance range representing one standard deviation above and below the point estimate, employed within this procedure as a secondary validation mechanism to identify potential appraisal quality issues when the manual appraisal value deviates from the AVM estimate by more than ten percent in either direction.

        **Compensating Factors** denote specific positive financial characteristics or circumstances that partially offset elevated risk indicators in a borrower's credit profile, including but not limited to liquid asset reserves exceeding twelve months of proposed total housing payment (principal, interest, taxes, insurance, homeowners association fees, and mortgage insurance premiums), substantial equity contribution resulting in loan-to-value ratios at or below seventy percent, documented income stability evidenced by five or more years of continuous employment in the same field or with progressive increases in compensation, residual monthly income remaining after satisfaction of all debt obligations exceeding regional cost-of-living benchmarks by at least twenty percent, or demonstrable history of managing comparable or higher housing payment obligations without delinquency, which may be considered by underwriters when evaluating applications that marginally exceed policy thresholds for debt-to-income ratios or present credit history anomalies explainable by isolated economic events with subsequent rehabilitation.

        **Debt Service Coverage Ratio (DSCR)** represents the quotient obtained by dividing the borrower's gross monthly income by the sum of the proposed total housing payment and all recurring monthly debt obligations as reported on the credit report and verified through documentation, mathematically expressed as DSCR = Gross Monthly Income / (Proposed PITI + Monthly Debt Obligations), where a DSCR value less than one point zero indicates negative cash flow and values approaching one point zero indicate minimal margin for financial contingencies, with the institution's policy requiring DSCR values of at least one point one-five for standard approval consideration and values between one point zero and one point one-five triggering mandatory compensating factor analysis.

        **Debt-to-Income Ratio (DTI)** constitutes the percentage calculated by dividing the borrower's total monthly debt obligations by gross monthly income, expressed as DTI = [(Proposed PITI + Monthly Debt Obligations) / Gross Monthly Income] × 100, where the numerator includes the proposed housing payment consisting of principal, interest, property taxes, hazard insurance, mortgage insurance premiums if applicable, homeowners association dues if applicable, plus all installment debt obligations with more than ten months remaining, revolving credit obligations calculated using either the minimum payment shown on the credit report or one percent of the outstanding balance (whichever is greater), student loan payments whether in deferment or active repayment using the documented payment or one-half percent of the outstanding balance if no payment is shown, alimony and child support obligations, and any other court-ordered or contractually obligated recurring payments, with the denominator representing verified gross income before any tax withholdings or voluntary deductions, categorized as front-end DTI when considering only housing payment divided by income and back-end DTI when considering all obligations divided by income, with institutional policy establishing maximum acceptable thresholds of twenty-eight percent front-end and forty-three percent back-end for Qualified Mortgage compliance absent compensating factors.

        **FICO Score Tri-Merge Middle Score Methodology** describes the protocol for determining the representative credit score when multiple credit bureau reports are obtained, requiring the procurement of credit reports and associated FICO scores from all three national credit repositories (Equifax, Experian, and TransUnion), identification of the middle score by ranking the three scores in ascending order and selecting the median value (or the lower of two scores if only two bureaus provide scores, or the single score if only one bureau provides a score), and in cases involving co-borrowers or co-applicants, first determining each individual's representative middle score through the process described above, then selecting the lower of the two representative middle scores as the qualifying score for credit policy evaluation, interest rate pricing, and mortgage insurance premium calculation, with documented rationale required whenever a credit report is unavailable from one or more bureaus due to insufficient credit history or credit file freeze.

        **Loan-to-Value Ratio (LTV)** signifies the percentage calculated by dividing the loan amount by the lesser of the purchase price or appraised value of the subject property, expressed as LTV = [Loan Amount / MIN(Purchase Price, Appraised Value)] × 100, serving as the primary indicator of collateral risk exposure whereby higher LTV ratios correspond to diminished equity cushions and elevated loss severity in default scenarios, with institutional policy establishing tiered risk categories of LTV ≤ 80% (low risk, no mortgage insurance required), LTV 80.01% to 85% (moderate risk, mortgage insurance required), LTV 85.01% to 90% (elevated risk, mortgage insurance required with potential pricing adjustments), LTV 90.01% to 95% (high risk, mortgage insurance required with mandatory compensating factors), and LTV > 95% (generally prohibited except for portfolio retention products with explicit executive approval), and further distinguishing Combined Loan-to-Value Ratio (CLTV) when subordinate financing exists, calculated as CLTV = [(First Mortgage Amount + Subordinate Lien Amount) / MIN(Purchase Price, Appraised Value)] × 100.

        **Probability of Default (PD) Score** represents the institution's proprietary quantitative assessment of the statistical likelihood that a borrower will experience a payment delinquency of ninety days or greater within thirty-six months following loan origination, expressed as a percentage between zero and one hundred, derived from a logistic regression model that incorporates as independent variables the borrower's representative FICO score, debt-to-income ratio, loan-to-value ratio, liquid asset reserves measured in months of housing payment, employment tenure measured in years, property type categorization, occupancy status, loan purpose, and presence or absence of derogatory credit marks within the preceding twenty-four months, with model coefficients calibrated through analysis of the institution's historical loan performance data and validated against out-of-sample test datasets to ensure predictive accuracy, and with resultant PD scores mapped to Credit Score Grades using predetermined thresholds of PD ≤ 2.0% corresponding to Grade A, PD 2.01% to 4.0% corresponding to Grade B, PD 4.01% to 7.0% corresponding to Grade C, and PD > 7.0% corresponding to Grade D or automatic referral to special underwriting committee review.

        **Uniform Residential Appraisal Report (URAR)** designates the Fannie Mae Form 1004 / Freddie Mac Form 70 standardized appraisal reporting format that licensed or certified real estate appraisers must utilize when conducting full interior and exterior inspections of one-unit residential properties, requiring the appraiser to provide detailed property descriptions including gross living area square footage, room counts, condition ratings, quality of construction classifications, functional utility assessments, identification and analysis of at least three closed sales of comparable properties that occurred within twelve months preceding the appraisal effective date and located within one mile of the subject property (or as close as available market data permits), adjustment grids quantifying differences between the subject and comparable properties across dimensions including sale date, location, site characteristics, view, design and appeal, quality of construction, age, condition, gross living area, basement characteristics, functional utility, heating and cooling systems, garage or carport spaces, and porches or patios, with individual line-item adjustments generally not exceeding ten percent of the comparable's sale price and cumulative gross adjustments not exceeding twenty-five percent absent compelling market conditions justification, reconciliation of adjusted comparable values into a final opinion of market value with explanation of weighting factors, and certification that the appraisal was conducted in conformity with the Uniform Standards of Professional Appraisal Practice and without any direct or indirect pressure from parties having a financial interest in the transaction.

        4. Input

        The initiation of this Standard Operating Procedure requires the completed submission of a Uniform Residential Loan Application capturing the applicant's identifying information including applicant_name, co_applicant_name if applicable, applicant_ssn and co_applicant_ssn masked for privacy protection in accordance with the Gramm-Leach-Bliley Act, applicant_dob for age verification and regulatory reporting, and applicant_address representing current residential occupancy, along with property-specific information including property_address identifying the subject collateral, property_type classification among Single Family, Condo, Townhouse, or Multi-Family up to four units, property_occupancy designation as Primary Residence, Investment Property, or Second Home which directly impacts risk assessment and pricing, estimated_property_value representing the applicant's good-faith estimate of market value, requested_loan_amount denominated in United States Dollars, down_payment_amount representing the applicant's equity contribution, loan_purpose classification as Purchase, Refinance, or Cash-Out Refinance, and loan_term_months typically selecting between three hundred sixty months (thirty-year fixed), one hundred eighty months (fifteen-year fixed), or other alternative term structures offered by the institution. Additionally, the procedure requires the applicant's employment and income declarations including employment_status categorized as Full-Time, Self-Employed, Part-Time, or Retired, employer_name and employment_years establishing employment stability, gross_monthly_income representing base salary or wages before taxes and deductions, additional_income_monthly capturing bonus compensation, commission income, rental property income, investment income, alimony or child support received, or other recurring income sources subject to verification and continuance probability assessment, and asset information including liquid_assets representing checking accounts, savings accounts, money market accounts, and brokerage accounts readily convertible to cash, and retirement_accounts representing 401(k), IRA, or other tax-deferred retirement savings which may be considered with vesting and penalty discounts. The procedure further necessitates the applicant's execution of credit_consent_signed documented with credit_consent_timestamp authorizing the institution to obtain consumer credit reports from Equifax, Experian, and TransUnion credit repositories as well as ongoing monitoring for adverse credit events during the application pendency period, and requires the uploading of supporting documentation evidenced by Boolean confirmation flags including pay_stubs_uploaded reflecting the most recent thirty days of pay history, tax_returns_uploaded covering the most recent two calendar years with all schedules for self-employed applicants or those declaring rental income, w2_forms_uploaded for the corresponding two-year period, and bank_statements_uploaded covering the most recent two consecutive months for all accounts cited in the asset declaration, with all uploaded documents subject to image quality standards ensuring text legibility and metadata preservation for audit trail purposes. The assignment of loan_officer_id, processor_id, and ultimately underwriter_id establishes accountability and workload distribution within the Loan Origination System, while the system-generated application_id and application_timestamp create the unique identifier and temporal reference point for calculating turnaround_time_days and establishing regulatory disclosure timing requirements under the Truth in Lending Act and Real Estate Settlement Procedures Act integrated disclosure framework.

        5. Main Procedure

        5.1 Mortgage Application Intake and Pre-qualification Phase

        5.1.1 Upon receipt of a completed loan application package meeting the input requirements specified in Section 4 and assigned a unique application_id with system-recorded application_timestamp, the designated Loan Officer identified by loan_officer_id shall conduct comprehensive Customer Identification and Know Your Customer verification procedures by examining government-issued photographic identification documents including driver's licenses, state identification cards, or passports for the applicant and any co_applicant, verifying that the name on the identification matches the applicant_name field, that the date of birth matches the applicant_dob field, and that the address matches or can be reconciled with the applicant_address field, with any discrepancies requiring documented explanation such as recent marriage resulting in name change supported by marriage certificate, recent relocation supported by utility bills or lease agreements, or other life events, and the Loan Officer must further execute the institution's Customer Identification Program protocols mandated by Section 326 of the USA PATRIOT Act by searching the applicant and co-applicant names and dates of birth against the Office of Foreign Assets Control Specially Designated Nationals list and other government-provided watch lists, with any positive matches requiring immediate escalation to the Bank Secrecy Act compliance officer and suspension of application processing pending clearance, while simultaneously verifying the authenticity of the Social Security Numbers provided as applicant_ssn and co_applicant_ssn through pattern validation checking for known invalid ranges, sequential or repetitive digits indicative of fabrication, or numbers issued after the applicant's date of birth, with marginal cases requiring submission to the Social Security Administration's Consent Based Social Security Number Verification Service for definitive validation.

        5.1.2 Following successful identity verification, the Loan Officer shall proceed to collect and validate the applicant's employment, income, and asset declarations by reviewing the self-reported employment_status designation and, for applicants reporting Full-Time employment status, recording the employer_name and employment_years as stated by the applicant while simultaneously examining the uploaded pay_stubs_uploaded documentation to verify consistency between the stated employer name and the employer name printed on the pay stubs, confirming that the employment_years assertion is plausible based on the pay period dates visible on the pay stubs, and calculating a preliminary gross_monthly_income figure by analyzing the pay frequency indicated on the pay stubs (weekly pay periods multiplied by 4.333 weeks per month, bi-weekly pay periods multiplied by 2.167 pay periods per month, semi-monthly pay periods multiplied by 2.0, or monthly pay periods multiplied by 1.0) and the gross pay per period shown on the most recent pay stub, with particular attention to distinguishing base salary or wages from overtime, bonus, or commission components that require separate analysis for income stability and continuance probability. For applicants reporting Self-Employed employment status, the Loan Officer must confirm the presence of complete tax_returns_uploaded documentation including IRS Form 1040 individual tax returns with all schedules for the two most recent calendar years, Schedule C (Profit or Loss from Business) for sole proprietorships, Schedule E (Supplemental Income and Loss) for rental properties or partnerships, Form 1065 (U.S. Return of Partnership Income) with K-1 distributions for partnership interests, or Form 1120S (U.S. Income Tax Return for an S Corporation) with K-1 distributions for S-corporation ownership, and the Loan Officer must perform preliminary income averaging by extracting the adjusted gross income from line 11 of Form 1040 for each year, adding back non-cash deductions including depreciation and amortization expenses that did not represent actual cash outflows, subtracting non-recurring income items including one-time capital gains or casualty loss reimbursements, calculating the arithmetic mean of the two-year adjusted figures to derive a preliminary gross_monthly_income equivalent calculated as the two-year average divided by twelve months, and flagging for underwriter attention any year-over-year income decline exceeding ten percent which may indicate deteriorating business conditions requiring additional documentation and explanation.

        5.1.3 Concurrent with income validation, the Loan Officer shall verify the applicant's declared liquid_assets and retirement_accounts by examining the uploaded bank_statements_uploaded covering two consecutive months and confirming that the account balances shown as of the most recent statement date aggregate to a figure equal to or greater than the declared liquid_assets amount, with specific attention to identifying large deposits defined as any single deposit exceeding fifty percent of the applicant's stated monthly income or any aggregate deposits within a single statement period exceeding the stated monthly income, which require sourcing documentation demonstrating the origin of funds such as asset transfer statements, gift letters accompanied by donor bank statements showing the withdrawal of gifted funds, or explanation letters for proceeds from the sale of personal property, and the Loan Officer must further assess whether the liquid_assets remaining after accounting for the down_payment_amount and estimated closing costs of approximately two to four percent of the loan amount constitute adequate reserves calculated in terms of months of proposed total housing payment, with institutional policy requiring minimum reserves of two months for Primary Residence properties with LTV ratios at or below eighty percent, four months for Primary Residence properties with LTV ratios exceeding eighty percent, six months for Second Home properties regardless of LTV, and twelve months for Investment Property regardless of LTV, and any application showing insufficient reserves must be annotated for underwriter compensating factor analysis before proceeding to credit authorization.

        5.1.4 Upon completion of income and asset validation, the Loan Officer shall obtain and document the applicant's authorization for credit reporting by presenting the consumer disclosure and consent form that satisfies the Fair Credit Reporting Act's permissible purpose requirements under 15 USC 1681b, ensuring that the applicant provides either handwritten signature with date or electronic signature captured through E-SIGN Act compliant methodology that records the applicant's intent to sign, identity authentication through knowledge-based authentication questions or similar verification, and consent timestamp recorded as credit_consent_timestamp, and setting the credit_consent_signed Boolean field to true in the Loan Origination System, after which the Loan Officer shall immediately initiate tri-bureau credit report acquisition by transmitting the applicant's name, current address, date of birth, and Social Security Number to the institution's credit bureau interface which maintains Application Programming Interface connections to Equifax, Experian, and TransUnion repositories, requesting in-file credit reports with corresponding FICO scores calculated using the FICO Score 2 (Experian/Fair Isaac Risk Model v2), FICO Score 5 (Equifax Beacon 5.0), and FICO Score 4 (TransUnion FICO Risk Score, Classic 04) models designated by Fannie Mae and Freddie Mac for mortgage lending purposes, receiving within seconds the electronically delivered credit reports with system-recorded credit_pull_timestamp, and immediately populating the fields fico_score_equifax, fico_score_experian, and fico_score_transunion with the respective scores, or recording null values if a particular bureau returns an insufficient credit file error indicating the applicant lacks sufficient credit history with that repository.

        5.1.5 Upon receipt of the tri-bureau credit report, the Loan Officer or automated system logic shall execute the FICO Score Tri-Merge Middle Score Methodology as defined in Section 3 by first verifying that all three scores from fico_score_equifax, fico_score_experian, and fico_score_transunion are non-null numeric values, then ranking these three scores in ascending numerical order to identify the median value which becomes the representative credit score for the individual applicant, or if only two scores are available due to one bureau returning insufficient file, selecting the lower of the two available scores as the representative score, or if only one score is available, designating that single score as representative while documenting the reason for missing bureau data, and in cases where a co_applicant_name exists requiring the same tri-bureau pull and middle score determination for the co-applicant, after which the system selects the lower of the applicant's representative middle score and the co-applicant's representative middle score as the qualifying score that will be used for all subsequent credit policy evaluation, pricing determination, and mortgage insurance premium calculation. Concurrently, the Loan Officer shall analyze the credit report narrative sections to extract and populate derogatory_marks_count by counting instances of collections accounts regardless of amount, charge-offs, public records including tax liens or civil judgments, foreclosures, deeds in lieu of foreclosure, short sales, loan modifications, or accounts with payment history showing thirty-day late payments within the preceding twelve months, sixty-day late payments within the preceding twenty-four months, or ninety-day or greater late payments within the preceding thirty-six months, and shall set the bankruptcy_history Boolean field to true if the credit report indicates a Chapter 7 bankruptcy discharge, Chapter 13 bankruptcy discharge, or currently active Chapter 13 repayment plan, while recording false if no bankruptcy history appears. The Loan Officer must further calculate total_monthly_debt_obligations by summing all recurring monthly payment obligations shown on the credit report including installment loans with more than ten scheduled payments remaining (car loans, student loans, personal loans) using the monthly payment amount shown, revolving credit accounts (credit cards, home equity lines of credit) using the greater of the minimum payment shown or one percent of the outstanding balance, student loans in deferment or forbearance calculated as either the documented payment amount if available or one-half percent of the outstanding balance if no payment is shown per Federal Housing Finance Agency guidance, and any alimony, child support, or separate maintenance payment obligations discoverable through credit report inquiry or applicant disclosure.

        5.1.6 With the representative credit score determined and monthly debt obligations quantified, the system shall execute automated pre-qualification screening by comparing the representative credit score against the institution's minimum acceptable threshold which is established at a FICO score of 620 for conforming conventional mortgage products eligible for Government-Sponsored Enterprise purchase, with scores below this threshold generating a preliminary classification of prequalification_decision equal to "Not Qualified" and requiring referral to alternative product structures or correspondent lending channels if available, while scores at or above 620 permit continuation of the pre-qualification analysis. The system then calculates a preliminary debt-to-income ratio using the formula preliminary_DTI = [(total_monthly_debt_obligations + estimated_housing_payment) / gross_monthly_income] × 100, where estimated_housing_payment is derived by applying an assumed interest rate based on the current rate environment and the applicant's credit score tier (for this calculation, using a proxy rate of the current 30-year Freddie Mac Primary Mortgage Market Survey rate plus 0.250% for credit scores 740 and above, plus 0.500% for scores 700-739, plus 0.750% for scores 660-699, plus 1.000% for scores 620-659) to the requested_loan_amount using standard mortgage payment calculation formula P = L[c(1 + c)^n]/[(1 + c)^n - 1] where P is monthly principal and interest payment, L is loan amount, c is monthly interest rate (annual rate divided by 12), and n is number of months from loan_term_months, then adding estimated property taxes calculated as 1.2% of estimated_property_value divided by 12 months (national average), estimated hazard insurance calculated as 0.5% of estimated_property_value divided by 12 months, estimated mortgage insurance premium if LTV exceeds 80% calculated using standard mortgage insurance rate tables (approximately 0.5% annually for 85% LTV, 0.8% annually for 90% LTV, 1.0% annually for 95% LTV, divided by 12 for monthly), and any estimated homeowners association dues disclosed by the applicant. If the calculated preliminary_DTI exceeds 50%, the system generates prequalification_decision equal to "Not Qualified" with annotation that debt ratios exceed maximum permissible thresholds; if preliminary_DTI falls between 43.01% and 50%, the system generates "Conditionally Qualified" requiring mandatory compensating factors and manual underwriting; if preliminary_DTI is at or below 43%, the system generates "Qualified" and proceeds to calculate maximum loan capacity.

        5.1.7 For applications receiving "Qualified" or "Conditionally Qualified" designations, the system shall calculate max_loan_amount_prequalified by solving the debt-to-income formula for maximum loan amount that results in a 43% back-end DTI ratio (or 36% for applicants seeking "prime" loan terms with optimal pricing), using the iterative equation max_loan = [(0.43 × gross_monthly_income) - total_monthly_debt_obligations - estimated_non_PI_housing_costs] / [c(1 + c)^n / ((1 + c)^n - 1)] where c is the monthly interest rate corresponding to the estimated_interest_rate_prequalification determined by the applicant's credit score band and the requested property_type and property_occupancy (with rate premiums of +0.250% for Condo, +0.375% for Townhouse, +0.500% for Multi-Family two-unit, +0.750% for Multi-Family three-to-four unit, and additional premiums of +0.375% for Second Home, +0.750% to +2.000% for Investment Property depending on LTV and credit score), and estimated_non_PI_housing_costs represents the monthly property taxes, insurance, mortgage insurance if applicable, and HOA fees calculated as described in the previous substep, with the resulting max_loan_amount_prequalified rounded down to the nearest one thousand dollars to provide conservative guidance to the applicant regarding affordable property search parameters. The system populates estimated_interest_rate_prequalification by retrieving the current rate sheet from the institution's pricing engine which maintains a multi-dimensional matrix indexed by credit score bands (780+, 760-779, 740-759, 720-739, 700-719, 680-699, 660-679, 640-659, 620-639), LTV ratio bands (≤60%, 60.01-70%, 70.01-75%, 75.01-80%, 80.01-85%, 85.01-90%, 90.01-95%, >95%), property type categories, occupancy categories, and loan purpose categories, with each cell in the matrix containing a base interest rate plus pricing adjustments expressed in both rate terms and discount point terms for borrower selection. The Loan Officer then generates and delivers to the applicant a Pre-qualification Letter document suitable for presentation to real estate sellers or agents, clearly stating that the letter represents a preliminary assessment based on applicant-provided information subject to verification, property identification, appraisal completion, and final underwriting approval, and is not a commitment to lend, while displaying the max_loan_amount_prequalified figure and estimated_interest_rate_prequalification with appropriate disclaimers regarding rate lock periods and market volatility.

        5.2 Credit Underwriting Phase

        5.2.1 Following the applicant's identification of a specific property and execution of a purchase contract for Purchase transactions or declaration of intent to refinance an existing mortgage for Refinance transactions, with the property_address now definitively established in the Loan Origination System and property_type definitively classified through Multiple Listing Service data or county recorder documents, the loan file progresses from application_status of "Pre-qualification" to "Underwriting" and is assigned to a Credit Underwriter identified by underwriter_id based on workload balancing algorithms, geographic specialization if the institution operates in multiple markets, or loan complexity routing that assigns self-employed borrower files, high balance loan amount files exceeding one million dollars, or files involving non-warrantable condominium projects to senior underwriters with specialized expertise. The assigned Underwriter initiates the Credit Underwriting Phase by conducting comprehensive document verification procedures beginning with employment and income validation, which for Full-Time employed applicants requires the Underwriter to contact the employer identified as employer_name by telephone using contact information independently verified through directory services rather than applicant-provided numbers, requesting verbal or written Verification of Employment that confirms the applicant's current employment status, job title, hire date permitting calculation of employment_years, and current base salary or hourly wage permitting calculation of verified_gross_monthly_income, with the verification call documented in the loan file including date, time, name and title of employer representative contacted, and confirmation that the employment is expected to continue for the foreseeable future absent specific knowledge of impending layoff or termination.

        5.2.2 For self-employed applicants where employment_status equals "Self-Employed," the Underwriter must perform detailed analysis of the uploaded tax_returns_uploaded documentation by examining the IRS Form 1040 individual returns for the two most recent completed tax years, extracting the Adjusted Gross Income from Line 11 of each year's return, then drilling into Schedule C for sole proprietorship income by taking the Net Profit or Loss from Line 31 and adding back to income any depreciation expense from Line 13, depletion claimed on Line 12, business use of home expenses from Line 30 that reduced taxable income but do not represent actual cash outflow, and one-half of self-employment tax from Line 27 that represents a tax rather than business expense, while subtracting any one-time or non-recurring income such as proceeds from sale of business assets or insurance proceeds from casualty losses, then performing the identical analysis for the prior year's tax return, calculating the average of the two years' adjusted income figures, dividing by twelve to produce a monthly figure, and recording this as verified_gross_monthly_income with mandatory documentation in the underwriting analysis describing the calculation methodology and year-over-year income trend. If year-over-year income has declined by more than ten percent, the Underwriter must document compelling justification for considering the income stable and likely to continue, such as the decline resulting from one-time business equipment purchases that increased depreciation deductions but will not recur, or the decline resulting from industry-wide temporary disruptions with documented recovery trajectory evidenced by year-to-date profit and loss statements showing monthly income returning to or exceeding historical averages. For applicants claiming additional_income_monthly from sources including bonus compensation, commission income, overtime pay, rental property income, investment dividends, alimony received, or other non-base income, the Underwriter must verify each income type through examination of tax returns showing consistent receipt over the most recent two-year period, with bonus and commission income subject to two-year averaging and requiring employer verification that such variable compensation is expected to continue, overtime income requiring employer confirmation that overtime availability is not temporary or diminishing, rental property income requiring review of executed lease agreements and verification that the lease extends beyond the mortgage closing date or evidence of tenant occupancy stability, and investment income requiring brokerage statements showing consistent dividend or interest receipts from established portfolios rather than one-time capital gains distributions.

        5.2.3 Concurrent with income verification, the Underwriter shall verify the applicant's declared liquid_assets by examining the bank_statements_uploaded documentation covering two consecutive months for all checking accounts, savings accounts, money market accounts, and taxable brokerage accounts, confirming that the ending balance shown on the most recent statement month equals or exceeds the declared liquid_assets figure, and investigating any deposits during the two-month statement period that individually exceed fifty percent of the verified_gross_monthly_income or that cumulatively exceed one hundred percent of monthly income across the two-month period, requiring the applicant to provide source documentation for such large or unusual deposits including copies of checks showing drawer identity for deposit verification, account transfer statements demonstrating movement from one borrower-owned account to another, gift letters conforming to Government-Sponsored Enterprise requirements that include donor name, donor relationship to applicant, gift amount, property address, explicit statement that no repayment is expected, and donor signature, accompanied by donor bank statements showing withdrawal of the gifted amount and borrower bank statements showing receipt of the identical amount, or other explanations documented through affidavits or supporting evidence that the deposits represent legitimate sources not requiring repayment, with particular scrutiny applied to rapid asset accumulation that could indicate undisclosed borrowed funds or "cash back at close" schemes prohibited under regulatory anti-fraud provisions. For retirement_accounts including 401(k) plans, Individual Retirement Accounts, SEP-IRAs, or other tax-deferred savings vehicles, the Underwriter may consider these accounts as compensating factors for reserve requirements but must apply a discount factor of twenty-five percent to the account balance to reflect taxes and early withdrawal penalties that would be incurred if the funds were liquidated, unless the applicant has attained age 59.5 at which point a reduced discount of ten percent may be applied to reflect income tax liability only without early withdrawal penalty. The Underwriter calculates total available liquid assets net of required funds for down payment by taking verified_gross_monthly_income, subtracting the down_payment_amount, subtracting estimated closing costs calculated as the sum of lender fees typically ranging from 0.5% to 1.0% of loan amount, third-party fees including appraisal (typically $500-$700), title insurance and settlement fees (typically 0.5%-1.0% of purchase price), prepaid items including initial escrow deposits for property taxes and insurance (typically 3-6 months of estimated annual costs), and recording fees (typically $100-$300), with the remainder representing reserves measured in months of total housing payment calculated by dividing remaining liquid assets by the monthly housing payment amount including principal, interest, property taxes, insurance, HOA fees, and mortgage insurance.

        5.2.4 Having verified income and assets, the Underwriter proceeds to detailed credit report analysis by reviewing the full narrative tri-bureau credit report obtained during pre-qualification and now supplemented with additional scrutiny of payment history patterns, specifically examining each tradeline shown on the report for the most recent twelve-month payment history to identify any thirty-day late payments (which add 1 point to derogatory_marks_count), sixty-day late payments (which add 2 points), or ninety-day or greater late payments (which add 3 points), and identifying any collections accounts by reviewing the public records section for unpaid collections or accounts assigned to collection agencies, determining whether medical collections under $500 may be disregarded pursuant to Government-Sponsored Enterprise guidelines effective 2016, assessing whether non-medical collections exceeding $500 require payment in full prior to closing or may be left outstanding if the cumulative balance of all unpaid collections is less than $1,000 and credit score suggests overall creditworthiness, and reviewing the inquiries section to identify multiple recent credit inquiries within the preceding ninety days which may indicate credit-seeking behavior and potential undisclosed new debt obligations requiring explanation letters from the applicant. The Underwriter must specifically verify that bankruptcy_history equals false or, if true, that sufficient time has elapsed since the bankruptcy discharge or dismissal to meet institutional seasoning requirements of four years from Chapter 7 discharge date or two years from Chapter 13 discharge date with satisfactory payment history on all re-established credit, and that foreclosure history is absent or, if present, that seven years have elapsed since the foreclosure completion date, or three years if the foreclosure resulted from extenuating circumstances such as documented job loss, death of primary wage earner, or serious illness, provided the applicant has re-established acceptable credit since the event.

        5.2.5 The Underwriter shall calculate the precise debt_to_income_ratio using verified rather than stated information by first computing the exact monthly housing payment for the proposed loan using the formula PITI = [L × c(1 + c)^n / ((1 + c)^n - 1)] + [Property_Tax_Annual / 12] + [Insurance_Annual / 12] + [MI_Annual / 12] + HOA_monthly, where L equals the approved loan amount (either requested_loan_amount if within policy limits or adjusted amount based on Underwriter determination), c equals the monthly interest rate derived from the rate sheet price corresponding to the applicant's credit score tier and LTV band, n equals loan_term_months, Property_Tax_Annual is obtained from the tax assessor's record for the specific property_address showing the most recent annual tax bill, Insurance_Annual is quoted by the applicant's selected hazard insurance carrier in an amount sufficient to cover the replacement cost of improvements, MI_Annual represents mortgage insurance premium calculated from the institution's mortgage insurance rate card based on specific LTV ratio and credit score with rates ranging from 0.30% of loan amount annually for 85% LTV with 760 FICO to 1.50% of loan amount annually for 95% LTV with 680 FICO, and HOA_monthly is extracted from the homeowners association budget or condominium documents if applicable. The Underwriter then adds to this housing payment amount the previously calculated total_monthly_debt_obligations from the credit report, divides this sum by the verified_gross_monthly_income, and multiplies by 100 to express as percentage, recording this value as debt_to_income_ratio with precision to two decimal places. Institutional policy establishes that debt_to_income_ratio must not exceed 43.00% for loans intended to meet Qualified Mortgage standards under 12 CFR 1026.43, and must not exceed 36.00% for portfolio loans designated for balance sheet retention, with ratios between 43.01% and 50.00% requiring referral to the Chief Credit Officer for exception approval supported by comprehensive compensating factor documentation. Simultaneously, the Underwriter calculates debt_service_coverage_ratio using the formula DSCR = verified_gross_monthly_income / (PITI + total_monthly_debt_obligations), recording the result with precision to two decimal places, with institutional policy requiring DSCR ≥ 1.15 for standard approval, DSCR between 1.00 and 1.14 requiring compensating factors analysis, and DSCR < 1.00 generally resulting in decline absent extraordinary circumstances such as applicant possessing liquid assets exceeding 24 months of total housing payment.

        5.2.6 The Underwriter shall execute the institution's proprietary Probability of Default scoring model by entering the verified data elements into the logistic regression equation PD = 1 / (1 + e^-z), where z = β₀ + β₁(FICO) + β₂(DTI) + β₃(LTV) + β₄(Reserves) + β₅(Employment_Years) + β₆(Property_Type_Indicator) + β₇(Occupancy_Indicator) + β₈(Loan_Purpose_Indicator) + β₉(Derogatory_Marks) and the beta coefficients are institution-specific values calibrated through historical loan performance analysis (with representative values for illustrative purposes being β₀ = 8.5, β₁ = -0.012 per FICO point, β₂ = 0.08 per DTI percentage point, β₃ = 0.05 per LTV percentage point, β₄ = -0.15 per month of reserves, β₅ = -0.05 per year of employment, β₆ = 0.3 for Multi-Family property types and 0 otherwise, β₇ = 0.8 for Investment Property occupancy and 0 otherwise, β₈ = 0.4 for Cash-Out Refinance purpose and 0 otherwise, β₉ = 0.2 per derogatory mark count), and the input variables are the representative middle credit score from the tri-merge methodology, the calculated debt_to_income_ratio, the loan-to-value ratio calculated as (approved loan amount / appraised value) which will be refined upon appraisal completion but initially using estimated_property_value, months of reserves calculated in substep 5.2.3, employment_years, categorical indicators converted to dummy variables, and derogatory_marks_count from credit analysis. The model outputs a probability_of_default_score expressed as a percentage between 0 and 100, which the Underwriter then maps to credit_score_grade using the thresholds defined in Section 3 such that PD ≤ 2.0% yields Grade "A", PD between 2.01% and 4.0% yields Grade "B", PD between 4.01% and 7.0% yields Grade "C", and PD > 7.0% yields Grade "D" requiring escalation to the Credit Committee for special review. The Underwriter documents in the loan file a comprehensive narrative analysis justifying the credit_score_grade assignment and identifying any factors that elevated or mitigated the calculated probability of default.

        5.2.7 Based on the assigned credit_score_grade, the calculated loan_to_value_ratio (initially estimated but to be refined upon appraisal completion), the property_type classification, property_occupancy designation, and loan_purpose categorization, the Underwriter or automated pricing engine determines the final_risk_adjusted_interest_rate by consulting the institution's rate sheet matrix which establishes base interest rates indexed to the current 30-year Freddie Mac Primary Mortgage Market Survey rate or the institution's cost of funds benchmark, then applying additive pricing adjustments for risk factors including credit score adjustments of -0.500% for FICO ≥ 780, -0.250% for FICO 760-779, 0.000% base for FICO 740-759, +0.125% for FICO 720-739, +0.250% for FICO 700-719, +0.500% for FICO 680-699, +0.875% for FICO 660-679, +1.375% for FICO 640-659, +2.000% for FICO 620-639, LTV ratio adjustments of +0.000% for LTV ≤ 60%, +0.125% for LTV 60.01-70%, +0.250% for LTV 70.01-80%, +0.375% for LTV 80.01-85%, +0.625% for LTV 85.01-90%, +1.000% for LTV 90.01-95%, property type adjustments of +0.000% for Single Family, +0.250% for Condo, +0.375% for Townhouse, +0.500% for two-unit Multi-Family, +1.000% for three-to-four unit Multi-Family, occupancy adjustments of +0.000% for Primary Residence, +0.375% for Second Home, +0.750% for Investment Property single-family, +1.500% for Investment Property multi-family, loan purpose adjustments of +0.000% for Purchase, +0.125% for Rate-Term Refinance, +0.375% for Cash-Out Refinance with LTV ≤ 70%, +0.750% for Cash-Out Refinance with LTV > 70%, and debt-to-income adjustments of +0.000% for DTI ≤ 36%, +0.125% for DTI 36.01-40%, +0.250% for DTI 40.01-43%, +0.500% for DTI 43.01-45% if permitted by compensating factors, with all adjustments summed algebraically to produce the final_risk_adjusted_interest_rate expressed to three decimal places such as 6.875%.

        5.2.8 In cases where the applicant's debt_to_income_ratio exceeds 43.00% but remains below 50.00%, or where the calculated loan_to_value_ratio exceeds 90% requiring heightened scrutiny, or where derogatory_marks_count exceeds 2, or where employment_years is less than 2.0 indicating limited job stability, the Underwriter must conduct compensating factors analysis by systematically evaluating and documenting the presence or absence of offsetting positive characteristics, beginning with reserves calculation by determining whether liquid assets after down payment and closing costs exceed 12 months of total housing payment (conferring 3 compensating factor points), exceed 6 months but are less than 12 months (conferring 2 points), exceed 3 months but are less than 6 months (conferring 1 point), or are less than 3 months (conferring 0 points), continuing with housing payment history assessment by determining whether the applicant has successfully made housing payments (rent or mortgage) equal to or greater than the proposed new housing payment for at least 12 consecutive months without any 30-day late payments (conferring 2 points if affirmative), evaluating residual income by calculating monthly gross income minus all monthly debt obligations including the proposed housing payment and determining whether the residual exceeds 120% of the regional median income adjusted for family size per Department of Housing and Urban Development poverty guideline tables (conferring 2 points if affirmative), assessing loan-to-value position by determining whether the LTV ratio is at or below 70% representing substantial equity cushion (conferring 2 points), examining employment stability by determining whether employment_years exceeds 5.0 years in the same line of work even if not with the same employer (conferring 1 point), and reviewing credit depth by determining whether the applicant maintains three or more active revolving credit accounts with credit limits exceeding $5,000 each and utilization ratios below 30% on each account indicating sophisticated credit management (conferring 1 point). The Underwriter sums the total compensating factor points and records compensating_factors_present as true if the total equals or exceeds 5 points for DTI exceptions between 43% and 45%, or equals or exceeds 7 points for DTI exceptions between 45% and 50%, and documents in narrative form the specific compensating factors present with quantified evidence supporting each claimed factor, with the compensating factor analysis serving as mandatory support for any policy exception request submitted to the Chief Credit Officer or designated approval authority.

        5.2.9 Upon completion of income verification, asset verification, credit analysis, ratio calculations, probability of default scoring, credit grading, interest rate determination, and compensating factor evaluation when applicable, the Underwriter prepares a comprehensive Underwriting Summary Memorandum documenting the loan file analysis and preliminary credit decision, subject to satisfactory completion of property appraisal, stating whether the file is approved subject to conditions, approved with no conditions (achieving first_pass_yield status), or declined, and if approved, specifying the approved_loan_amount which may equal the requested_loan_amount if supported by verified income and asset calculations or may be reduced to a lower figure that achieves acceptable debt-to-income ratios, specifying the approved_interest_rate which equals the final_risk_adjusted_interest_rate calculated in substep 5.2.7, specifying the approved_loan_term_months which typically matches the requested term but may be modified if a shorter term improves debt ratios or payment capacity, and enumerating any conditions_count representing outstanding documentation or verification requirements that must be satisfied prior to loan closing, including typical conditions such as "Provide payoff statement for automobile loan account ending in 1234 showing exact payoff amount and per diem interest charge," "Provide satisfactory explanation letter regarding credit inquiry from XYZ Bank dated March 15, 2025," "Provide evidence of homeowner's insurance policy with coverage amount of at least $450,000 naming the institution as mortgagee," or "Provide verification that earnest money deposit of $10,000 has been transmitted to title company escrow account." If the Underwriter determines that the application must be declined, the decline_reason field is populated with specific rationale such as "Debt-to-income ratio of 51.2% exceeds maximum permissible threshold of 50.0% and insufficient compensating factors present," "Insufficient verified income to support requested loan amount," "Credit history shows undischarged Chapter 13 bankruptcy filed less than two years ago, failing to meet institutional seasoning requirements," "Employment verification unsuccessful; applicant no longer employed by stated employer," or other factual basis for the adverse decision, with the decline notice prepared in compliance with Equal Credit Opportunity Act requirements under Regulation B specifying the principal reasons for adverse action and the applicant's right to request disclosure of the specific reasons within 60 days.

        5.3 Property Valuation and Appraisal Phase

        5.3.1 Upon the Underwriter's preliminary approval subject to appraisal or upon the application achieving application_status of "Appraisal Pending," the Loan Processor identified by processor_id initiates the property valuation process by transmitting an appraisal order to either the institution's internal appraisal desk if such capacity exists or more commonly to an independent Appraisal Management Company selected through competitive procurement processes that ensure appraiser independence and compliance with Dodd-Frank Act appraiser independence requirements codified in 12 CFR 1026.42, providing the appraisal management company with the property_address, property_type classification derived from Multiple Listing Service data or public records, property_occupancy status based on applicant declaration verified through occupancy certification at closing, estimated_property_value from the purchase contract sale price or applicant estimate for refinance transactions, and specific appraisal form requirements including the need for a full Uniform Residential Appraisal Report (URAR) Fannie Mae Form 1004 / Freddie Mac Form 70 for single-family properties, a Fannie Mae Form 1073 / Freddie Mac Form 466 for condominium units requiring both unit appraisal and condominium project review, or a Fannie Mae Form 1025 / Freddie Mac Form 72 for small residential income properties (two-to-four units), with the appraisal order recorded in the Loan Origination System with appraisal_order_timestamp marking the date and time of order transmission and initiating the timeline expectation of appraisal completion within ten business days per standard service level agreements.

        5.3.2 The Appraisal Management Company assigns the appraisal engagement to a state-licensed or state-certified real estate appraiser holding credentials appropriate to the assignment complexity, recording the appraiser_id and appraisal_company name in the loan file, with state certification required for all transactions exceeding $400,000 in value or involving complex property characteristics, and state licensing sufficient for transactions below $400,000 of non-complex residential property, ensuring that the selected appraiser maintains geographic competency in the specific market area by having completed at least five appraisal assignments within the subject property's ZIP code or adjacent ZIP codes within the preceding twelve months and maintaining knowledge of neighborhood boundaries, market conditions, typical price ranges, and local value influences. The assigned appraiser contacts the applicant or the applicant's real estate agent to schedule the site_inspection_timestamp for a date and time convenient for property access, requiring that the inspection occur during daylight hours to facilitate exterior photography and that the property be vacant of occupants during the inspection to enable thorough interior examination, and on the scheduled date, the appraiser travels to the property_address to conduct a comprehensive interior and exterior inspection involving measurement of the gross living area using external perimeter measurements excluding garages, unfinished basements, and attic spaces unless finished to the same standard as the main living areas, recording of property_square_footage with accuracy to the nearest ten square feet, examination and photography of all interior rooms including kitchens, bathrooms, bedrooms, living areas, and finished basement or attic spaces if present, assessment of overall property_condition using the standardized rating scale of C1 (new construction or recently fully renovated with no deferred maintenance), C2 (property shows minor wear but is well-maintained with no obvious deferred maintenance), C3 (property shows normal wear and tear appropriate to its age with some minor deferred maintenance present such as worn flooring or outdated fixtures not affecting safety or soundness), C4 (property shows deferred maintenance with functional issues requiring near-term attention such as aging roof requiring replacement within two years, failing HVAC system, or significant cosmetic deterioration), C5 (property shows substantial deferred maintenance with health or safety issues requiring immediate repair such as damaged roof causing leaks, non-functional mechanical systems, or structural deficiencies), with institutional policy requiring properties rated C4 to be subject to mandatory repair completion prior to loan closing and properties rated C5 to be ineligible for financing until comprehensive rehabilitation is completed, and observation of property characteristics including architectural style, construction quality (economy, average, good, excellent), effective age accounting for updates and maintenance, functional utility assessment identifying any superadequacies such as excessive square footage for the neighborhood or functional obsolescence such as lack of bathroom or bedroom count typical for the market, and site characteristics including lot size, topography, view amenities or detriments, landscaping quality, and external factors such as proximity to commercial zones, busy streets, power transmission lines, or other locational influences on value.

        5.3.3 Following site inspection completion, the appraiser conducts comparable sales analysis by searching the Multiple Listing Service database, public records, and proprietary data sources to identify recently sold properties that are similar to the subject property in terms of location, size, age, condition, and quality, selecting a minimum of three closed sales designated as comparable_properties_count equal to or exceeding three, with ideal comparables meeting the criteria of having closed within six months of the appraisal effective date (though twelve months is acceptable if market activity is limited), located within one mile of the subject property or as close as available data permits in rural areas, representing arm's-length transactions between unrelated parties without unusual financing concessions or seller-paid closing costs exceeding typical market levels, and exhibiting similar gross living area square footage within 20% of the subject property's property_square_footage, similar lot size within reasonable variance, identical or highly similar property_type classification such as Single Family detached for subject and comparables, similar age plus or minus fifteen years, and similar overall condition. For each selected comparable sale, the appraiser documents the property address, sale date, sale price, gross living area, lot size, age, condition rating, number of bedrooms and bathrooms, basement characteristics, garage or carport spaces, and other relevant features, then constructs an adjustment grid that quantifies the dollar value differences between each comparable and the subject property across each characteristic dimension, applying adjustments for sale date using local market appreciation or depreciation trends measured in percentage per month and multiplied by the comparable's sale price, applying location adjustments to account for superior or inferior neighborhoods typically ranging from $5,000 to $25,000 depending on market price levels and the magnitude of locational difference, applying site characteristic adjustments for lot size differences calculated on a per-square-foot basis or as lump sum adjustments for superior or inferior views, applying gross living area adjustments calculated on a per-square-foot basis derived from matched-pair analysis of sales differing primarily in size (typically ranging from $50 to $150 per square foot depending on the overall quality and market), applying room count adjustments for differences in bedroom count (typically $3,000 to $8,000 per bedroom) or bathroom count (typically $2,000 to $6,000 per bathroom), applying basement adjustments distinguishing between unfinished basement space valued at approximately 20% to 30% of above-grade square footage cost and finished basement space valued at 40% to 60% of above-grade square footage cost, applying garage adjustments (typically $5,000 to $12,000 per car space depending on market), and applying quality and condition adjustments when one property exhibits superior or inferior construction quality or maintenance condition compared to another, with individual adjustments typically not exceeding 10% of the comparable's sale price and cumulative gross adjustments (sum of absolute values of all adjustments) typically not exceeding 25% of the comparable's sale price, unless compelling explanation is provided for unusually large adjustments based on specific market conditions or unique property characteristics.

        5.3.4 After applying all appropriate adjustments to each comparable sale's transaction price, the appraiser derives an adjusted sale price for each comparable that theoretically represents what that comparable would have sold for if it had possessed identical characteristics to the subject property, then analyzes the range and clustering of the three or more adjusted comparable values to reconcile a final opinion of market value for the subject property recorded as final_appraised_value expressed in United States Dollars rounded to the nearest one thousand dollars, with reconciliation methodology considering the relative similarity of each comparable to the subject (giving more weight to comparables requiring fewer and smaller adjustments), the relative recency of each comparable sale (giving more weight to more recent sales in rapidly appreciating or depreciating markets), and the overall reliability of the data sources and verification of sale details, and documenting in the appraisal report narrative a clear explanation of the weighting factors that led to the final value conclusion. The appraiser completes the Uniform Residential Appraisal Report form including all required sections such as the neighborhood description outlining boundaries, predominant occupancy percentages, demand/supply trends, marketing time expectations, property value trends (increasing, stable, declining), the site description detailing zoning compliance and highest and best use analysis confirming that residential use represents the optimal utilization of the property, the improvements description providing architectural style and detailed feature lists, the sales comparison approach grid with all comparables and adjustments, the cost approach if required (typically for new construction or unique properties) involving estimation of land value, reproduction or replacement cost of improvements, less physical depreciation, functional obsolescence, and external obsolescence, the income approach if applicable (required for two-to-four unit properties) involving analysis of market rent potential and gross rent multipliers or direct capitalization rates, reconciliation of the value approaches to arrive at final opinion of market value, and certifications including the statement that the appraiser has no present or prospective interest in the property or parties to the transaction, that the fee was not contingent upon a predetermined value, that the appraisal was conducted in conformity with the Uniform Standards of Professional Appraisal Practice, and other required representations under the Dodd-Frank Act appraisal independence rules.

        5.3.5 Upon receipt of the completed appraisal report from the Appraisal Management Company, the Loan Processor uploads the report into the Loan Origination System and triggers an automated quality control review conducted by the institution's Collateral Risk Management desk, which executes a series of validation procedures beginning with verification that the appraiser's license or certification is active and in good standing through query of state regulatory databases, confirmation that the appraiser has not been sanctioned or excluded from GSE appraiser panels, review of the appraisal for completeness ensuring all sections of the URAR form are fully populated without fields left blank that require narrative or numerical entries, validation that photographs include front, rear, and street scene views of the subject property as well as exterior photographs of all comparable properties used in the sales comparison approach, checking for mathematical accuracy by independently recalculating all adjustment grid arithmetic to ensure adjusted comparable values are correctly computed, and executing the institution's Automated Valuation Model secondary check by electronically submitting the property_address to the AVM platform (such as CoreLogic HouseCanopy, Freddie Mac Home Value Explorer, or proprietary models) which returns an avm_estimated_value representing the model's statistically derived market value estimate along with a confidence score and forecast standard deviation. The Collateral Risk Management analyst calculates the appraisal_avm_variance_percent using the formula variance = [(final_appraised_value - avm_estimated_value) / avm_estimated_value] × 100, expressing the result as a signed percentage where positive values indicate the appraisal exceeds the AVM estimate and negative values indicate the appraisal falls below the AVM estimate. If the absolute value of appraisal_avm_variance_percent exceeds 10.00%, the analyst sets second_review_triggered to true and initiates a secondary field review appraisal whereby a different independent appraiser is engaged to re-inspect the property and provide an independent assessment of value, with the two appraisal values then reconciled through one of several protocols: if the field review supports the original appraisal within five percent variance, the original appraisal is accepted; if the field review differs by more than five percent, a third "tie-breaker" desk review is conducted by a supervisory appraiser who analyzes both reports and renders a final opinion regarding the most credible value estimate, which becomes the final_appraised_value used for underwriting purposes. If appraisal_avm_variance_percent is within the acceptable plus-or-minus 10.00% tolerance range, second_review_triggered remains false and the appraisal is accepted without additional review, permitting the process to advance to final underwriting decision.

        5.3.6 With final_appraised_value definitively established, the Underwriter recalculates the loan_to_value_ratio using the formula LTV = (requested_loan_amount / lesser_of_value_or_price) × 100, where lesser_of_value_or_price equals the lower of final_appraised_value or the property purchase price stated in the executed purchase contract for Purchase transactions, or equals final_appraised_value for Refinance transactions where no purchase price exists, expressing the result as a percentage with precision to two decimal places. For example, if requested_loan_amount equals $380,000, the purchase contract price is $450,000, and final_appraised_value is $465,000, the lesser_of_value_or_price is $450,000, yielding LTV = ($380,000 / $450,000) × 100 = 84.44%. If the calculated loan_to_value_ratio exceeds institutional policy thresholds of 80.00% for no-mortgage-insurance products, 85.00% for standard mortgage insurance products, 90.00% for expanded mortgage insurance eligibility, or 95.00% for maximum allowable LTV on retained portfolio products, the Underwriter must either reduce the approved_loan_amount to a figure that brings LTV into compliance, require the applicant to increase the down_payment_amount to reduce the loan amount needed, or decline the application if the LTV combined with other risk factors such as elevated DTI or marginal credit score creates unacceptable cumulative risk that exceeds the institution's risk appetite. The Underwriter documents in the loan file the specific LTV policy threshold applicable based on credit score tier, property type, and occupancy status, confirms the calculated LTV complies with that threshold, confirms that the final_appraised_value supports the loan amount and purchase price making the transaction commercially reasonable, and updates the previously calculated debt_to_income_ratio if any adjustments to loan amount were required based on LTV constraints, ensuring the final debt_to_income_ratio reflects the actual loan amount that will be approved rather than the initially requested amount.

        5.4 Final Decision Rendering and Approval Conditions

        5.4.1 Upon completion of credit underwriting per Section 5.2 and property valuation per Section 5.3, with verified_gross_monthly_income determined, debt_to_income_ratio calculated and within acceptable thresholds or supported by compensating factors, probability_of_default_score computed and mapped to credit_score_grade, final_risk_adjusted_interest_rate established through the pricing matrix, final_appraised_value determined and loan_to_value_ratio calculated to be within policy limits, the Credit Underwriter proceeds to final decision rendering by synthesizing all underwriting findings into a comprehensive decision determination, first verifying that no new derogatory information has appeared on the credit report since the initial pull by ordering a credit report refresh or supplement within five business days of anticipated loan closing to detect any new tradelines, inquiries, or adverse public records that could indicate deteriorating creditworthiness or undisclosed debt obligations, ensuring that the debt_to_income_ratio remains accurate and that no fraud indicators have emerged. The Underwriter then executes the final decision tree logic: if debt_to_income_ratio is at or below 43.00% (or the applicable portfolio threshold), and loan_to_value_ratio is at or below the maximum threshold for the credit score tier, and credit_score_grade is "A" or "B" without significant derogatory marks, and all income and asset verifications are satisfactory without unexplained discrepancies, and property_condition is rated C3 or better requiring no mandatory repairs, and the appraisal shows no adverse external factors or functional obsolescence issues, then the Underwriter populates final_decision with the value "Approved" indicating unconditional approval subject only to standard closing conditions such as clear title report, final verification of employment within ten days of closing, and receipt of homeowner's insurance policy declarations page.

        5.4.2 If the application meets the general credit parameters in substep 5.4.1 but requires the satisfaction of specific outstanding documentation or clarification items before the institution can be fully secured in its lending decision, the Underwriter populates final_decision with "Approved with Conditions" and enumerates each specific condition in the loan file with corresponding tracking in the Loan Origination System, recording conditions_count as the integer number of discrete conditions imposed, with typical conditions including: "Prior to closing, borrower must provide satisfactory evidence that medical collection account number 123456 reported on Equifax in the amount of $385 has been paid in full, in the form of a receipt or letter from the collection agency confirming zero balance"; "Prior to closing, borrower must provide written explanation regarding credit inquiry from ABC Bank dated April 5, 2025, clarifying whether new credit was obtained and if so, providing account statements showing the new tradeline and updating debt-to-income calculation accordingly"; "Prior to closing, borrower must provide IRS Form 4506-T signed and dated within 120 days of closing to permit lender to obtain tax return transcripts directly from the Internal Revenue Service for verification of reported income"; "Prior to closing, borrower must provide final verification of employment completed by the lender or designee within ten calendar days of the closing date confirming borrower remains employed in the same position with the same salary"; "Prior to closing, borrower must provide proof that earnest money deposit of $10,000 cleared the borrower's account and was received by the settlement agent, in the form of a bank statement showing the withdrawal and a receipt or ledger from the title company showing receipt"; "Prior to closing, for the subject property, borrower must provide evidence of homeowner's hazard insurance policy with effective date on or before closing date, covering the property for replacement cost of at least the final_appraised_value, naming the lender as mortgagee and loss payee, and showing the policy is paid in full for the first year"; "For the property located at [specific property address for investment property owned by borrower], borrower must provide evidence of current landlord insurance policy or add this property to the new loan closing requirements if it secures the subject transaction." The Underwriter documents each condition with sufficient specificity that the Loan Processor can clearly communicate the requirement to the borrower and can objectively determine when the condition has been satisfied, avoiding vague conditions such as "provide additional documentation regarding income" and instead stating precise requirements such as "provide 2024 IRS Form 1040 with all schedules, Schedule C showing business income, and signed tax preparer transmittal form or IRS account transcript confirming the return was filed."

        5.4.3 If the application fails to meet minimum credit standards due to excessive debt_to_income_ratio that cannot be mitigated by compensating factors, insufficient verified income to support the requested loan amount even at maximum permissible DTI thresholds, loan_to_value_ratio exceeding institutional maximums without ability to increase down payment or reduce loan amount while still meeting borrower objectives, credit_score_grade of "D" resulting from probability_of_default_score exceeding 7.00% without extraordinary offsetting factors, or appraisal identifying property_condition rating of C4 or C5 with repair costs exceeding borrower financial capacity or lender tolerance for renovation lending, the Underwriter populates final_decision with "Declined" and documents in the decline_reason field a clear, specific, factual explanation of the principal reasons for the adverse action decision, complying with the Equal Credit Opportunity Act's adverse action notice requirements under 12 CFR 1002.9 which mandate disclosure of the specific reasons for credit denial when an application is declined, with examples of compliant decline reasons including: "Credit score of 592 falls below the institution's minimum acceptable credit score of 620 for conventional mortgage products; debt-to-income ratio of 51.3% exceeds the maximum acceptable ratio of 50.0% and no compensating factors are present to warrant exception approval"; "Verified gross monthly income of $4,200 is insufficient to support the requested loan amount of $380,000 which would result in a debt-to-income ratio of 58.7% exceeding maximum permissible thresholds even with maximum compensating factors"; "Loan-to-value ratio of 96.2% exceeds the institution's maximum LTV of 95.0% for the borrower's credit score tier, and borrower has insufficient additional down payment funds available to reduce the loan amount to acceptable LTV levels"; "Credit report shows current Chapter 13 bankruptcy active payment plan filed February 2024, failing to meet the institution's requirement of Chapter 13 bankruptcy discharge or dismissal at least two years prior to application date"; "Employment verification unsuccessful; stated employer Acme Corporation reported that applicant has not been employed with the company since February 1, 2025, contradicting the applicant's representation of current full-time employment; no alternative verifiable income source documented"; "Property appraisal identified foundation cracking and differential settlement requiring structural engineer evaluation and potential foundation repair estimated at $25,000 to $40,000, resulting in property condition rating of C4, and borrower has insufficient liquid assets beyond down payment and closing costs to fund necessary repairs prior to closing."

        5.4.4 Regardless of whether final_decision is "Approved," "Approved with Conditions," or "Declined," the Underwriter records the final_decision_timestamp in the Loan Origination System representing the date and time when the underwriting decision was rendered, enabling calculation of turnaround_time_days by subtracting the application_timestamp from the final_decision_timestamp and converting the elapsed time to calendar days, with institutional performance targets establishing that turnaround_time_days should not exceed 25 calendar days for standard employment W-2 wage earner applications and should not exceed 35 calendar days for complex self-employed applications requiring detailed tax return analysis. For approved applications, the Underwriter confirms that the approved_loan_amount equals the requested_loan_amount if no adjustments were necessary based on income, DTI, or LTV constraints, or specifies a reduced approved_loan_amount if underwriting analysis determined that the requested amount was not supportable, confirms that approved_interest_rate equals the final_risk_adjusted_interest_rate calculated through the pricing matrix in substep 5.2.7, and confirms that approved_loan_term_months equals the requested loan_term_months unless a modification was made to improve payment affordability or meet investor requirements. The Underwriter determines first_pass_yield by evaluating whether the approval was granted with conditions_count equal to zero, representing a "clear to close" decision requiring no additional borrower documentation or clarification, in which case first_pass_yield is set to true indicating operational efficiency and high-quality initial file submission, or whether conditions_count exceeds zero indicating that additional work is required before the file can proceed to closing, in which case first_pass_yield is set to false and the types of conditions imposed are analyzed to identify process improvement opportunities such as enhancing initial application checklists, improving borrower communication regarding required documentation, or providing additional training to Loan Officers on complete file assembly prior to underwriting submission.

        5.4.5 For applications receiving final_decision of "Approved" or "Approved with Conditions," the loan file progresses to the closing preparation phase where the Loan Processor coordinates satisfaction of all outstanding conditions, orders the title examination and title insurance policy ensuring the lender will receive a first-priority lien position on the subject property free of encumbrances other than permitted exceptions such as property tax liens and utility easements, prepares the Closing Disclosure form required by the Truth in Lending Act and Real Estate Settlement Procedures Act integrated disclosure rule which must be provided to the borrower at least three business days before consummation and which itemizes all loan terms including the approved_loan_amount, approved_interest_rate, approved_loan_term_months, monthly principal and interest payment, estimated property taxes, estimated homeowner's insurance, mortgage insurance premium if applicable, total monthly payment, all lender charges, all third-party charges, all prepaid items and escrow deposits, cash required from borrower at closing, and annual percentage rate (APR) calculated pursuant to Regulation Z, coordinates the closing meeting at the title company or settlement attorney's office where the borrower executes the promissory note evidencing the debt obligation, the mortgage or deed of trust creating the security interest in the real property, the itemized settlement statement, and various ancillary documents including hazard insurance authorization, escrow account authorization if applicable, IRS Form 4506-T for tax transcript retrieval, and compliance acknowledgments, and following successful closing, the institution funds the loan by disbursing the approved_loan_amount to the settlement agent for payment to the seller (in Purchase transactions) or to payoff existing liens (in Refinance transactions) with any remainder returned to the borrower (in Cash-Out Refinance transactions), at which point funded_status is set to true and funding_date is recorded representing the date the institution committed its capital to the loan, thereby transitioning the loan from the origination workflow to the servicing and portfolio management workflow where monthly payment collection, escrow administration, investor reporting, and ongoing performance monitoring commence.

        5.4.6 Throughout the entire process from application through decision rendering, the institution's compliance monitoring systems execute automated fair lending surveillance by capturing for each application the demographic information collected pursuant to Home Mortgage Disclosure Act requirements including applicant ethnicity, race, sex, and Census tract geocode for the property_address, and the Quality Control and Fair Lending departments conduct ongoing statistical analysis comparing approval rates across demographic cohorts to identify any statistically significant disparities that could indicate potential disparate treatment or disparate impact violations of the Equal Credit Opportunity Act, analyzing whether approval rates for protected class applicants differ from approval rates for non-protected class applicants after controlling for legitimate risk factors such as credit score, DTI, and LTV, analyzing whether final_risk_adjusted_interest_rate spreads show statistically significant variation across demographic groups after controlling for credit risk factors, and analyzing whether conditions_count differs systematically across demographic groups which could suggest inconsistent application of underwriting standards, with the fair_lending_audit_flag set to true for any application randomly selected for detailed file review as part of the institution's ongoing fair lending quality control program or any application that triggers algorithmic surveillance alerts suggesting potential discriminatory pricing or terms, requiring specialized Fair Lending analysts to review the complete underwriting file, examine the justification for all pricing decisions and credit judgments, verify that compensating factor evaluations and exception approvals were applied consistently with comparable applications from other demographic groups, and document findings confirming non-discriminatory treatment or identifying training opportunities and process corrections if inconsistencies are detected, ensuring the institution maintains ongoing compliance with Regulation B implementing the Equal Credit Opportunity Act, Regulation C implementing the Home Mortgage Disclosure Act, and prudential regulatory guidance on fair lending risk management from federal banking regulators.

        5.4.7 For applications receiving final_decision of "Approved" or "Approved with Conditions" that successfully satisfy all closing conditions and achieve funded_status equal to true, the institution tracks ongoing performance through the servicing lifecycle monitoring pull_through_rate calculated as the percentage of approved applications that ultimately result in funded loans (with institutional targets typically ranging from 75% to 85%, as approved loans may fall out before closing due to borrower finding alternative financing, purchase contract cancellation, borrower disqualification through adverse credit changes, or other factors), analyzing reasons for fallout between approval and funding to identify controllable process improvements such as reducing cycle time between approval and closing to minimize exposure to rate volatility or borrower decision changes, and monitoring loan performance during the first twelve months post-origination to validate the accuracy of the underwriting and credit grading processes by calculating actual delinquency rates (defined as payments more than 30 days past due) segmented by credit_score_grade, comparing observed delinquency rates to the predicted probability_of_default_score estimates to calibrate the PD model through Bayesian updating of model coefficients using actual performance data, and identifying systematic underwriting weaknesses if certain loan characteristics or underwriter judgments correlate with elevated early payment default rates exceeding predictions, feeding these performance insights back into underwriting training programs, refinement of credit policy thresholds, and enhancement of automated underwriting system logic to continuously improve the precision and risk-sensitivity of the mortgage origination process while maintaining fair and equitable access to credit for all qualified applicants consistent with safe and sound lending practices and applicable legal and regulatory requirements.

        6. Output

        Upon complete execution of this Standard Operating Procedure, the institution shall have produced for each mortgage loan application a comprehensive set of deterministic outputs documented within the Loan Origination System and supporting loan file records, comprising: a prequalification_decision classification of "Qualified," "Conditionally Qualified," or "Not Qualified" based on preliminary credit screening, a max_loan_amount_prequalified denominated in United States Dollars representing the maximum borrowing capacity calculated from income, debt obligations, and credit profile subject to property identification and appraisal, an estimated_interest_rate_prequalification expressed as an annual percentage rate to three decimal places providing preliminary pricing guidance; a verified_gross_monthly_income figure denominated in United States Dollars reflecting underwriter-validated income through employment verification, tax return analysis, and documentation review rather than applicant self-declaration, a debt_to_income_ratio calculated to two decimal places representing the percentage of gross monthly income consumed by the proposed housing payment and all recurring debt obligations, a debt_service_coverage_ratio calculated to two decimal places indicating the margin between income and debt obligations available for savings and discretionary spending, a probability_of_default_score expressed as a percentage to two decimal places quantifying the statistical likelihood of loan default within 36 months based on multivariate logistic regression incorporating credit, capacity, collateral, and character factors, a credit_score_grade classification of "A," "B," "C," or "D" mapped from the probability of default score and used for risk-based capital allocation and pricing determination, a compensating_factors_present Boolean indicator documenting whether offsetting positive characteristics justify approval of applications with marginal risk metrics, and a final_risk_adjusted_interest_rate expressed as an annual percentage rate to three decimal places derived from the institution's multi-dimensional pricing matrix incorporating all risk factors; a final_appraised_value for the subject property denominated in United States Dollars rounded to the nearest thousand dollars reflecting the appraiser's opinion of market value supported by comparable sales analysis, an avm_estimated_value providing an independent statistical valuation for quality control comparison, an appraisal_avm_variance_percent calculated to two decimal places indicating the magnitude and direction of difference between manual and automated valuations, a second_review_triggered Boolean indicator documenting whether valuation variance exceeded tolerance thresholds requiring additional appraisal review, and a loan_to_value_ratio calculated to two decimal places representing the percentage relationship between loan amount and property value serving as the primary indicator of collateral risk; a final_decision classification of "Approved," "Approved with Conditions," or "Declined" representing the ultimate credit determination, a final_decision_timestamp recording the date and time of decision rendering for turnaround time calculation and regulatory compliance timing, an approved_loan_amount specifying the exact principal amount the institution is willing to lend, an approved_interest_rate specifying the exact annual percentage rate that will be charged on the approved loan amount, an approved_loan_term_months specifying the amortization period in months, a conditions_count enumerating the number of outstanding requirements that must be satisfied before loan closing for conditionally approved applications, a decline_reason providing specific factual explanation for applications receiving adverse action decisions to comply with Equal Credit Opportunity Act disclosure requirements; a turnaround_time_days metric calculated as calendar days elapsed from application_timestamp to final_decision_timestamp for operational performance measurement, a first_pass_yield Boolean indicator documenting whether the application was approved without conditions on initial underwriting review signaling high-quality file preparation and processing efficiency, a funded_status Boolean indicator confirming whether an approved loan ultimately closed and was disbursed, a funding_date recording when the institution's funds were advanced to complete the transaction, and a fair_lending_audit_flag Boolean indicator documenting whether the application was selected for detailed compliance review to ensure consistent and non-discriminatory application of credit standards; with all outputs documented in auditable form within the Loan Origination System's database tables, loan file images, and underwriting decision memoranda, available for retrieval during regulatory examinations, investor due diligence reviews, quality control audits, secondary market loan delivery validation, and internal performance analysis, providing the institution with quantifiable decision support for individual credit judgments, portfolio-level risk management, capital adequacy assessment under Basel III standardized approach for residential mortgage risk-weighted asset calculation, Home Mortgage Disclosure Act regulatory reporting, fair lending statistical monitoring, and continuous process improvement through analysis of cycle time metrics, approval rate trends, pull-through rate optimization, and early payment default correlation studies linking underwriting decisions to actual loan performance outcomes.

        </SOP>

        ***The data schema is as follows***:

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

        ***The data dataset is as follows***:

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
            ['application_id:MLN-2025-053247', 'application_timestamp:2025-05-09T11:10:00Z', 'loan_officer_id:LO-6215', 'applicant_name:Rachel Marie Young', 'applicant_ssn:***-**-7318', 'applicant_dob:1995-06-23', 'applicant_address:8146 Mountain Ridge, Raleigh, NC 27601', 'co_applicant_name:Joshua Allen Young', 'co_applicant_ssn:***-**-2954', 'employment_status:Full-Time', 'employer_name:Cisco Systems', 'employment_years:2.5', 'gross_monthly_income:8600.00', 'additional_income_monthly:400.00', 'liquid_assets:41000.00', 'retirement_accounts:62000.00', 'requested_loan_amount:335000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:4295 Research Park Drive, Raleigh, NC 27602', 'property_type:Townhouse', 'property_occupancy:Primary Residence', 'estimated_property_value:375000.00', 'down_payment_amount:56250.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-09T11:25:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-12T14:00:00Z', 'credit_pull_timestamp:2025-05-09T11:30:00Z', 'fico_score_equifax:695', 'fico_score_experian:702', 'fico_score_transunion:698', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3150.00', 'appraisal_order_timestamp:2025-05-13T09:30:00Z', 'appraiser_id:APR-8164', 'appraisal_company:Triangle Area Valuations', 'site_inspection_timestamp:2025-05-16T11:00:00Z', 'property_square_footage:1900', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-3956', 'processor_id:LP-6287', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053248', 'application_timestamp:2025-05-10T08:15:00Z', 'loan_officer_id:LO-4926', 'applicant_name:Steven Paul Anderson', 'applicant_ssn:***-**-6829', 'applicant_dob:1986-02-14', 'applicant_address:3742 Lakeshore Drive, Cleveland, OH 44101', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Cleveland Clinic', 'employment_years:11.0', 'gross_monthly_income:10800.00', 'additional_income_monthly:950.00', 'liquid_assets:78000.00', 'retirement_accounts:185000.00', 'requested_loan_amount:425000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:9251 Euclid Avenue, Cleveland, OH 44102', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:485000.00', 'down_payment_amount:72750.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-10T08:30:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-13T10:45:00Z', 'credit_pull_timestamp:2025-05-10T08:35:00Z', 'fico_score_equifax:764', 'fico_score_experian:771', 'fico_score_transunion:767', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2950.00', 'appraisal_order_timestamp:2025-05-14T08:00:00Z', 'appraiser_id:APR-5832', 'appraisal_company:Great Lakes Appraisals', 'site_inspection_timestamp:2025-05-17T10:00:00Z', 'property_square_footage:2550', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-7249', 'processor_id:LP-3681', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053249', 'application_timestamp:2025-05-10T13:20:00Z', 'loan_officer_id:LO-8371', 'applicant_name:Michelle Anne Roberts', 'applicant_ssn:***-**-4152', 'applicant_dob:1992-09-08', 'applicant_address:6483 Harbor View, San Francisco, CA 94101', 'co_applicant_name:Derek John Roberts', 'co_applicant_ssn:***-**-7698', 'employment_status:Self-Employed', 'employer_name:Roberts Design Studio', 'employment_years:6.0', 'gross_monthly_income:16500.00', 'additional_income_monthly:3200.00', 'liquid_assets:165000.00', 'retirement_accounts:285000.00', 'requested_loan_amount:1150000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:2847 Pacific Heights Boulevard, San Francisco, CA 94102', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:1380000.00', 'down_payment_amount:345000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-10T13:35:00Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-05-13T15:20:00Z', 'credit_pull_timestamp:2025-05-10T13:40:00Z', 'fico_score_equifax:812', 'fico_score_experian:819', 'fico_score_transunion:808', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3850.00', 'appraisal_order_timestamp:2025-05-14T09:00:00Z', 'appraiser_id:APR-7194', 'appraisal_company:Bay Area Premium Valuations', 'site_inspection_timestamp:2025-05-17T14:00:00Z', 'property_square_footage:3650', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-5827', 'processor_id:LP-9463', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053250', 'application_timestamp:2025-05-11T10:05:00Z', 'loan_officer_id:LO-2741', 'applicant_name:Brandon Keith Turner', 'applicant_ssn:***-**-9374', 'applicant_dob:1988-11-30', 'applicant_address:4916 Desert Springs Road, Tucson, AZ 85701', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Raytheon Technologies', 'employment_years:6.5', 'gross_monthly_income:9100.00', 'additional_income_monthly:700.00', 'liquid_assets:53000.00', 'retirement_accounts:98000.00', 'requested_loan_amount:325000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:7538 Saguaro Vista, Tucson, AZ 85702', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:365000.00', 'down_payment_amount:54750.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-11T10:20:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-14T11:15:00Z', 'credit_pull_timestamp:2025-05-11T10:25:00Z', 'fico_score_equifax:706', 'fico_score_experian:713', 'fico_score_transunion:709', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2750.00', 'appraisal_order_timestamp:2025-05-15T08:30:00Z', 'appraiser_id:APR-3947', 'appraisal_company:Sonoran Desert Appraisals', 'site_inspection_timestamp:2025-05-18T13:00:00Z', 'property_square_footage:2100', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-6138', 'processor_id:LP-4795', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053251', 'application_timestamp:2025-05-11T14:50:00Z', 'loan_officer_id:LO-5394', 'applicant_name:Kimberly Dawn Phillips', 'applicant_ssn:***-**-2856', 'applicant_dob:1983-04-27', 'applicant_address:8217 Riverside Parkway, Sacramento, CA 95801', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Kaiser Permanente', 'employment_years:13.5', 'gross_monthly_income:8900.00', 'additional_income_monthly:500.00', 'liquid_assets:92000.00', 'retirement_accounts:265000.00', 'requested_loan_amount:440000.00', 'loan_purpose:Refinance', 'loan_term_months:360', 'property_address:8217 Riverside Parkway, Sacramento, CA 95801', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:560000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-11T15:05:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-14T13:30:00Z', 'credit_pull_timestamp:2025-05-11T15:10:00Z', 'fico_score_equifax:781', 'fico_score_experian:788', 'fico_score_transunion:784', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2450.00', 'appraisal_order_timestamp:2025-05-15T09:00:00Z', 'appraiser_id:APR-6528', 'appraisal_company:Capital City Valuations', 'site_inspection_timestamp:2025-05-18T10:30:00Z', 'property_square_footage:2450', 'property_condition:Excellent', 'comparable_properties_count:3', 'underwriter_id:UW-9472', 'processor_id:LP-2864', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053252', 'application_timestamp:2025-05-12T09:35:00Z', 'loan_officer_id:LO-7618', 'applicant_name:Matthew Scott Campbell', 'applicant_ssn:***-**-5147', 'applicant_dob:1991-07-19', 'applicant_address:5629 Mountain Creek Lane, Boulder, CO 80301', 'co_applicant_name:Ashley Marie Campbell', 'co_applicant_ssn:***-**-8923', 'employment_status:Full-Time', 'employer_name:Google LLC', 'employment_years:4.0', 'gross_monthly_income:13200.00', 'additional_income_monthly:1800.00', 'liquid_assets:105000.00', 'retirement_accounts:142000.00', 'requested_loan_amount:625000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:3184 Flatiron Drive, Boulder, CO 80302', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:725000.00', 'down_payment_amount:145000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-12T09:50:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-15T10:20:00Z', 'credit_pull_timestamp:2025-05-12T09:55:00Z', 'fico_score_equifax:749', 'fico_score_experian:756', 'fico_score_transunion:752', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:3550.00', 'appraisal_order_timestamp:2025-05-16T08:00:00Z', 'appraiser_id:APR-4216', 'appraisal_company:Front Range Appraisals', 'site_inspection_timestamp:2025-05-19T11:00:00Z', 'property_square_footage:3050', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-8295', 'processor_id:LP-5637', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
            ['application_id:MLN-2025-053253', 'application_timestamp:2025-05-12T15:15:00Z', 'loan_officer_id:LO-3249', 'applicant_name:Eric Wayne Foster', 'applicant_ssn:***-**-7362', 'applicant_dob:1978-12-03', 'applicant_address:9461 Lakefront Terrace, Milwaukee, WI 53201', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Full-Time', 'employer_name:Northwestern Mutual', 'employment_years:15.0', 'gross_monthly_income:12600.00', 'additional_income_monthly:1100.00', 'liquid_assets:148000.00', 'retirement_accounts:425000.00', 'requested_loan_amount:385000.00', 'loan_purpose:Cash-Out Refinance', 'loan_term_months:360', 'property_address:9461 Lakefront Terrace, Milwaukee, WI 53201', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:550000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-05-12T15:30:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-05-15T14:00:00Z', 'credit_pull_timestamp:2025-05-12T15:35:00Z', 'fico_score_equifax:795', 'fico_score_experian:802', 'fico_score_transunion:798', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2280.00', 'appraisal_order_timestamp:2025-05-16T09:30:00Z', 'appraiser_id:APR-9583', 'appraisal_company:Great Lakes Property Assessors', 'site_inspection_timestamp:2025-05-19T14:00:00Z', 'property_square_footage:2950', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-4716', 'processor_id:LP-8129', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
        </data>

        ***The function APIs are as follows***:
        from typing import Dict
        from langchain.tools import StructuredTool  # 确保导入StructuredTool
        # from langchain_core.tools.structured import StructuredTool

        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.callbacks.base import BaseCallbackHandler
        import yaml
        import pandas as pd
        import json
        import ast
        from datetime import datetime
        from typing import Dict, List, Any, Optional
        from pydantic import BaseModel, Field
        import hashlib
        import random
        from datetime import datetime, timedelta
        import math

        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor

        import threading
        import os
        import sys
        from io import StringIO
        # ==================== 数据存储层 ====================
        MORTGAGE_DATA = []

        def parse_sample_data():
            """Parse the sample data into structured dictionaries"""
            global MORTGAGE_DATA
            
            sample_data_raw = [
                ['application_id:MLN-2025-048732', 'application_timestamp:2025-04-18T10:15:30Z', 'loan_officer_id:LO-8492', 'applicant_name:John Michael Smith', 'applicant_ssn:***-**-4829', 'applicant_dob:1985-06-15', 'applicant_address:1234 Oak Street, Springfield, IL 62704', 'co_applicant_name:Jane Elizabeth Smith', 'co_applicant_ssn:***-**-7391', 'employment_status:Full-Time', 'employer_name:Acme Corporation', 'employment_years:5.5', 'gross_monthly_income:8500.00', 'additional_income_monthly:1200.00', 'liquid_assets:75000.00', 'retirement_accounts:150000.00', 'requested_loan_amount:380000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:5678 Elm Avenue, Springfield, IL 62705', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:450000.00', 'down_payment_amount:90000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-18T10:22:45Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-20T14:30:00Z', 'credit_pull_timestamp:2025-04-18T10:25:00Z', 'fico_score_equifax:742', 'fico_score_experian:738', 'fico_score_transunion:745', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:2800.00', 'appraisal_order_timestamp:2025-04-21T09:00:00Z', 'appraiser_id:APR-3291', 'appraisal_company:National Valuation Services', 'site_inspection_timestamp:2025-04-23T13:00:00Z', 'property_square_footage:2400', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-6174', 'processor_id:LP-2938', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
                ['application_id:MLN-2025-049103', 'application_timestamp:2025-04-19T09:30:15Z', 'loan_officer_id:LO-7234', 'applicant_name:Maria Garcia Rodriguez', 'applicant_ssn:***-**-6721', 'applicant_dob:1982-03-22', 'applicant_address:890 Pine Road, Austin, TX 78701', 'co_applicant_name:', 'co_applicant_ssn:', 'employment_status:Self-Employed', 'employer_name:MG Consulting LLC', 'employment_years:8.0', 'gross_monthly_income:11200.00', 'additional_income_monthly:2400.00', 'liquid_assets:125000.00', 'retirement_accounts:220000.00', 'requested_loan_amount:525000.00', 'loan_purpose:Purchase', 'loan_term_months:360', 'property_address:3456 Maple Drive, Austin, TX 78702', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:625000.00', 'down_payment_amount:125000.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-19T09:45:30Z', 'pay_stubs_uploaded:False', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:False', 'document_verification_timestamp:2025-04-22T11:15:00Z', 'credit_pull_timestamp:2025-04-19T09:50:00Z', 'fico_score_equifax:798', 'fico_score_experian:805', 'fico_score_transunion:792', 'derogatory_marks_count:0', 'bankruptcy_history:False', 'total_monthly_debt_obligations:1850.00', 'appraisal_order_timestamp:2025-04-23T10:30:00Z', 'appraiser_id:APR-4512', 'appraisal_company:Premier Appraisal Group', 'site_inspection_timestamp:2025-04-26T14:00:00Z', 'property_square_footage:3200', 'property_condition:Excellent', 'comparable_properties_count:4', 'underwriter_id:UW-8823', 'processor_id:LP-4621', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
                ['application_id:MLN-2025-049287', 'application_timestamp:2025-04-20T14:20:00Z', 'loan_officer_id:LO-5639', 'applicant_name:Robert James Thompson', 'applicant_ssn:***-**-9182', 'applicant_dob:1978-11-08', 'applicant_address:2345 Cedar Lane, Denver, CO 80202', 'co_applicant_name:Susan Marie Thompson', 'co_applicant_ssn:***-**-3047', 'employment_status:Full-Time', 'employer_name:TechCorp Industries', 'employment_years:12.0', 'gross_monthly_income:14500.00', 'additional_income_monthly:800.00', 'liquid_assets:185000.00', 'retirement_accounts:425000.00', 'requested_loan_amount:640000.00', 'loan_purpose:Cash-Out Refinance', 'loan_term_months:360', 'property_address:2345 Cedar Lane, Denver, CO 80202', 'property_type:Single Family', 'property_occupancy:Primary Residence', 'estimated_property_value:800000.00', 'down_payment_amount:0.00', 'credit_consent_signed:True', 'credit_consent_timestamp:2025-04-20T14:35:00Z', 'pay_stubs_uploaded:True', 'tax_returns_uploaded:True', 'bank_statements_uploaded:True', 'w2_forms_uploaded:True', 'document_verification_timestamp:2025-04-23T09:45:00Z', 'credit_pull_timestamp:2025-04-20T14:40:00Z', 'fico_score_equifax:712', 'fico_score_experian:718', 'fico_score_transunion:709', 'derogatory_marks_count:2', 'bankruptcy_history:False', 'total_monthly_debt_obligations:4200.00', 'appraisal_order_timestamp:2025-04-24T08:00:00Z', 'appraiser_id:APR-2876', 'appraisal_company:Rocky Mountain Valuations', 'site_inspection_timestamp:2025-04-27T10:30:00Z', 'property_square_footage:3800', 'property_condition:Good', 'comparable_properties_count:3', 'underwriter_id:UW-5491', 'processor_id:LP-7832', 'application_status:', 'prequalification_decision:', 'max_loan_amount_prequalified:', 'estimated_interest_rate_prequalification:', 'verified_gross_monthly_income:', 'debt_to_income_ratio:', 'debt_service_coverage_ratio:', 'probability_of_default_score:', 'compensating_factors_present:', 'credit_score_grade:', 'final_risk_adjusted_interest_rate:', 'final_appraised_value:', 'avm_estimated_value:', 'appraisal_avm_variance_percent:', 'second_review_triggered:', 'loan_to_value_ratio:', 'final_decision:', 'final_decision_timestamp:', 'approved_loan_amount:', 'approved_interest_rate:', 'approved_loan_term_months:', 'conditions_count:', 'decline_reason:', 'turnaround_time_days:', 'first_pass_yield:', 'funded_status:', 'funding_date:', 'fair_lending_audit_flag:'],
                ...
            ]
            
            for row in sample_data_raw:
                record = {}
                for item in row:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        record[key] = value if value else None
                MORTGAGE_DATA.append(record)

        # Initialize data
        parse_sample_data()

        def get_application_data(application_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve application data by application_id"""
            for record in MORTGAGE_DATA:
                if record.get('application_id') == application_id:
                    return record
            return None

        def update_application_data(application_id: str, updates: Dict[str, Any]) -> bool:
            """Update application data with new values"""
            for record in MORTGAGE_DATA:
                if record.get('application_id') == application_id:
                    record.update(updates)
                    return True
            return False

        def convert_value(value: str, target_type: str):
            """Convert string value to appropriate type"""
            if value is None or value == '':
                return None
            
            if target_type == 'bool':
                return value.lower() == 'true'
            elif target_type == 'int':
                return int(float(value))
            elif target_type == 'float':
                return float(value)
            else:
                return value


        # ==================== API函数实现 ====================
        def ValidateApplicantIdentity(
            application_id: str,
            applicant_name: str,
            applicant_ssn: str,
            applicant_dob: str,
            applicant_address: str,
            co_applicant_name: Optional[str] = None,
            co_applicant_ssn: Optional[str] = None
        ) -> Dict[str, Any]:
            """Validates applicant identity and performs OFAC screening"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Validate SSN pattern (masked format ***-**-NNNN)
            ssn_pattern_valid = len(applicant_ssn) == 11 and applicant_ssn[0:7] == "***-**-"
            
            # OFAC screening simulation - always pass for valid SSN patterns
            ofac_clear = ssn_pattern_valid
            
            # Identity verification
            identity_verified = True
            verification_issues = []
            
            if not ssn_pattern_valid:
                identity_verified = False
                verification_issues.append("Invalid SSN format")
            
            # Co-applicant verification if present
            co_applicant_verified = True
            if co_applicant_name and co_applicant_ssn:
                co_ssn_pattern_valid = len(co_applicant_ssn) == 11 and co_applicant_ssn[0:7] == "***-**-"
                if not co_ssn_pattern_valid:
                    co_applicant_verified = False
                    verification_issues.append("Invalid co-applicant SSN format")
            
            result = {
                "application_id": application_id,
                "identity_verification_status": "Verified" if identity_verified and co_applicant_verified else "Failed",
                "ofac_screening_status": "Clear" if ofac_clear else "Flagged",
                "verified_applicant_name": applicant_name if identity_verified else None,
                "verified_applicant_ssn": applicant_ssn if identity_verified else None,
                "verified_applicant_dob": applicant_dob if identity_verified else None,
                "verification_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "verification_issues": verification_issues
            }
            
            # Update application data
            update_application_data(application_id, {
                "identity_verification_status": result["identity_verification_status"],
                "ofac_screening_status": result["ofac_screening_status"]
            })
            
            return result

        # API 2: ValidateEmploymentAndIncome
        def ValidateEmploymentAndIncome(
            application_id: str,
            employment_status: str,
            gross_monthly_income: float,
            pay_stubs_uploaded: bool,
            w2_forms_uploaded: bool,
            employer_name: Optional[str] = None,
            employment_years: Optional[float] = None,
            additional_income_monthly: Optional[float] = None,
            tax_returns_uploaded: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Validates employment and income information"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            validation_passed = True
            validation_issues = []
            
            # Check document requirements
            if employment_status in ["Full-Time", "Part-Time"]:
                if not pay_stubs_uploaded or not w2_forms_uploaded:
                    validation_passed = False
                    validation_issues.append("Missing required income documents for W-2 employment")
            
            if employment_status == "Self-Employed":
                if not tax_returns_uploaded:
                    validation_passed = False
                    validation_issues.append("Missing tax returns for self-employed status")
            
            # Calculate preliminary income
            preliminary_monthly_income = gross_monthly_income
            if additional_income_monthly:
                preliminary_monthly_income += additional_income_monthly
            
            # Income stability assessment
            income_stability_rating = "Stable"
            if employment_years and employment_years < 2.0:
                income_stability_rating = "Moderate"
                validation_issues.append("Employment tenure less than 2 years")
            elif employment_years and employment_years >= 5.0:
                income_stability_rating = "Strong"
            
            result = {
                "application_id": application_id,
                "income_validation_status": "Passed" if validation_passed else "Failed",
                "preliminary_gross_monthly_income": round(preliminary_monthly_income, 2),
                "income_stability_rating": income_stability_rating,
                "employment_verification_required": employment_status in ["Full-Time", "Part-Time"],
                "validation_issues": validation_issues,
                "validation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "preliminary_gross_monthly_income": result["preliminary_gross_monthly_income"],
                "income_validation_status": result["income_validation_status"]
            })
            
            return result

        # API 3: ValidateLiquidAssets
        def ValidateLiquidAssets(
            application_id: str,
            liquid_assets: float,
            bank_statements_uploaded: bool,
            down_payment_amount: float,
            requested_loan_amount: float,
            applicant_dob: str,
            retirement_accounts: Optional[float] = None
        ) -> Dict[str, Any]:
            """Validates liquid assets and calculates reserves"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            validation_passed = True
            validation_issues = []
            
            if not bank_statements_uploaded:
                validation_passed = False
                validation_issues.append("Bank statements not uploaded")
            
            # Calculate applicant age for retirement account discount
            birth_date = datetime.strptime(applicant_dob, "%Y-%m-%d")
            current_date = datetime.utcnow()
            age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
            
            # Apply retirement account discount
            retirement_discount_factor = 0.75 if age < 59.5 else 0.90  # 25% or 10% discount
            discounted_retirement = (retirement_accounts * retirement_discount_factor) if retirement_accounts else 0.0
            
            # Calculate total available assets
            total_available_assets = liquid_assets + discounted_retirement
            
            # Estimate closing costs (3% of loan amount)
            estimated_closing_costs = requested_loan_amount * 0.03
            
            # Calculate reserves after down payment and closing costs
            reserves_after_purchase = total_available_assets - down_payment_amount - estimated_closing_costs
            
            # Check asset sufficiency
            if total_available_assets < (down_payment_amount + estimated_closing_costs):
                validation_passed = False
                validation_issues.append("Insufficient assets for down payment and closing costs")
            
            result = {
                "application_id": application_id,
                "asset_validation_status": "Passed" if validation_passed else "Failed",
                "verified_liquid_assets": round(liquid_assets, 2),
                "discounted_retirement_accounts": round(discounted_retirement, 2),
                "total_available_assets": round(total_available_assets, 2),
                "estimated_closing_costs": round(estimated_closing_costs, 2),
                "reserves_after_purchase": round(reserves_after_purchase, 2),
                "applicant_age": age,
                "retirement_discount_factor": retirement_discount_factor,
                "validation_issues": validation_issues,
                "validation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "asset_validation_status": result["asset_validation_status"],
                "verified_liquid_assets": result["verified_liquid_assets"]
            })
            
            return result

        # API 4: PullCreditReports
        def PullCreditReports(
            application_id: str,
            verified_applicant_name: str,
            verified_applicant_ssn: str,
            verified_applicant_dob: str,
            applicant_address: str,
            credit_consent_signed: bool,
            credit_consent_timestamp: str,
            co_applicant_name: Optional[str] = None,
            co_applicant_ssn: Optional[str] = None
        ) -> Dict[str, Any]:
            """Pulls tri-bureau credit reports"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            if not credit_consent_signed:
                return {"error": "Credit consent not signed - cannot pull credit reports"}
            
            # Retrieve credit scores from application data
            fico_score_equifax = convert_value(app_data.get('fico_score_equifax'), 'int')
            fico_score_experian = convert_value(app_data.get('fico_score_experian'), 'int')
            fico_score_transunion = convert_value(app_data.get('fico_score_transunion'), 'int')
            derogatory_marks_count = convert_value(app_data.get('derogatory_marks_count'), 'int')
            bankruptcy_history = convert_value(app_data.get('bankruptcy_history'), 'bool')
            total_monthly_debt_obligations = convert_value(app_data.get('total_monthly_debt_obligations'), 'float')
            
            # If co-applicant exists, would pull their reports too
            co_applicant_fico_equifax = None
            co_applicant_fico_experian = None
            co_applicant_fico_transunion = None
            
            result = {
                "application_id": application_id,
                "credit_pull_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fico_score_equifax": fico_score_equifax,
                "fico_score_experian": fico_score_experian,
                "fico_score_transunion": fico_score_transunion,
                "co_applicant_fico_equifax": co_applicant_fico_equifax,
                "co_applicant_fico_experian": co_applicant_fico_experian,
                "co_applicant_fico_transunion": co_applicant_fico_transunion,
                "derogatory_marks_count": derogatory_marks_count,
                "bankruptcy_history": bankruptcy_history,
                "total_monthly_debt_obligations": round(total_monthly_debt_obligations, 2) if total_monthly_debt_obligations else 0.0,
                "credit_report_full_data": {
                    "applicant": {
                        "equifax_score": fico_score_equifax,
                        "experian_score": fico_score_experian,
                        "transunion_score": fico_score_transunion,
                        "derogatory_marks": derogatory_marks_count,
                        "bankruptcy": bankruptcy_history
                    }
                }
            }
            
            # Update application data
            update_application_data(application_id, {
                "credit_pull_timestamp": result["credit_pull_timestamp"],
                "fico_score_equifax": fico_score_equifax,
                "fico_score_experian": fico_score_experian,
                "fico_score_transunion": fico_score_transunion
            })
            
            return result

        # API 5: CalculateRepresentativeCreditScore
        def CalculateRepresentativeCreditScore(
            application_id: str,
            fico_score_equifax: Optional[int] = None,
            fico_score_experian: Optional[int] = None,
            fico_score_transunion: Optional[int] = None,
            co_applicant_fico_equifax: Optional[int] = None,
            co_applicant_fico_experian: Optional[int] = None,
            co_applicant_fico_transunion: Optional[int] = None
        ) -> Dict[str, Any]:
            """Calculates representative credit score using tri-merge methodology"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Collect applicant scores
            applicant_scores = []
            if fico_score_equifax:
                applicant_scores.append(fico_score_equifax)
            if fico_score_experian:
                applicant_scores.append(fico_score_experian)
            if fico_score_transunion:
                applicant_scores.append(fico_score_transunion)
            
            # Calculate representative score for applicant
            if len(applicant_scores) == 0:
                return {"error": "No credit scores available"}
            elif len(applicant_scores) == 1:
                applicant_representative_score = applicant_scores[0]
            elif len(applicant_scores) == 2:
                applicant_representative_score = min(applicant_scores)
            else:  # 3 scores
                applicant_scores_sorted = sorted(applicant_scores)
                applicant_representative_score = applicant_scores_sorted[1]  # Middle score
            
            # Calculate co-applicant representative score if exists
            co_applicant_representative_score = None
            if co_applicant_fico_equifax or co_applicant_fico_experian or co_applicant_fico_transunion:
                co_scores = []
                if co_applicant_fico_equifax:
                    co_scores.append(co_applicant_fico_equifax)
                if co_applicant_fico_experian:
                    co_scores.append(co_applicant_fico_experian)
                if co_applicant_fico_transunion:
                    co_scores.append(co_applicant_fico_transunion)
                
                if len(co_scores) == 1:
                    co_applicant_representative_score = co_scores[0]
                elif len(co_scores) == 2:
                    co_applicant_representative_score = min(co_scores)
                else:
                    co_scores_sorted = sorted(co_scores)
                    co_applicant_representative_score = co_scores_sorted[1]
            
            # Final representative score (lower of the two if co-applicant exists)
            if co_applicant_representative_score:
                representative_credit_score = min(applicant_representative_score, co_applicant_representative_score)
            else:
                representative_credit_score = applicant_representative_score
            
            result = {
                "application_id": application_id,
                "applicant_representative_score": applicant_representative_score,
                "co_applicant_representative_score": co_applicant_representative_score,
                "representative_credit_score": representative_credit_score,
                "tri_merge_methodology": "Middle score of three bureaus, or lower of two, or single score if only one available",
                "calculation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "representative_credit_score": representative_credit_score
            })
            
            return result

        # API 6: ExecutePrequalificationAnalysis
        def ExecutePrequalificationAnalysis(
            application_id: str,
            representative_credit_score: int,
            preliminary_gross_monthly_income: float,
            total_monthly_debt_obligations: float,
            requested_loan_amount: float,
            loan_term_months: int,
            estimated_property_value: float,
            down_payment_amount: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str
        ) -> Dict[str, Any]:
            """Executes pre-qualification analysis"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Check minimum credit score requirement
            if representative_credit_score < 620:
                result = {
                    "application_id": application_id,
                    "prequalification_decision": "Not Qualified",
                    "decline_reason": "Credit score below minimum threshold of 620",
                    "representative_credit_score": representative_credit_score
                }
                update_application_data(application_id, {
                    "prequalification_decision": "Not Qualified",
                    "decline_reason": "Credit score below minimum threshold of 620"
                })
                return result
            
            # Determine base interest rate based on credit score tier
            if representative_credit_score >= 780:
                base_rate = 6.500
                rate_adjustment = -0.500
            elif representative_credit_score >= 760:
                base_rate = 6.500
                rate_adjustment = -0.250
            elif representative_credit_score >= 740:
                base_rate = 6.500
                rate_adjustment = 0.000
            elif representative_credit_score >= 720:
                base_rate = 6.500
                rate_adjustment = 0.125
            elif representative_credit_score >= 700:
                base_rate = 6.500
                rate_adjustment = 0.250
            elif representative_credit_score >= 680:
                base_rate = 6.500
                rate_adjustment = 0.500
            elif representative_credit_score >= 660:
                base_rate = 6.500
                rate_adjustment = 0.875
            elif representative_credit_score >= 640:
                base_rate = 6.500
                rate_adjustment = 1.375
            else:
                base_rate = 6.500
                rate_adjustment = 2.000
            
            # Apply property type adjustment
            if property_type == "Condo":
                rate_adjustment += 0.250
            elif property_type == "Townhouse":
                rate_adjustment += 0.375
            elif property_type == "Multi-Family":
                rate_adjustment += 0.625
            
            # Apply occupancy adjustment
            if property_occupancy == "Second Home":
                rate_adjustment += 0.375
            elif property_occupancy == "Investment Property":
                rate_adjustment += 1.000
            
            # Apply loan purpose adjustment
            if loan_purpose == "Refinance":
                rate_adjustment += 0.125
            elif loan_purpose == "Cash-Out Refinance":
                rate_adjustment += 0.500
            
            estimated_interest_rate = base_rate + rate_adjustment
            
            # Calculate preliminary LTV
            preliminary_ltv = (requested_loan_amount / estimated_property_value) * 100.0
            
            # Calculate monthly payment (P&I only for preliminary)
            monthly_rate = estimated_interest_rate / 100.0 / 12.0
            n_payments = loan_term_months
            if monthly_rate > 0:
                monthly_pi = requested_loan_amount * (monthly_rate * math.pow(1 + monthly_rate, n_payments)) / (math.pow(1 + monthly_rate, n_payments) - 1)
            else:
                monthly_pi = requested_loan_amount / n_payments
            
            # Estimate property taxes (1.2% annually)
            monthly_taxes = (estimated_property_value * 0.012) / 12.0
            
            # Estimate insurance (0.5% annually)
            monthly_insurance = (estimated_property_value * 0.005) / 12.0
            
            # Estimate MI if LTV > 80%
            monthly_mi = 0.0
            if preliminary_ltv > 80.0:
                # Simple MI calculation based on LTV and credit
                if representative_credit_score >= 760:
                    mi_rate = 0.005
                elif representative_credit_score >= 700:
                    mi_rate = 0.007
                else:
                    mi_rate = 0.010
                monthly_mi = (requested_loan_amount * mi_rate) / 12.0
            
            # Total housing payment
            monthly_housing_payment = monthly_pi + monthly_taxes + monthly_insurance + monthly_mi
            
            # Calculate DTI
            total_monthly_obligations = monthly_housing_payment + total_monthly_debt_obligations
            debt_to_income_ratio = (total_monthly_obligations / preliminary_gross_monthly_income) * 100.0
            
            # Determine prequalification decision
            if debt_to_income_ratio <= 43.0:
                prequalification_decision = "Qualified"
            elif debt_to_income_ratio <= 50.0:
                prequalification_decision = "Conditionally Qualified"
            else:
                prequalification_decision = "Not Qualified"
            
            # Calculate max loan amount at 43% DTI
            max_housing_payment_43 = (preliminary_gross_monthly_income * 0.43) - total_monthly_debt_obligations
            # Reverse calculate loan amount (simplified)
            max_loan_amount_prequalified = (max_housing_payment_43 - monthly_taxes - monthly_insurance - monthly_mi) / (monthly_rate * math.pow(1 + monthly_rate, n_payments) / (math.pow(1 + monthly_rate, n_payments) - 1)) if monthly_rate > 0 else 0.0
            
            result = {
                "application_id": application_id,
                "prequalification_decision": prequalification_decision,
                "preliminary_debt_to_income_ratio": round(debt_to_income_ratio, 2),
                "estimated_interest_rate_prequalification": round(estimated_interest_rate, 3),
                "preliminary_ltv_ratio": round(preliminary_ltv, 2),
                "estimated_monthly_payment": round(monthly_housing_payment, 2),
                "max_loan_amount_prequalified": round(max_loan_amount_prequalified, 2),
                "prequalification_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "prequalification_decision": prequalification_decision,
                "preliminary_ltv_ratio": round(preliminary_ltv, 2),
                "estimated_interest_rate_prequalification": round(estimated_interest_rate, 3),
                "max_loan_amount_prequalified": round(max_loan_amount_prequalified, 2)
            })
            
            return result

        # API 7: VerifyEmploymentAndIncome
        def VerifyEmploymentAndIncome(
            application_id: str,
            prequalification_decision: str,
            employment_status: str,
            gross_monthly_income: float,
            pay_stubs_uploaded: bool,
            w2_forms_uploaded: bool,
            employer_name: Optional[str] = None,
            employment_years: Optional[float] = None,
            additional_income_monthly: Optional[float] = None,
            tax_returns_uploaded: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Verifies employment and income through VOE and document analysis"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Check prequalification gate
            if prequalification_decision == "Not Qualified":
                return {"error": "Cannot verify income for Not Qualified applications"}
            
            verification_passed = True
            verification_issues = []
            
            # Simulate VOE for Full-Time/Part-Time
            if employment_status in ["Full-Time", "Part-Time"]:
                if not pay_stubs_uploaded or not w2_forms_uploaded:
                    verification_passed = False
                    verification_issues.append("Missing required documents")
                
                # VOE simulation - assume all verifications pass
                verified_gross_monthly_income = gross_monthly_income
                if additional_income_monthly:
                    verified_gross_monthly_income += additional_income_monthly * 0.95  # Apply 5% haircut to additional income
            
            # Self-employed income verification
            elif employment_status == "Self-Employed":
                if not tax_returns_uploaded:
                    verification_passed = False
                    verification_issues.append("Tax returns required for self-employed")
                    verified_gross_monthly_income = 0.0
                else:
                    # Simulate tax return analysis with 2-year averaging
                    # Assume 3% year-over-year decline for complexity
                    year1_income = gross_monthly_income * 12
                    year2_income = year1_income * 0.97  # 3% decline
                    average_annual = (year1_income + year2_income) / 2
                    verified_gross_monthly_income = average_annual / 12
                    
                    if additional_income_monthly:
                        verified_gross_monthly_income += additional_income_monthly * 0.90
            
            else:
                verified_gross_monthly_income = gross_monthly_income
            
            # Income variance check
            income_variance_percent = ((verified_gross_monthly_income - gross_monthly_income) / gross_monthly_income * 100) if gross_monthly_income > 0 else 0
            if abs(income_variance_percent) > 20:
                verification_issues.append(f"Income variance exceeds 20%: {income_variance_percent:.1f}%")
            
            result = {
                "application_id": application_id,
                "income_verification_status": "Verified" if verification_passed else "Failed",
                "verified_gross_monthly_income": round(verified_gross_monthly_income, 2),
                "income_variance_percent": round(income_variance_percent, 2),
                "employment_verified": verification_passed,
                "verification_method": "VOE" if employment_status in ["Full-Time", "Part-Time"] else "Tax Return Analysis",
                "verification_issues": verification_issues,
                "verification_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "verified_gross_monthly_income": round(verified_gross_monthly_income, 2),
                "income_verification_status": result["income_verification_status"]
            })
            
            return result

        # API 8: VerifyAssetsAndReserves
        def VerifyAssetsAndReserves(
            application_id: str,
            prequalification_decision: str,
            liquid_assets: float,
            bank_statements_uploaded: bool,
            down_payment_amount: float,
            requested_loan_amount: float,
            applicant_dob: str,
            verified_gross_monthly_income: float,
            property_type: str,
            property_occupancy: str,
            retirement_accounts: Optional[float] = None
        ) -> Dict[str, Any]:
            """Verifies assets and calculates reserve months"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            if prequalification_decision == "Not Qualified":
                return {"error": "Cannot verify assets for Not Qualified applications"}
            
            verification_passed = True
            verification_issues = []
            
            if not bank_statements_uploaded:
                verification_passed = False
                verification_issues.append("Bank statements not uploaded")
            
            # Calculate age for retirement discount
            birth_date = datetime.strptime(applicant_dob, "%Y-%m-%d")
            current_date = datetime.utcnow()
            age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
            
            retirement_discount = 0.75 if age < 59.5 else 0.90
            discounted_retirement = (retirement_accounts * retirement_discount) if retirement_accounts else 0.0
            
            # Total verified assets
            verified_liquid_assets_final = liquid_assets + discounted_retirement
            
            # Closing costs estimate (3%)
            estimated_closing_costs = requested_loan_amount * 0.03
            
            # Calculate remaining assets after down payment and closing
            remaining_assets = verified_liquid_assets_final - down_payment_amount - estimated_closing_costs
            
            # Check sufficiency
            if remaining_assets < 0:
                verification_passed = False
                verification_issues.append("Insufficient assets for down payment and closing costs")
            
            # Estimate monthly housing payment for reserve calculation (simplified)
            # Use 1% of loan amount as rough monthly payment estimate
            estimated_monthly_housing = requested_loan_amount * 0.01
            
            # Calculate reserve months
            reserve_months_calculated = remaining_assets / estimated_monthly_housing if estimated_monthly_housing > 0 else 0
            
            # Large deposit analysis - simulate checking for deposits > 50% of monthly income
            large_deposits_flagged = []
            # For simulation, assume no large deposits flagged
            
            result = {
                "application_id": application_id,
                "asset_verification_status": "Verified" if verification_passed else "Failed",
                "verified_liquid_assets_final": round(verified_liquid_assets_final, 2),
                "discounted_retirement_assets": round(discounted_retirement, 2),
                "remaining_assets_after_closing": round(remaining_assets, 2),
                "reserve_months_calculated": round(reserve_months_calculated, 2),
                "large_deposits_flagged": large_deposits_flagged,
                "verification_issues": verification_issues,
                "verification_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "verified_liquid_assets_final": round(verified_liquid_assets_final, 2),
                "reserve_months_calculated": round(reserve_months_calculated, 2)
            })
            
            return result

        # API 9: AnalyzeCreditHistory
        def AnalyzeCreditHistory(
            application_id: str,
            credit_report_full_data: Dict[str, Any],
            derogatory_marks_count: int,
            bankruptcy_history: bool
        ) -> Dict[str, Any]:
            """Analyzes credit history in detail"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Detailed derogatory analysis with weighted scoring
            # 30-day late = 1 point, 60-day = 2 points, 90+ = 3 points
            # Simulate based on derogatory_marks_count
            detailed_derogatory_marks_count = derogatory_marks_count
            
            # Payment history quality assessment
            if derogatory_marks_count == 0:
                payment_history_quality = "Excellent"
            elif derogatory_marks_count <= 2:
                payment_history_quality = "Good"
            elif derogatory_marks_count <= 4:
                payment_history_quality = "Fair"
            else:
                payment_history_quality = "Poor"
            
            # Credit depth rating
            # Simulate based on credit score
            representative_score = convert_value(app_data.get('representative_credit_score'), 'int')
            if representative_score and representative_score >= 760:
                credit_depth_rating = "Deep"
            elif representative_score and representative_score >= 700:
                credit_depth_rating = "Moderate"
            elif representative_score and representative_score >= 640:
                credit_depth_rating = "Thin"
            else:
                credit_depth_rating = "Insufficient"
            
            # Bankruptcy seasoning check
            bankruptcy_seasoning_compliant = True
            if bankruptcy_history:
                # Assume sufficient time has passed for existing data
                bankruptcy_seasoning_compliant = True
            
            # Foreclosure check (simulate from data if available)
            foreclosure_history_compliant = True
            
            result = {
                "application_id": application_id,
                "detailed_derogatory_marks_count": detailed_derogatory_marks_count,
                "payment_history_quality": payment_history_quality,
                "credit_depth_rating": credit_depth_rating,
                "bankruptcy_seasoning_compliant": bankruptcy_seasoning_compliant,
                "foreclosure_history_compliant": foreclosure_history_compliant,
                "collections_count": 0,  # Simplified
                "public_records_count": 0,  # Simplified
                "analysis_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "payment_history_quality": payment_history_quality,
                "credit_depth_rating": credit_depth_rating,
                "detailed_derogatory_marks_count": detailed_derogatory_marks_count
            })
            
            return result

        # API 10: CalculateDebtRatios
        def CalculateDebtRatios(
            application_id: str,
            verified_gross_monthly_income: float,
            total_monthly_debt_obligations: float,
            requested_loan_amount: float,
            loan_term_months: int,
            estimated_interest_rate_prequalification: float,
            property_address: str,
            estimated_property_value: float,
            representative_credit_score: int,
            preliminary_ltv_ratio: float
        ) -> Dict[str, Any]:
            """Calculates precise debt-to-income and debt service coverage ratios"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Calculate monthly P&I payment
            monthly_rate = estimated_interest_rate_prequalification / 100.0 / 12.0
            n_payments = loan_term_months
            
            if monthly_rate > 0:
                monthly_pi = requested_loan_amount * (monthly_rate * math.pow(1 + monthly_rate, n_payments)) / (math.pow(1 + monthly_rate, n_payments) - 1)
            else:
                monthly_pi = requested_loan_amount / n_payments
            
            # Property taxes (1.2% annually, adjusted by location - simplified to 1.2%)
            monthly_property_tax = (estimated_property_value * 0.012) / 12.0
            
            # Hazard insurance (0.5% annually)
            monthly_hazard_insurance = (estimated_property_value * 0.005) / 12.0
            
            # Mortgage insurance if LTV > 80%
            monthly_mortgage_insurance = 0.0
            if preliminary_ltv_ratio > 80.0:
                # MI rate based on LTV and credit score
                if representative_credit_score >= 760 and preliminary_ltv_ratio <= 85:
                    mi_annual_rate = 0.003
                elif representative_credit_score >= 760 and preliminary_ltv_ratio <= 90:
                    mi_annual_rate = 0.005
                elif representative_credit_score >= 700 and preliminary_ltv_ratio <= 85:
                    mi_annual_rate = 0.005
                elif representative_credit_score >= 700 and preliminary_ltv_ratio <= 90:
                    mi_annual_rate = 0.007
                elif representative_credit_score >= 680 and preliminary_ltv_ratio <= 90:
                    mi_annual_rate = 0.010
                else:
                    mi_annual_rate = 0.012
                
                monthly_mortgage_insurance = (requested_loan_amount * mi_annual_rate) / 12.0
            
            # HOA dues (assume $0 for simplification unless property is condo)
            monthly_hoa = 0.0
            property_type = app_data.get('property_type', '')
            if property_type == "Condo":
                monthly_hoa = 200.0  # Assumed monthly HOA
            
            # Total monthly housing payment
            monthly_housing_payment_total = monthly_pi + monthly_property_tax + monthly_hazard_insurance + monthly_mortgage_insurance + monthly_hoa
            
            # Calculate DTI
            total_monthly_obligations = monthly_housing_payment_total + total_monthly_debt_obligations
            debt_to_income_ratio = (total_monthly_obligations / verified_gross_monthly_income) * 100.0
            
            # Calculate DSCR
            debt_service_coverage_ratio = verified_gross_monthly_income / total_monthly_obligations if total_monthly_obligations > 0 else 0.0
            
            result = {
                "application_id": application_id,
                "monthly_principal_and_interest": round(monthly_pi, 2),
                "monthly_property_tax": round(monthly_property_tax, 2),
                "monthly_hazard_insurance": round(monthly_hazard_insurance, 2),
                "monthly_mortgage_insurance": round(monthly_mortgage_insurance, 2),
                "monthly_hoa_dues": round(monthly_hoa, 2),
                "monthly_housing_payment_total": round(monthly_housing_payment_total, 2),
                "total_monthly_obligations": round(total_monthly_obligations, 2),
                "debt_to_income_ratio": round(debt_to_income_ratio, 2),
                "debt_service_coverage_ratio": round(debt_service_coverage_ratio, 2),
                "calculation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "debt_to_income_ratio": round(debt_to_income_ratio, 2),
                "debt_service_coverage_ratio": round(debt_service_coverage_ratio, 2),
                "monthly_housing_payment_total": round(monthly_housing_payment_total, 2)
            })
            
            return result

        # API 11: CalculateProbabilityOfDefault
        def CalculateProbabilityOfDefault(
            application_id: str,
            representative_credit_score: int,
            debt_to_income_ratio: float,
            preliminary_ltv_ratio: float,
            reserve_months_calculated: float,
            employment_years: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str,
            detailed_derogatory_marks_count: int
        ) -> Dict[str, Any]:
            """Calculates probability of default using logistic regression model"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Logistic regression coefficients (institution-calibrated)
            beta_0 = 5.0  # Intercept
            beta_fico = -0.012
            beta_dti = 0.08
            beta_ltv = 0.05
            beta_reserves = -0.15
            beta_employment = -0.05
            beta_property_type = 0.3 if property_type == "Multi-Family" else 0.0
            beta_occupancy = 0.8 if property_occupancy == "Investment Property" else 0.0
            beta_loan_purpose = 0.4 if loan_purpose == "Cash-Out Refinance" else 0.0
            beta_derogs = 0.2
            
            # Calculate z
            z = (beta_0 + 
                beta_fico * representative_credit_score + 
                beta_dti * debt_to_income_ratio + 
                beta_ltv * preliminary_ltv_ratio + 
                beta_reserves * reserve_months_calculated + 
                beta_employment * employment_years + 
                beta_property_type + 
                beta_occupancy + 
                beta_loan_purpose + 
                beta_derogs * detailed_derogatory_marks_count)
            
            # Calculate PD = 1 / (1 + e^-z)
            probability_of_default_score = 1.0 / (1.0 + math.exp(-z)) * 100.0  # Convert to percentage
            
            # Map to credit grade
            if probability_of_default_score <= 2.0:
                credit_score_grade = "A"
            elif probability_of_default_score <= 4.0:
                credit_score_grade = "B"
            elif probability_of_default_score <= 7.0:
                credit_score_grade = "C"
            else:
                credit_score_grade = "D"
            
            result = {
                "application_id": application_id,
                "probability_of_default_score": round(probability_of_default_score, 2),
                "credit_score_grade": credit_score_grade,
                "pd_model_version": "v2.1",
                "calculation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "probability_of_default_score": round(probability_of_default_score, 2),
                "credit_score_grade": credit_score_grade
            })
            
            return result

        # API 12: DetermineFinalInterestRate
        def DetermineFinalInterestRate(
            application_id: str,
            credit_score_grade: str,
            representative_credit_score: int,
            preliminary_ltv_ratio: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str,
            debt_to_income_ratio: float,
            loan_term_months: int
        ) -> Dict[str, Any]:
            """Determines final risk-adjusted interest rate"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Base rate (current market rate)
            base_rate = 6.500
            
            # Credit score adjustment
            if representative_credit_score >= 780:
                credit_adjustment = -0.500
            elif representative_credit_score >= 760:
                credit_adjustment = -0.250
            elif representative_credit_score >= 740:
                credit_adjustment = 0.000
            elif representative_credit_score >= 720:
                credit_adjustment = 0.125
            elif representative_credit_score >= 700:
                credit_adjustment = 0.250
            elif representative_credit_score >= 680:
                credit_adjustment = 0.500
            elif representative_credit_score >= 660:
                credit_adjustment = 0.875
            elif representative_credit_score >= 640:
                credit_adjustment = 1.375
            else:
                credit_adjustment = 2.000
            
            # LTV adjustment
            if preliminary_ltv_ratio <= 60:
                ltv_adjustment = 0.000
            elif preliminary_ltv_ratio <= 70:
                ltv_adjustment = 0.125
            elif preliminary_ltv_ratio <= 80:
                ltv_adjustment = 0.250
            elif preliminary_ltv_ratio <= 85:
                ltv_adjustment = 0.375
            elif preliminary_ltv_ratio <= 90:
                ltv_adjustment = 0.625
            else:
                ltv_adjustment = 1.000
            
            # Property type adjustment
            if property_type == "Single Family":
                property_adjustment = 0.000
            elif property_type == "Condo":
                property_adjustment = 0.250
            elif property_type == "Townhouse":
                property_adjustment = 0.375
            elif property_type == "Multi-Family":
                property_adjustment = 0.750
            else:
                property_adjustment = 0.000
            
            # Occupancy adjustment
            if property_occupancy == "Primary Residence":
                occupancy_adjustment = 0.000
            elif property_occupancy == "Second Home":
                occupancy_adjustment = 0.375
            elif property_occupancy == "Investment Property":
                occupancy_adjustment = 1.250
            else:
                occupancy_adjustment = 0.000
            
            # Loan purpose adjustment
            if loan_purpose == "Purchase":
                purpose_adjustment = 0.000
            elif loan_purpose == "Refinance":
                purpose_adjustment = 0.125
            elif loan_purpose == "Cash-Out Refinance":
                if preliminary_ltv_ratio <= 70:
                    purpose_adjustment = 0.375
                else:
                    purpose_adjustment = 0.750
            else:
                purpose_adjustment = 0.000
            
            # DTI adjustment
            if debt_to_income_ratio <= 36:
                dti_adjustment = 0.000
            elif debt_to_income_ratio <= 40:
                dti_adjustment = 0.125
            elif debt_to_income_ratio <= 43:
                dti_adjustment = 0.250
            else:
                dti_adjustment = 0.500
            
            # Loan term adjustment (15-year gets discount)
            if loan_term_months == 180:
                term_adjustment = -0.375
            else:
                term_adjustment = 0.000
            
            # Calculate final rate
            final_risk_adjusted_interest_rate = (base_rate + credit_adjustment + ltv_adjustment + 
                                                property_adjustment + occupancy_adjustment + 
                                                purpose_adjustment + dti_adjustment + term_adjustment)
            
            result = {
                "application_id": application_id,
                "base_interest_rate": round(base_rate, 3),
                "credit_score_adjustment": round(credit_adjustment, 3),
                "ltv_adjustment": round(ltv_adjustment, 3),
                "property_type_adjustment": round(property_adjustment, 3),
                "occupancy_adjustment": round(occupancy_adjustment, 3),
                "loan_purpose_adjustment": round(purpose_adjustment, 3),
                "dti_adjustment": round(dti_adjustment, 3),
                "loan_term_adjustment": round(term_adjustment, 3),
                "final_risk_adjusted_interest_rate": round(final_risk_adjusted_interest_rate, 3),
                "rate_lock_eligible": True,
                "pricing_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "final_risk_adjusted_interest_rate": round(final_risk_adjusted_interest_rate, 3)
            })
            
            return result

        # API 13: EvaluateCompensatingFactors
        def EvaluateCompensatingFactors(
            application_id: str,
            debt_to_income_ratio: float,
            reserve_months_calculated: float,
            payment_history_quality: str,
            verified_gross_monthly_income: float,
            monthly_housing_payment_total: float,
            total_monthly_debt_obligations: float,
            preliminary_ltv_ratio: float,
            employment_years: float,
            credit_depth_rating: str,
            applicant_address: str
        ) -> Dict[str, Any]:
            """Evaluates compensating factors for marginal applications"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            total_compensating_factor_points = 0
            compensating_factors_list = []
            
            # Reserve months points
            if reserve_months_calculated > 12:
                points = 3
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Reserves >12 months: {points} points")
            elif reserve_months_calculated >= 6:
                points = 2
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Reserves 6-12 months: {points} points")
            elif reserve_months_calculated >= 3:
                points = 1
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Reserves 3-6 months: {points} points")
            
            # Payment history points
            if payment_history_quality in ["Excellent", "Good"]:
                points = 2
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Excellent/Good payment history: {points} points")
            
            # Residual income points (simplified - assume 120% of regional median is $4000/month)
            residual_income = verified_gross_monthly_income - monthly_housing_payment_total - total_monthly_debt_obligations
            regional_median_120_percent = 4000.0
            if residual_income > regional_median_120_percent:
                points = 2
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Strong residual income: {points} points")
            
            # LTV position points
            if preliminary_ltv_ratio <= 70:
                points = 2
                total_compensating_factor_points += points
                compensating_factors_list.append(f"LTV ≤70% (substantial equity): {points} points")
            
            # Employment stability points
            if employment_years > 5.0:
                points = 1
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Employment >5 years: {points} points")
            
            # Credit depth points
            if credit_depth_rating == "Deep":
                points = 1
                total_compensating_factor_points += points
                compensating_factors_list.append(f"Deep credit profile: {points} points")
            
            # Determine if compensating factors are present
            compensating_factors_present = total_compensating_factor_points >= 5
            
            # Determine if sufficient for DTI exception
            sufficient_for_dti_exception = False
            if 43.0 < debt_to_income_ratio <= 45.0 and total_compensating_factor_points >= 5:
                sufficient_for_dti_exception = True
            elif 45.0 < debt_to_income_ratio <= 50.0 and total_compensating_factor_points >= 7:
                sufficient_for_dti_exception = True
            
            result = {
                "application_id": application_id,
                "total_compensating_factor_points": total_compensating_factor_points,
                "compensating_factors_present": compensating_factors_present,
                "compensating_factors_list": compensating_factors_list,
                "sufficient_for_dti_exception": sufficient_for_dti_exception,
                "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "compensating_factors_present": compensating_factors_present,
                "total_compensating_factor_points": total_compensating_factor_points
            })
            
            return result

        # API 14: OrderPropertyAppraisal
        def OrderPropertyAppraisal(
            application_id: str,
            prequalification_decision: str,
            property_address: str,
            property_type: str,
            property_occupancy: str,
            estimated_property_value: float,
            requested_loan_amount: float,
            loan_purpose: str
        ) -> Dict[str, Any]:
            """Orders property appraisal"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            if prequalification_decision == "Not Qualified":
                return {"error": "Cannot order appraisal for Not Qualified applications"}
            
            # Generate appraisal order number
            appraisal_order_number = f"APR-{application_id.split('-')[-1]}-001"
            
            # Determine appraisal form
            if property_type == "Single Family" or property_type == "Townhouse":
                appraisal_form = "URAR Form 1004"
            elif property_type == "Condo":
                appraisal_form = "Form 1073"
            elif property_type == "Multi-Family":
                appraisal_form = "Form 1025"
            else:
                appraisal_form = "URAR Form 1004"
            
            # Calculate expected completion date (10 business days)
            order_timestamp = datetime.utcnow()
            expected_completion_date = order_timestamp + timedelta(days=14)  # Approximate 10 business days
            
            result = {
                "application_id": application_id,
                "appraisal_order_number": appraisal_order_number,
                "appraisal_order_timestamp": order_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "property_address": property_address,
                "appraisal_form_required": appraisal_form,
                "estimated_property_value": round(estimated_property_value, 2),
                "appraisal_purpose": loan_purpose,
                "expected_completion_date": expected_completion_date.strftime("%Y-%m-%d"),
                "appraisal_status": "Ordered"
            }
            
            # Update application data
            update_application_data(application_id, {
                "appraisal_order_number": appraisal_order_number,
                "appraisal_order_timestamp": result["appraisal_order_timestamp"],
                "appraisal_status": "Ordered"
            })
            
            return result

        # API 15: ConductPropertyValuation
        def ConductPropertyValuation(
            application_id: str,
            appraisal_order_number: str,
            property_address: str,
            property_type: str,
            estimated_property_value: float
        ) -> Dict[str, Any]:
            """Conducts property valuation through appraiser site inspection"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Site inspection timestamp
            site_inspection_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Get property details from app data
            property_square_footage = convert_value(app_data.get('property_square_footage'), 'int')
            property_condition_str = app_data.get('property_condition', 'Good')
            comparable_properties_count = convert_value(app_data.get('comparable_properties_count'), 'int')
            
            # Map property condition to standard scale
            condition_mapping = {
                "Excellent": "C1",
                "Good": "C2",
                "Fair": "C3",
                "Poor": "C4"
            }
            property_condition = condition_mapping.get(property_condition_str, "C2")
            
            # Simulate final appraised value (within ±5% of estimated)
            # Use deterministic calculation based on application_id
            app_number = int(application_id.split('-')[-1])
            variance_factor = 1.0 + ((app_number % 11) - 5) / 100.0  # Range: 0.95 to 1.05
            final_appraised_value = estimated_property_value * variance_factor
            
            result = {
                "application_id": application_id,
                "appraisal_order_number": appraisal_order_number,
                "site_inspection_timestamp": site_inspection_timestamp,
                "property_square_footage": property_square_footage,
                "property_condition": property_condition,
                "comparable_properties_count": comparable_properties_count,
                "final_appraised_value": round(final_appraised_value, 2),
                "appraisal_completion_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "appraiser_certification": "Licensed Residential Appraiser"
            }
            
            # Update application data
            update_application_data(application_id, {
                "final_appraised_value": round(final_appraised_value, 2),
                "site_inspection_timestamp": site_inspection_timestamp,
                "property_condition": property_condition
            })
            
            return result

        # API 16: ValidateAppraisalWithAVM
        def ValidateAppraisalWithAVM(
            application_id: str,
            property_address: str,
            final_appraised_value: float,
            property_type: str
        ) -> Dict[str, Any]:
            """Validates appraisal using AVM"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Simulate AVM value (typically within ±8% of appraised value)
            # Use deterministic calculation
            app_number = int(application_id.split('-')[-1])
            avm_variance_factor = 1.0 + ((app_number % 17) - 8) / 100.0  # Range: 0.92 to 1.08
            avm_estimated_value = final_appraised_value * avm_variance_factor
            
            # Calculate variance
            appraisal_avm_variance_percent = ((final_appraised_value - avm_estimated_value) / avm_estimated_value) * 100.0
            
            # Check if second review triggered (variance > ±10%)
            second_review_triggered = abs(appraisal_avm_variance_percent) > 10.0
            
            result = {
                "application_id": application_id,
                "avm_estimated_value": round(avm_estimated_value, 2),
                "final_appraised_value": round(final_appraised_value, 2),
                "appraisal_avm_variance_percent": round(appraisal_avm_variance_percent, 2),
                "second_review_triggered": second_review_triggered,
                "avm_confidence_score": 0.85,
                "validation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "avm_estimated_value": round(avm_estimated_value, 2),
                "appraisal_avm_variance_percent": round(appraisal_avm_variance_percent, 2),
                "second_review_triggered": second_review_triggered
            })
            
            return result

        # API 17: CalculateLoanToValueRatio
        def CalculateLoanToValueRatio(
            application_id: str,
            final_appraised_value: float,
            requested_loan_amount: float,
            estimated_property_value: float,
            loan_purpose: str,
            representative_credit_score: int,
            property_type: str,
            property_occupancy: str
        ) -> Dict[str, Any]:
            """Calculates loan-to-value ratio"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Determine collateral value
            if loan_purpose == "Purchase":
                # Use lesser of appraised value or purchase price
                collateral_value = min(final_appraised_value, estimated_property_value)
            else:
                # Refinance uses appraised value
                collateral_value = final_appraised_value
            
            # Calculate LTV
            loan_to_value_ratio = (requested_loan_amount / collateral_value) * 100.0
            
            # Determine max LTV policy based on credit score, property type, and occupancy
            if property_occupancy == "Primary Residence":
                if property_type == "Single Family":
                    if representative_credit_score >= 740:
                        max_ltv_policy = 95.0
                    elif representative_credit_score >= 700:
                        max_ltv_policy = 90.0
                    elif representative_credit_score >= 680:
                        max_ltv_policy = 85.0
                    else:
                        max_ltv_policy = 80.0
                elif property_type == "Condo":
                    max_ltv_policy = 90.0 if representative_credit_score >= 720 else 85.0
                else:
                    max_ltv_policy = 90.0
            elif property_occupancy == "Second Home":
                max_ltv_policy = 90.0
            else:  # Investment Property
                if property_type == "Multi-Family":
                    max_ltv_policy = 75.0
                else:
                    max_ltv_policy = 85.0
            
            # Check if loan amount adjustment required
            loan_amount_adjustment_required = loan_to_value_ratio > max_ltv_policy
            
            # Calculate adjusted loan amount if needed
            if loan_amount_adjustment_required:
                adjusted_loan_amount = collateral_value * (max_ltv_policy / 100.0)
            else:
                adjusted_loan_amount = requested_loan_amount
            
            result = {
                "application_id": application_id,
                "collateral_value": round(collateral_value, 2),
                "requested_loan_amount": round(requested_loan_amount, 2),
                "loan_to_value_ratio": round(loan_to_value_ratio, 2),
                "max_ltv_policy_threshold": round(max_ltv_policy, 2),
                "loan_amount_adjustment_required": loan_amount_adjustment_required,
                "adjusted_loan_amount": round(adjusted_loan_amount, 2),
                "calculation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "loan_to_value_ratio": round(loan_to_value_ratio, 2),
                "loan_amount_adjustment_required": loan_amount_adjustment_required
            })
            
            return result

        # API 18: RenderFinalLoanDecision
        def RenderFinalLoanDecision(
            application_id: str,
            debt_to_income_ratio: float,
            debt_service_coverage_ratio: float,
            loan_to_value_ratio: float,
            probability_of_default_score: float,
            credit_score_grade: str,
            final_risk_adjusted_interest_rate: float,
            compensating_factors_present: bool,
            property_condition: str,
            bankruptcy_seasoning_compliant: bool,
            foreclosure_history_compliant: bool,
            verified_gross_monthly_income: float,
            requested_loan_amount: float,
            loan_term_months: int,
            application_timestamp: str
        ) -> Dict[str, Any]:
            """Renders final loan decision"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Decision logic
            decline_reasons = []
            
            # Check property condition
            if property_condition in ["C4", "C5"]:
                decline_reasons.append("Property condition requires repairs or is unsuitable")
            
            # Check bankruptcy/foreclosure seasoning
            if not bankruptcy_seasoning_compliant:
                decline_reasons.append("Bankruptcy seasoning period not met")
            
            if not foreclosure_history_compliant:
                decline_reasons.append("Foreclosure seasoning period not met")
            
            # Check credit grade
            if credit_score_grade == "D":
                decline_reasons.append("Credit grade D requires special approval")
            
            # Check DTI
            if debt_to_income_ratio > 50.0:
                decline_reasons.append("DTI exceeds absolute maximum of 50%")
            elif debt_to_income_ratio > 43.0 and not compensating_factors_present:
                decline_reasons.append("DTI exceeds 43% without sufficient compensating factors")
            
            # Check DSCR
            if debt_service_coverage_ratio < 1.0:
                decline_reasons.append("Negative cash flow - DSCR below 1.0")
            
            # Check PD score
            if probability_of_default_score > 10.0:
                decline_reasons.append("Probability of default exceeds acceptable threshold")
            
            # Determine decision
            if len(decline_reasons) > 0:
                final_decision = "Declined"
                conditions_count = 0
                approved_loan_amount = None
                approved_interest_rate = None
                approved_loan_term_months = None
            else:
                # Check if conditions needed
                conditions_count = 0
                conditions_list = []
                
                if debt_to_income_ratio > 43.0 and compensating_factors_present:
                    conditions_count += 1
                    conditions_list.append("Verify compensating factors documentation")
                
                if credit_score_grade == "C":
                    conditions_count += 1
                    conditions_list.append("Additional income verification required")
                
                if loan_to_value_ratio > 80.0:
                    conditions_count += 1
                    conditions_list.append("Mortgage insurance approval required")
                
                if conditions_count > 0:
                    final_decision = "Approved with Conditions"
                else:
                    final_decision = "Approved"
                
                # Get adjusted loan amount if applicable
                loan_amount_adjustment_required = app_data.get('loan_amount_adjustment_required')
                if loan_amount_adjustment_required == 'True':
                    # Would get adjusted amount from app data
                    approved_loan_amount = requested_loan_amount * 0.95  # Simplified
                else:
                    approved_loan_amount = requested_loan_amount
                
                approved_interest_rate = final_risk_adjusted_interest_rate
                approved_loan_term_months = loan_term_months
            
            # Calculate turnaround time
            final_decision_timestamp = datetime.utcnow()
            app_timestamp = datetime.strptime(application_timestamp, "%Y-%m-%dT%H:%M:%SZ")
            turnaround_time_days = (final_decision_timestamp - app_timestamp).days
            
            result = {
                "application_id": application_id,
                "final_decision": final_decision,
                "final_decision_timestamp": final_decision_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "approved_loan_amount": round(approved_loan_amount, 2) if approved_loan_amount else None,
                "approved_interest_rate": round(approved_interest_rate, 3) if approved_interest_rate else None,
                "approved_loan_term_months": approved_loan_term_months,
                "conditions_count": conditions_count,
                "decline_reasons": decline_reasons if final_decision == "Declined" else [],
                "turnaround_time_days": turnaround_time_days,
                "underwriter_notes": "Decision rendered per policy guidelines"
            }
            
            # Update application data
            update_application_data(application_id, {
                "final_decision": final_decision,
                "final_decision_timestamp": result["final_decision_timestamp"],
                "approved_loan_amount": round(approved_loan_amount, 2) if approved_loan_amount else '',
                "approved_interest_rate": round(approved_interest_rate, 3) if approved_interest_rate else '',
                "approved_loan_term_months": approved_loan_term_months if approved_loan_term_months else '',
                "conditions_count": conditions_count,
                "decline_reason": '; '.join(decline_reasons) if decline_reasons else '',
                "turnaround_time_days": turnaround_time_days
            })
            
            return result

        # API 19: ExecuteFairLendingMonitoring
        def ExecuteFairLendingMonitoring(
            application_id: str,
            final_decision: str,
            representative_credit_score: int,
            debt_to_income_ratio: float,
            loan_to_value_ratio: float,
            property_address: str,
            approved_interest_rate: Optional[float] = None,
            conditions_count: Optional[int] = None
        ) -> Dict[str, Any]:
            """Executes fair lending compliance monitoring"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Extract Census tract from property address (simulated)
            # In reality, would geocode to actual Census tract
            zip_code = property_address.split()[-1] if property_address else "00000"
            census_tract = f"TRACT-{zip_code[:5]}"
            
            # Fair lending analysis (simplified)
            # Would compare approval rates and pricing across demographics
            fair_lending_audit_flag = False
            
            # Simulate statistical analysis
            # Check for potential disparate treatment indicators
            audit_findings = []
            
            # Rate variance check (if approved)
            if approved_interest_rate:
                # Compare to expected rate for risk profile
                expected_rate_for_profile = 6.500  # Simplified
                rate_variance = approved_interest_rate - expected_rate_for_profile
                
                if abs(rate_variance) > 1.0:
                    audit_findings.append(f"Rate variance of {rate_variance:.3f}% from expected")
            
            # Conditions count analysis
            if conditions_count and conditions_count > 5:
                audit_findings.append(f"High conditions count: {conditions_count}")
            
            # Decision consistency check
            if final_decision == "Declined" and representative_credit_score > 700:
                audit_findings.append("Decline with score >700 requires review")
            
            # Set audit flag if findings present
            if len(audit_findings) > 0:
                fair_lending_audit_flag = True
            
            result = {
                "application_id": application_id,
                "fair_lending_audit_flag": fair_lending_audit_flag,
                "census_tract": census_tract,
                "decision_outcome": final_decision,
                "pricing_within_tolerance": not fair_lending_audit_flag,
                "audit_findings": audit_findings,
                "monitoring_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "fair_lending_audit_flag": fair_lending_audit_flag
            })
            
            return result

        # API 20: TrackLoanPerformanceMetrics
        def TrackLoanPerformanceMetrics(
            application_id: str,
            application_timestamp: str,
            final_decision: str,
            final_decision_timestamp: str,
            conditions_count: int,
            approved_loan_amount: Optional[float] = None
        ) -> Dict[str, Any]:
            """Tracks loan performance metrics"""
            
            app_data = get_application_data(application_id)
            if not app_data:
                return {"error": "Application not found"}
            
            # Calculate turnaround time
            app_time = datetime.strptime(application_timestamp, "%Y-%m-%dT%H:%M:%SZ")
            decision_time = datetime.strptime(final_decision_timestamp, "%Y-%m-%dT%H:%M:%SZ")
            turnaround_time_days = (decision_time - app_time).days
            
            # First pass yield
            first_pass_yield = conditions_count == 0
            
            # Funded status (simulated - would track actual funding)
            funded_status = False
            funding_date = None
            
            if final_decision in ["Approved", "Approved with Conditions"]:
                # Simulate funding for approved loans
                funded_status = True
                funding_date = (decision_time + timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Performance against targets
            employment_status = app_data.get('employment_status', '')
            if employment_status == "Self-Employed":
                target_days = 35
            else:
                target_days = 25
            
            meets_turnaround_target = turnaround_time_days <= target_days
            
            result = {
                "application_id": application_id,
                "turnaround_time_days": turnaround_time_days,
                "target_turnaround_days": target_days,
                "meets_turnaround_target": meets_turnaround_target,
                "first_pass_yield": first_pass_yield,
                "funded_status": funded_status,
                "funding_date": funding_date,
                "approved_volume": round(approved_loan_amount, 2) if approved_loan_amount else 0.0,
                "tracking_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            
            # Update application data
            update_application_data(application_id, {
                "turnaround_time_days": turnaround_time_days,
                "first_pass_yield": first_pass_yield,
                "funded_status": funded_status,
                "funding_date": funding_date if funding_date else ''
            })
            
            return result
        '''


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
