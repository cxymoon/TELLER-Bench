import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI
def load_yaml_prompt(file_path, sop_file_contents, additional_context, sample_data, api_graph):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text_1"]
    text = text.replace("{{sop_file_contents}}", sop_file_contents)
    text = text.replace("{{additional_context}}", additional_context)
    text = text.replace("{{sample_data}}", sample_data)
    text = text.replace("{{api_graph}}", api_graph)
    return text

def main():
    # === 输入参数 ===
    sop_file_contents = '''
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

        </SOP>'''
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
    yaml_file = "/Loans/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_API_Generation.yaml"
    system_prompt = "You are an SOP-related API documentation generator. Every input parameter of the APIs you generate must be either already present in the sample_data or obtainable through calculations from other API functions. You must implement the APIs in the API dependency graph using their original names."
    api_graph = '''
        <api_dependency_graph>
        apis:
        - api_name: "ValidateApplicantIdentity"
            description: "Performs comprehensive Know Your Customer (KYC) verification by validating government-issued identification documents, checking applicant and co-applicant information against OFAC Specially Designated Nationals lists and other government watchlists, validating Social Security Number patterns and authenticity through SSA Consent Based Verification Service when necessary, and ensuring identity documents match application data fields to prevent identity fraud and comply with USA PATRIOT Act Section 326 Customer Identification Program requirements."
            purpose: "Establish verified identity foundation for all subsequent credit and underwriting processes, ensuring the institution is lending to legitimate borrowers and maintaining compliance with anti-money laundering and counter-terrorism financing regulations."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_name"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_ssn"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_dob"
                data_type: "date"
                source: "user_input"
                required: true
            - name: "applicant_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "co_applicant_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "co_applicant_ssn"
                data_type: "string"
                source: "user_input"
                required: false
            output_parameters:
            - name: "identity_verification_status"
                data_type: "enum"
                destination: "other_api"
            - name: "ofac_clearance_status"
                data_type: "boolean"
                destination: "other_api"
            - name: "verified_applicant_name"
                data_type: "string"
                destination: "other_api"
            - name: "verified_applicant_ssn"
                data_type: "string"
                destination: "other_api"
            - name: "verified_applicant_dob"
                data_type: "date"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "ValidateEmploymentAndIncome"
            description: "Analyzes uploaded income documentation including pay stubs, W-2 forms, and tax returns to calculate preliminary gross monthly income, validates employment status and employer information against document data, performs income averaging for self-employed applicants using Schedule C or K-1 distributions with add-backs for non-cash deductions, flags year-over-year income declines exceeding 10% for underwriter attention, and establishes initial income figures for pre-qualification debt-to-income calculations."
            purpose: "Provide preliminary income validation to support pre-qualification analysis and maximum loan amount calculations, enabling early screening before resource-intensive deep underwriting verification processes."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "employment_status"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "employer_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "employment_years"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "gross_monthly_income"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "additional_income_monthly"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "pay_stubs_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "w2_forms_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "tax_returns_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: false
            output_parameters:
            - name: "preliminary_gross_monthly_income"
                data_type: "float"
                destination: "other_api"
            - name: "income_validation_status"
                data_type: "enum"
                destination: "other_api"
            - name: "income_stability_flag"
                data_type: "boolean"
                destination: "other_api"
            - name: "employment_verification_required"
                data_type: "boolean"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "ValidateLiquidAssets"
            description: "Reviews uploaded bank statements covering two consecutive months to verify declared liquid assets and retirement accounts, identifies and flags large deposits exceeding 50% of monthly income or cumulative deposits exceeding 100% of monthly income requiring source documentation, calculates available reserves after accounting for down payment and estimated closing costs, determines reserve months by dividing remaining liquid assets by proposed total housing payment, and applies appropriate discount factors to retirement accounts based on applicant age to reflect tax and penalty implications."
            purpose: "Verify applicant financial capacity to close the transaction and maintain adequate reserves, supporting pre-qualification approval and identifying potential red flags such as undisclosed borrowed funds or fraudulent asset documentation."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "liquid_assets"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "retirement_accounts"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "bank_statements_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "down_payment_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "applicant_dob"
                data_type: "date"
                source: "user_input"
                required: true
            output_parameters:
            - name: "verified_liquid_assets"
                data_type: "float"
                destination: "other_api"
            - name: "large_deposits_flagged"
                data_type: "boolean"
                destination: "other_api"
            - name: "reserve_months_preliminary"
                data_type: "float"
                destination: "other_api"
            - name: "asset_verification_status"
                data_type: "enum"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "PullCreditReports"
            description: "Executes tri-bureau credit report acquisition by transmitting applicant and co-applicant verified identity information to institution's credit bureau API interfaces connected to Equifax, Experian, and TransUnion repositories, retrieves in-file credit reports with FICO scores calculated using mortgage-specific scoring models (FICO Score 2, 5, and 4 respectively), extracts and populates credit score fields for each bureau, analyzes credit report narrative to count derogatory marks including collections, charge-offs, public records, late payments, identifies bankruptcy history and foreclosure history, and calculates total monthly debt obligations by summing all recurring payment obligations shown on tradelines."
            purpose: "Obtain comprehensive credit profile data required for credit score determination, debt-to-income calculation, risk assessment, and pricing decisions, establishing the foundation for all credit-based underwriting judgments."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "verified_applicant_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "verified_applicant_ssn"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "verified_applicant_dob"
                data_type: "date"
                source: "other_api"
                required: true
            - name: "applicant_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "co_applicant_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "co_applicant_ssn"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "credit_consent_signed"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "credit_consent_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            output_parameters:
            - name: "credit_pull_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "fico_score_equifax"
                data_type: "integer"
                destination: "other_api"
            - name: "fico_score_experian"
                data_type: "integer"
                destination: "other_api"
            - name: "fico_score_transunion"
                data_type: "integer"
                destination: "other_api"
            - name: "derogatory_marks_count"
                data_type: "integer"
                destination: "other_api"
            - name: "bankruptcy_history"
                data_type: "boolean"
                destination: "other_api"
            - name: "total_monthly_debt_obligations"
                data_type: "float"
                destination: "other_api"
            - name: "credit_report_full_data"
                data_type: "object"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ValidateApplicantIdentity"
                dependency_type: "data_dependency"
                dependency_reason: "Requires verified applicant name, SSN, and date of birth to submit accurate credit bureau queries and ensure credit reports are pulled for the correct individuals, preventing misidentification and ensuring FCRA compliance."
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "CalculateRepresentativeCreditScore"
            description: "Applies the FICO Score Tri-Merge Middle Score Methodology by ranking the three bureau credit scores (fico_score_equifax, fico_score_experian, fico_score_transunion) in ascending order and selecting the median value as the representative score, handling cases where only two scores are available by selecting the lower value, or using a single score if only one bureau provides data, and for co-applicant scenarios, determining each individual's representative score then selecting the lower of the two as the qualifying credit score used for all subsequent underwriting policy evaluation, pricing decisions, and mortgage insurance premium calculations."
            purpose: "Establish the single authoritative credit score that will govern all credit policy thresholds, risk-based pricing adjustments, and underwriting decisions, ensuring consistent application of credit standards across the institution's mortgage portfolio."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "fico_score_equifax"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "fico_score_experian"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "fico_score_transunion"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "co_applicant_fico_equifax"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "co_applicant_fico_experian"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "co_applicant_fico_transunion"
                data_type: "integer"
                source: "other_api"
                required: false
            output_parameters:
            - name: "representative_credit_score"
                data_type: "integer"
                destination: "other_api"
            - name: "credit_score_methodology_applied"
                data_type: "string"
                destination: "storage"
            - name: "credit_score_calculation_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "PullCreditReports"
                dependency_type: "data_dependency"
                dependency_reason: "Requires the three bureau FICO scores (fico_score_equifax, fico_score_experian, fico_score_transunion) from credit report pull to perform tri-merge middle score calculation, as the methodology cannot execute without the raw score inputs from each credit repository."
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "ExecutePrequalificationAnalysis"
            description: "Performs systematic pre-qualification screening by comparing representative credit score against institutional minimum threshold of 620 for conventional products, calculates preliminary debt-to-income ratio using estimated housing payment derived from requested loan amount and proxy interest rate based on credit score tier plus total monthly debt obligations divided by gross monthly income, evaluates calculated DTI against policy thresholds to generate prequalification_decision classification of Qualified (DTI ≤43%), Conditionally Qualified (DTI 43.01-50%), or Not Qualified (DTI >50% or FICO <620), calculates maximum loan capacity (max_loan_amount_prequalified) by solving DTI equation for loan amount that achieves 43% ratio, determines estimated_interest_rate_prequalification by applying credit score band and property type pricing adjustments to current market base rates, and generates pre-qualification letter suitable for property search and seller presentation."
            purpose: "Provide rapid preliminary eligibility determination and loan capacity guidance to applicants before property identification and resource-intensive underwriting processes, enabling efficient property search targeting and early identification of unqualified applicants to optimize operational efficiency."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "preliminary_gross_monthly_income"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "total_monthly_debt_obligations"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "loan_term_months"
                data_type: "integer"
                source: "user_input"
                required: true
            - name: "estimated_property_value"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "down_payment_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "loan_purpose"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "prequalification_decision"
                data_type: "enum"
                destination: "other_api"
            - name: "max_loan_amount_prequalified"
                data_type: "float"
                destination: "other_api"
            - name: "estimated_interest_rate_prequalification"
                data_type: "float"
                destination: "other_api"
            - name: "preliminary_dti_ratio"
                data_type: "float"
                destination: "other_api"
            - name: "preliminary_ltv_ratio"
                data_type: "float"
                destination: "other_api"
            - name: "prequalification_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CalculateRepresentativeCreditScore"
                dependency_type: "data_dependency"
                dependency_reason: "Requires representative_credit_score to compare against minimum FICO threshold of 620 and to determine appropriate interest rate pricing tier for preliminary DTI and loan capacity calculations."
            - dependent_api: "ValidateEmploymentAndIncome"
                dependency_type: "data_dependency"
                dependency_reason: "Requires preliminary_gross_monthly_income to calculate debt-to-income ratio and determine maximum affordable loan amount through capacity analysis."
            - dependent_api: "PullCreditReports"
                dependency_type: "data_dependency"
                dependency_reason: "Requires total_monthly_debt_obligations extracted from credit report tradelines to calculate accurate debt-to-income ratio including both housing payment and all recurring debt obligations."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "VerifyEmploymentAndIncome"
            description: "Conducts comprehensive income verification through Verification of Employment (VOE) telephone calls or written requests to employers using independently verified contact information to confirm current employment status, job title, hire date, base salary or hourly wage, and employment continuity expectations for Full-Time employed applicants, performs detailed tax return analysis for Self-Employed applicants by examining IRS Form 1040 with Schedule C, E, K-1 distributions over two years to calculate adjusted income with add-backs for depreciation and depletion while subtracting one-time income events, applies two-year income averaging methodology, flags year-over-year income declines exceeding 10% requiring documented justification, verifies bonus and commission income through employer confirmation of continuation likelihood, validates overtime income sustainability, and produces verified_gross_monthly_income representing underwriter-certified income figure used for final debt-to-income ratio calculations."
            purpose: "Establish definitive verified income figure through third-party validation and forensic document analysis to replace preliminary self-declared income, ensuring lending decisions are based on accurate capacity assessments and preventing income fraud that could lead to defaults and losses."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "prequalification_decision"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "employment_status"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "employer_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "employment_years"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "gross_monthly_income"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "additional_income_monthly"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "pay_stubs_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "w2_forms_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "tax_returns_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: false
            output_parameters:
            - name: "verified_gross_monthly_income"
                data_type: "float"
                destination: "other_api"
            - name: "income_verification_method"
                data_type: "string"
                destination: "storage"
            - name: "employment_stability_rating"
                data_type: "enum"
                destination: "other_api"
            - name: "income_trending_direction"
                data_type: "enum"
                destination: "other_api"
            - name: "document_verification_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ExecutePrequalificationAnalysis"
                dependency_type: "state_dependency"
                dependency_reason: "Only proceeds to resource-intensive deep income verification after prequalification_decision indicates Qualified or Conditionally Qualified status, preventing wasted effort on applications that fail preliminary screening and serving as a business process gate."
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "VerifyAssetsAndReserves"
            description: "Executes comprehensive asset verification by examining two consecutive months of bank statements for all checking, savings, and money market accounts to confirm ending balances match or exceed declared liquid_assets, investigates all individual deposits exceeding 50% of monthly income or cumulative deposits exceeding 100% of monthly income to identify potential undisclosed borrowed funds requiring source documentation such as asset transfer statements or gift letters with donor verification, applies appropriate discount factors to retirement_accounts (25% for under age 59.5, 10% for age 59.5 and above) to reflect tax and penalty implications, calculates net available assets after subtracting down_payment_amount and estimated closing costs, determines reserve months by dividing remaining assets by proposed total housing payment, and evaluates reserve adequacy against policy requirements (2-12 months depending on property type, occupancy, and LTV) while assigning compensating factor points based on reserve levels exceeding minimum thresholds."
            purpose: "Verify applicant financial capacity to close the transaction and maintain adequate reserves for financial contingencies, detect potential fraud through large deposit analysis, and establish reserve-based compensating factors that may offset elevated DTI or other risk indicators in marginal credit profiles."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "prequalification_decision"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "liquid_assets"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "retirement_accounts"
                data_type: "float"
                source: "user_input"
                required: false
            - name: "bank_statements_uploaded"
                data_type: "boolean"
                source: "user_input"
                required: true
            - name: "down_payment_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "applicant_dob"
                data_type: "date"
                source: "user_input"
                required: true
            - name: "verified_gross_monthly_income"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "verified_liquid_assets_final"
                data_type: "float"
                destination: "other_api"
            - name: "verified_retirement_accounts_discounted"
                data_type: "float"
                destination: "other_api"
            - name: "reserve_months_calculated"
                data_type: "float"
                destination: "other_api"
            - name: "large_deposit_explanations_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "reserve_compensating_factor_points"
                data_type: "integer"
                destination: "other_api"
            - name: "asset_verification_complete_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ExecutePrequalificationAnalysis"
                dependency_type: "state_dependency"
                dependency_reason: "Only proceeds to detailed asset verification after prequalification_decision indicates preliminary approval, serving as a business process gate to prevent unnecessary resource expenditure on likely-to-decline applications."
            - dependent_api: "VerifyEmploymentAndIncome"
                dependency_type: "data_dependency"
                dependency_reason: "Requires verified_gross_monthly_income to calculate reserve months by dividing available assets by total housing payment, which is derived from verified income-based payment capacity, and to flag large deposits as percentage of verified monthly income."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "AnalyzeCreditHistory"
            description: "Conducts forensic analysis of tri-bureau credit report narrative sections by examining each tradeline's 24-month payment history to identify and count derogatory marks including 30-day late payments (1 point), 60-day late payments (2 points), 90+ day late payments (3 points), collections accounts with distinction between medical collections under $500 (may be disregarded per GSE guidelines) and non-medical collections requiring payment or acceptance justification, charge-offs, public records including tax liens and civil judgments, reviews foreclosure and short sale history to verify compliance with institutional seasoning requirements (7 years standard, 3 years with extenuating circumstances), validates bankruptcy_history and ensures sufficient time has elapsed since discharge (4 years Chapter 7, 2 years Chapter 13), analyzes credit inquiry patterns in preceding 90 days to identify potential undisclosed debt, and produces comprehensive credit risk assessment documenting payment behavior patterns, credit depth, and overall creditworthiness that informs underwriting judgment and compensating factor evaluation."
            purpose: "Provide detailed credit behavior analysis beyond numeric credit scores to identify character risk indicators, verify compliance with credit history seasoning requirements, and document credit strengths or weaknesses that influence final credit decisions and exception approvals."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "credit_report_full_data"
                data_type: "object"
                source: "other_api"
                required: true
            - name: "derogatory_marks_count"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "bankruptcy_history"
                data_type: "boolean"
                source: "other_api"
                required: true
            output_parameters:
            - name: "detailed_derogatory_marks_count"
                data_type: "integer"
                destination: "other_api"
            - name: "payment_history_quality"
                data_type: "enum"
                destination: "other_api"
            - name: "bankruptcy_seasoning_compliant"
                data_type: "boolean"
                destination: "other_api"
            - name: "foreclosure_history_compliant"
                data_type: "boolean"
                destination: "other_api"
            - name: "credit_depth_rating"
                data_type: "enum"
                destination: "other_api"
            - name: "undisclosed_debt_risk_flag"
                data_type: "boolean"
                destination: "other_api"
            - name: "credit_analysis_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "PullCreditReports"
                dependency_type: "data_dependency"
                dependency_reason: "Requires full credit_report_full_data object containing detailed tradeline information, payment histories, public records, and inquiry data to perform comprehensive forensic analysis beyond the summary metrics extracted during initial credit pull."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "CalculateDebtRatios"
            description: "Computes precise debt-to-income ratio and debt service coverage ratio using verified financial data by first calculating exact monthly housing payment using mortgage payment formula P=L[c(1+c)^n]/[(1+c)^n-1] where P is principal and interest, L is approved loan amount, c is monthly interest rate, n is loan term in months, then adding property taxes obtained from tax assessor records, hazard insurance quoted by carrier, mortgage insurance premium calculated from MI rate tables based on LTV and credit score if LTV exceeds 80%, and HOA dues from association documents, summing this total housing payment with total_monthly_debt_obligations from credit report, dividing by verified_gross_monthly_income and multiplying by 100 to produce debt_to_income_ratio with precision to two decimal places, and calculating debt_service_coverage_ratio as verified_gross_monthly_income divided by sum of housing payment plus debt obligations with precision to two decimal places, comparing both ratios against institutional policy thresholds to determine compliance with QM standards (DTI ≤43%) and portfolio requirements (DSCR ≥1.15) while flagging marginal cases requiring compensating factor analysis."
            purpose: "Establish definitive debt burden metrics using verified rather than preliminary data to ensure lending decisions accurately reflect borrower payment capacity, support Qualified Mortgage compliance under 12 CFR 1026.43, and identify applications requiring compensating factor justification for policy exceptions."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "verified_gross_monthly_income"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "total_monthly_debt_obligations"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "loan_term_months"
                data_type: "integer"
                source: "user_input"
                required: true
            - name: "estimated_interest_rate_prequalification"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "property_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "estimated_property_value"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "preliminary_ltv_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            output_parameters:
            - name: "debt_to_income_ratio"
                data_type: "float"
                destination: "other_api"
            - name: "debt_service_coverage_ratio"
                data_type: "float"
                destination: "other_api"
            - name: "monthly_housing_payment_total"
                data_type: "float"
                destination: "other_api"
            - name: "dti_policy_compliance_status"
                data_type: "enum"
                destination: "other_api"
            - name: "dscr_policy_compliance_status"
                data_type: "enum"
                destination: "other_api"
            - name: "debt_ratio_calculation_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "VerifyEmploymentAndIncome"
                dependency_type: "data_dependency"
                dependency_reason: "Requires verified_gross_monthly_income (denominator in DTI and DSCR formulas) to calculate accurate debt ratios using underwriter-validated income figures rather than applicant self-declarations, ensuring ratio calculations reflect true payment capacity."
            - dependent_api: "PullCreditReports"
                dependency_type: "data_dependency"
                dependency_reason: "Requires total_monthly_debt_obligations (component of numerator in DTI and DSCR formulas) extracted from credit report tradelines to include all recurring debt payments in ratio calculations."
            - dependent_api: "ExecutePrequalificationAnalysis"
                dependency_type: "data_dependency"
                dependency_reason: "Uses estimated_interest_rate_prequalification to calculate principal and interest payment component of housing payment, which is refined from preliminary pricing but not yet finalized pending appraisal completion and final rate determination."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "CalculateProbabilityOfDefault"
            description: "Executes institution's proprietary multivariate logistic regression model to quantify default risk by computing probability_of_default_score as PD = 1/(1+e^-z) where z = β₀ + β₁(FICO) + β₂(DTI) + β₃(LTV) + β₄(Reserves) + β₅(Employment_Years) + β₆(Property_Type_Indicator) + β₇(Occupancy_Indicator) + β₈(Loan_Purpose_Indicator) + β₉(Derogatory_Marks) with institution-calibrated beta coefficients derived from historical loan performance data, inputs representative_credit_score, debt_to_income_ratio, preliminary or final LTV ratio, reserve_months_calculated, employment_years, property_type and property_occupancy dummy variables, loan_purpose indicator, and detailed_derogatory_marks_count to produce a probability of default percentage with precision to two decimal places representing statistical likelihood of 90+ day delinquency within 36 months, maps the PD score to credit_score_grade using thresholds of ≤2.0% for Grade A, 2.01-4.0% for Grade B, 4.01-7.0% for Grade C, and >7.0% for Grade D requiring special review, and documents the model inputs and calculation methodology for audit trail and regulatory examination purposes."
            purpose: "Provide quantitative risk assessment that translates qualitative credit characteristics into a standardized probability metric enabling consistent risk-based pricing, capital allocation under Basel III frameworks, portfolio risk aggregation, and data-driven credit decision support while maintaining model transparency and auditability."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "debt_to_income_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "preliminary_ltv_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "reserve_months_calculated"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "employment_years"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "loan_purpose"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "detailed_derogatory_marks_count"
                data_type: "integer"
                source: "other_api"
                required: true
            output_parameters:
            - name: "probability_of_default_score"
                data_type: "float"
                destination: "other_api"
            - name: "credit_score_grade"
                data_type: "enum"
                destination: "other_api"
            - name: "pd_model_version"
                data_type: "string"
                destination: "storage"
            - name: "pd_calculation_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "model_input_parameters"
                data_type: "object"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CalculateRepresentativeCreditScore"
                dependency_type: "data_dependency"
                dependency_reason: "Requires representative_credit_score as primary independent variable (β₁ coefficient) in PD logistic regression model, as credit score is the strongest predictor of default probability in mortgage lending."
            - dependent_api: "CalculateDebtRatios"
                dependency_type: "data_dependency"
                dependency_reason: "Requires debt_to_income_ratio as critical independent variable (β₂ coefficient) reflecting borrower payment capacity and default risk, as higher DTI correlates with elevated default probability."
            - dependent_api: "VerifyAssetsAndReserves"
                dependency_type: "data_dependency"
                dependency_reason: "Requires reserve_months_calculated as independent variable (β₄ coefficient) reflecting financial cushion available to absorb income disruptions, as higher reserves reduce default probability through enhanced financial resilience."
            - dependent_api: "AnalyzeCreditHistory"
                dependency_type: "data_dependency"
                dependency_reason: "Requires detailed_derogatory_marks_count as independent variable (β₉ coefficient) capturing character risk beyond credit score, as recent derogatory credit behavior indicates elevated default propensity."
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "DetermineFinalInterestRate"
            description: "Applies multi-dimensional risk-based pricing matrix to calculate final_risk_adjusted_interest_rate by retrieving base interest rate indexed to current 30-year Freddie Mac Primary Mortgage Market Survey rate or institution's cost of funds benchmark, then applying additive pricing adjustments across multiple risk dimensions including credit score adjustments ranging from -0.500% for FICO ≥780 to +2.000% for FICO 620-639, LTV ratio adjustments ranging from +0.000% for LTV ≤60% to +1.000% for LTV 90.01-95%, property type adjustments of +0.000% for Single Family to +1.000% for three-to-four unit Multi-Family, occupancy adjustments of +0.000% for Primary Residence to +1.500% for Investment Property multi-family, loan purpose adjustments of +0.000% for Purchase to +0.750% for Cash-Out Refinance LTV >70%, and DTI adjustments of +0.000% for DTI ≤36% to +0.500% for DTI 43.01-45%, summing all applicable adjustments algebraically to produce final interest rate expressed to three decimal places, and documenting the pricing breakdown for rate lock confirmation and Closing Disclosure preparation, with the rate subject to refinement after final_appraised_value and loan_to_value_ratio are definitively established post-appraisal."
            purpose: "Establish risk-commensurate interest rate pricing that ensures the institution earns appropriate return for credit, collateral, and concentration risks while maintaining competitive positioning in the mortgage market and supporting secondary market sale eligibility or portfolio retention profitability targets."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "credit_score_grade"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "preliminary_ltv_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "loan_purpose"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "debt_to_income_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "loan_term_months"
                data_type: "integer"
                source: "user_input"
                required: true
            output_parameters:
            - name: "final_risk_adjusted_interest_rate"
                data_type: "float"
                destination: "other_api"
            - name: "base_interest_rate"
                data_type: "float"
                destination: "storage"
            - name: "pricing_adjustments_detail"
                data_type: "object"
                destination: "storage"
            - name: "rate_calculation_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "rate_lock_eligible"
                data_type: "boolean"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "CalculateProbabilityOfDefault"
                dependency_type: "data_dependency"
                dependency_reason: "Requires credit_score_grade (A/B/C/D) mapped from PD score to determine appropriate risk tier for pricing matrix application, as credit grade drives base pricing adjustments and establishes fundamental risk-return relationship."
            - dependent_api: "CalculateDebtRatios"
                dependency_type: "data_dependency"
                dependency_reason: "Requires debt_to_income_ratio to apply DTI-based pricing adjustments, as elevated DTI ratios approaching or exceeding 43% warrant rate premiums to compensate for reduced payment capacity and elevated default risk."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "EvaluateCompensatingFactors"
            description: "Conducts systematic compensating factors analysis for applications with elevated risk indicators by evaluating and scoring positive offsetting characteristics including reserve months (3 points for >12 months, 2 points for 6-12 months, 1 point for 3-6 months), housing payment history assessment (2 points for 12+ months of equal or higher payment without 30-day lates), residual income analysis (2 points for residual exceeding 120% of regional HUD median income after all obligations), LTV position evaluation (2 points for LTV ≤70% representing substantial equity), employment stability assessment (1 point for employment_years >5.0 in same field), and credit depth review (1 point for 3+ revolving accounts with limits >$5,000 and utilization <30%), summing compensating factor points to determine if aggregate score meets threshold requirements of ≥5 points for DTI 43-45% exceptions or ≥7 points for DTI 45-50% exceptions, documenting each claimed compensating factor with quantified supporting evidence in narrative form to justify policy exception requests to Chief Credit Officer, and populating compensating_factors_present Boolean field based on threshold achievement to inform final underwriting decision logic regarding approval feasibility for marginal applications."
            purpose: "Provide structured methodology for evaluating positive borrower characteristics that partially offset elevated DTI ratios, marginal credit scores, or limited employment tenure, enabling appropriate credit access for borrowers with non-traditional but demonstrably strong credit profiles while maintaining sound risk management through objective, quantifiable compensating factor criteria."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "debt_to_income_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "reserve_months_calculated"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "payment_history_quality"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "verified_gross_monthly_income"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "monthly_housing_payment_total"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "total_monthly_debt_obligations"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "preliminary_ltv_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "employment_years"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "credit_depth_rating"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "applicant_address"
                data_type: "string"
                source: "user_input"
                required: true
            output_parameters:
            - name: "compensating_factors_present"
                data_type: "boolean"
                destination: "other_api"
            - name: "total_compensating_factor_points"
                data_type: "integer"
                destination: "other_api"
            - name: "compensating_factors_detail"
                data_type: "object"
                destination: "storage"
            - name: "exception_approval_eligible"
                data_type: "boolean"
                destination: "other_api"
            - name: "compensating_factor_analysis_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CalculateDebtRatios"
                dependency_type: "data_dependency"
                dependency_reason: "Requires debt_to_income_ratio to determine if compensating factor analysis is triggered (DTI >43%) and to assess the magnitude of DTI exception requiring correspondingly higher compensating factor point thresholds."
            - dependent_api: "VerifyAssetsAndReserves"
                dependency_type: "data_dependency"
                dependency_reason: "Requires reserve_months_calculated to assign reserve-based compensating factor points (3 points for >12 months, 2 for 6-12, 1 for 3-6), as reserves represent the strongest compensating factor offsetting payment capacity concerns."
            - dependent_api: "AnalyzeCreditHistory"
                dependency_type: "data_dependency"
                dependency_reason: "Requires payment_history_quality and credit_depth_rating to assign housing payment history points (2 points for 12+ months of comparable payment without lates) and credit depth points (1 point for sophisticated credit management), reflecting character strengths."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "OrderPropertyAppraisal"
            description: "Initiates property valuation process by transmitting appraisal order to independent Appraisal Management Company or internal appraisal desk with comprehensive property information including property_address, property_type classification, property_occupancy status, estimated_property_value from purchase contract or applicant estimate, specific appraisal form requirements (URAR Form 1004 for single-family, Form 1073 for condos, Form 1025 for 2-4 units), and special instructions for unique property characteristics or market conditions, recording appraisal_order_timestamp to initiate 10-business-day completion expectation per service level agreement, assigning appraiser_id for the licensed or certified appraiser selected by AMC based on geographic competency requirements, documenting appraisal_company name for vendor management tracking, and ensuring appraiser independence compliance with Dodd-Frank Act provisions in 12 CFR 1026.42 prohibiting lender influence on valuation, with the order transmission enabling parallel workflow execution where appraisal proceeds concurrently with underwriting verification activities to optimize overall turnaround time."
            purpose: "Trigger property valuation workflow early in the underwriting process to enable parallel execution with income and credit verification activities, reducing overall cycle time while ensuring appraiser independence and obtaining collateral value assessment required for LTV calculation and loan amount support determination."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "prequalification_decision"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "property_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "estimated_property_value"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "loan_purpose"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "appraisal_order_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "appraiser_id"
                data_type: "string"
                destination: "storage"
            - name: "appraisal_company"
                data_type: "string"
                destination: "storage"
            - name: "appraisal_order_number"
                data_type: "string"
                destination: "other_api"
            - name: "expected_completion_date"
                data_type: "date"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ExecutePrequalificationAnalysis"
                dependency_type: "state_dependency"
                dependency_reason: "Only orders appraisal after prequalification_decision indicates Qualified or Conditionally Qualified status to avoid incurring appraisal costs ($500-$700) for applications likely to be declined based on credit or capacity deficiencies, serving as a business process gate and cost control measure."
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "ConductPropertyValuation"
            description: "Executes comprehensive property appraisal by licensed appraiser conducting site inspection to measure property_square_footage using external perimeter methodology, assessing property_condition using standardized C1-C5 rating scale (C1=new/renovated, C2=well-maintained, C3=normal wear, C4=deferred maintenance, C5=substantial deficiencies), photographing interior rooms and exterior elevations for report documentation, identifying and analyzing minimum three closed sales of comparable properties within 12 months and 1 mile radius matching subject property characteristics, constructing adjustment grid quantifying dollar differences between subject and each comparable across dimensions of sale date, location, site characteristics, gross living area, room count, basement, garage, condition with individual adjustments typically ≤10% of comparable sale price and cumulative gross adjustments ≤25%, reconciling adjusted comparable values into final opinion of market value with explanation of weighting factors applied to each comparable's indicated value, recording site_inspection_timestamp and comparable_properties_count, completing URAR Form 1004 with all required sections including neighborhood description, site description, improvements description, sales comparison approach analysis, cost approach if applicable, income approach for 2-4 units, and appraiser certifications of USPAP compliance and independence, producing final_appraised_value rounded to nearest $1,000 representing appraiser's opinion of market value supported by comparable sales evidence."
            purpose: "Provide independent third-party assessment of property market value to support loan amount determination, calculate loan-to-value ratio for collateral risk assessment, ensure property condition is acceptable for financing without required repairs exceeding borrower capacity, and generate appraisal report suitable for GSE delivery, portfolio retention, or regulatory examination."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "appraisal_order_number"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "property_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "estimated_property_value"
                data_type: "float"
                source: "user_input"
                required: true
            output_parameters:
            - name: "site_inspection_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "property_square_footage"
                data_type: "integer"
                destination: "storage"
            - name: "property_condition"
                data_type: "enum"
                destination: "other_api"
            - name: "comparable_properties_count"
                data_type: "integer"
                destination: "storage"
            - name: "final_appraised_value"
                data_type: "float"
                destination: "other_api"
            - name: "appraisal_report_pdf"
                data_type: "binary"
                destination: "storage"
            - name: "appraisal_completion_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "OrderPropertyAppraisal"
                dependency_type: "temporal_dependency"
                dependency_reason: "Cannot conduct property valuation until appraisal order is transmitted to AMC, appraiser is assigned, and property access is scheduled, representing sequential workflow where order placement precedes execution by external third party."
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ValidateAppraisalWithAVM"
            description: "Executes automated appraisal quality control by submitting property_address to institution's Automated Valuation Model platform (CoreLogic, Freddie Mac Home Value Explorer, or proprietary model) which returns avm_estimated_value derived from statistical analysis of historical sales data, property characteristics from public records, hedonic pricing models, and market condition indices along with confidence score and forecast standard deviation, calculates appraisal_avm_variance_percent using formula [(final_appraised_value - avm_estimated_value) / avm_estimated_value] × 100 with precision to two decimal places, compares absolute variance value against 10.00% tolerance threshold established in institutional collateral risk management policy, sets second_review_triggered Boolean to true if variance exceeds ±10.00% indicating potential appraisal quality issue requiring secondary field review appraisal by different independent appraiser, or sets to false if variance is within tolerance indicating acceptable appraisal reliability, documents variance analysis and AVM confidence metrics in loan file for audit trail, and enables underwriter to proceed with final_appraised_value acceptance or initiate additional valuation review workflow based on variance magnitude and AVM confidence level."
            purpose: "Provide independent statistical validation of manual appraisal to detect potential valuation inflation, appraiser bias, or comparable selection errors that could result in excessive loan-to-value ratios and elevated collateral risk, ensuring institution's collateral valuations are reliable and supportable in loss scenarios."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "property_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "final_appraised_value"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "avm_estimated_value"
                data_type: "float"
                destination: "other_api"
            - name: "appraisal_avm_variance_percent"
                data_type: "float"
                destination: "other_api"
            - name: "second_review_triggered"
                data_type: "boolean"
                destination: "other_api"
            - name: "avm_confidence_score"
                data_type: "float"
                destination: "storage"
            - name: "avm_validation_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ConductPropertyValuation"
                dependency_type: "data_dependency"
                dependency_reason: "Requires final_appraised_value from completed appraisal report to calculate variance percentage against AVM estimate, as the comparison between manual appraisal and statistical model is the core quality control mechanism that cannot execute without the manual valuation input."
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "CalculateLoanToValueRatio"
            description: "Computes definitive loan-to-value ratio using validated final_appraised_value and requested_loan_amount by applying formula LTV = (requested_loan_amount / MIN(final_appraised_value, purchase_price)) × 100 for purchase transactions where the lesser of appraised value or contract price establishes collateral value per prudential lending standards, or LTV = (requested_loan_amount / final_appraised_value) × 100 for refinance transactions where no purchase price exists, expressing result as percentage with precision to two decimal places, comparing calculated loan_to_value_ratio against institutional policy thresholds stratified by credit score tier, property type, and occupancy status (80.00% for no-MI products, 85.00% standard MI, 90.00% expanded MI, 95.00% maximum portfolio), determining LTV policy compliance status and identifying whether mortgage insurance is required if LTV exceeds 80.00%, calculating Combined LTV (CLTV) if subordinate financing exists by adding first and second lien amounts before dividing by value, and updating debt_to_income_ratio calculation if loan amount requires reduction to meet LTV constraints, ensuring final loan amount simultaneously satisfies both DTI capacity limits and LTV collateral limits while documenting any adjustments required to achieve policy compliance."
            purpose: "Establish final collateral risk metric using validated appraised value to determine maximum supportable loan amount, calculate mortgage insurance requirements, validate that requested loan amount is supportable by collateral value, and enable final underwriting decision by confirming both payment capacity (DTI) and collateral coverage (LTV) are within acceptable risk parameters."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "final_appraised_value"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "estimated_property_value"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "loan_purpose"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "property_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "property_occupancy"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "loan_to_value_ratio"
                data_type: "float"
                destination: "other_api"
            - name: "ltv_policy_compliance_status"
                data_type: "enum"
                destination: "other_api"
            - name: "mortgage_insurance_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "maximum_loan_amount_by_ltv"
                data_type: "float"
                destination: "other_api"
            - name: "loan_amount_adjustment_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "ltv_calculation_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ValidateAppraisalWithAVM"
                dependency_type: "data_dependency"
                dependency_reason: "Requires validated final_appraised_value that has passed AVM quality control check (or second review if variance exceeded tolerance) to ensure LTV ratio is calculated using reliable collateral value, as using inflated or erroneous appraisal values would understate collateral risk and lead to inappropriate lending decisions."
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "RenderFinalLoanDecision"
            description: "Synthesizes all underwriting findings to render ultimate credit decision by executing comprehensive decision tree logic that evaluates debt_to_income_ratio against 43.00% QM threshold and 50.00% absolute maximum, loan_to_value_ratio against credit-score-stratified policy thresholds, credit_score_grade for A/B acceptability versus C requiring compensating factors or D requiring decline, property_condition for C3 or better acceptability versus C4/C5 requiring repairs or decline, probability_of_default_score for acceptable risk levels, compensating_factors_present Boolean for marginal DTI or LTV cases, and bankruptcy/foreclosure seasoning compliance, populating final_decision field with Approved for applications meeting all policy requirements, Approved with Conditions for applications meeting standards but requiring outstanding documentation satisfaction enumerated in conditions_count and condition descriptions, or Declined for applications failing minimum credit standards with specific decline_reason documented (DTI excessive, insufficient income, LTV excessive, credit score below minimum, bankruptcy/foreclosure seasoning insufficient, employment verification unsuccessful, property condition unacceptable), recording final_decision_timestamp to enable turnaround_time_days calculation, populating approved_loan_amount (may equal requested amount or reduced amount based on DTI/LTV constraints), approved_interest_rate from final_risk_adjusted_interest_rate potentially recalculated with actual LTV, approved_loan_term_months, calculating first_pass_yield as true if conditions_count equals zero indicating clean approval or false if conditions exist indicating file rework required, preparing adverse action notices for declined applications with specific reasons per ECOA Regulation B requirements, and transitioning approved applications to closing preparation workflow with pre-funding quality control review."
            purpose: "Render authoritative, auditable final credit decision supported by comprehensive quantitative analysis of capacity, credit, collateral, and character factors, ensuring lending decisions are consistent with institutional risk appetite and regulatory requirements while providing clear documentation of decision rationale for approved, conditionally approved, or declined applications to support secondary market delivery, regulatory examination, and fair lending compliance monitoring."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "debt_to_income_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "debt_service_coverage_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "loan_to_value_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "probability_of_default_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "credit_score_grade"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "final_risk_adjusted_interest_rate"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "compensating_factors_present"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "property_condition"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "bankruptcy_seasoning_compliant"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "foreclosure_history_compliant"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "verified_gross_monthly_income"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "requested_loan_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "loan_term_months"
                data_type: "integer"
                source: "user_input"
                required: true
            - name: "application_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            output_parameters:
            - name: "final_decision"
                data_type: "enum"
                destination: "other_api"
            - name: "final_decision_timestamp"
                data_type: "datetime"
                destination: "other_api"
            - name: "approved_loan_amount"
                data_type: "float"
                destination: "other_api"
            - name: "approved_interest_rate"
                data_type: "float"
                destination: "other_api"
            - name: "approved_loan_term_months"
                data_type: "integer"
                destination: "other_api"
            - name: "conditions_count"
                data_type: "integer"
                destination: "other_api"
            - name: "decline_reason"
                data_type: "string"
                destination: "other_api"
            - name: "first_pass_yield"
                data_type: "boolean"
                destination: "other_api"
            - name: "underwriting_decision_memorandum"
                data_type: "text"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CalculateDebtRatios"
                dependency_type: "data_dependency"
                dependency_reason: "Requires debt_to_income_ratio and debt_service_coverage_ratio to evaluate compliance with QM 43% DTI threshold and portfolio 1.15 DSCR requirement, as these capacity metrics are primary approval/decline decision factors that directly determine whether borrower has sufficient income to support the loan."
            - dependent_api: "CalculateProbabilityOfDefault"
                dependency_type: "data_dependency"
                dependency_reason: "Requires probability_of_default_score and credit_score_grade to assess overall credit risk level and determine if risk profile is acceptable (Grade A/B), marginal requiring compensating factors (Grade C), or unacceptable requiring decline (Grade D), as these risk metrics govern approval authority levels and pricing adequacy."
            - dependent_api: "DetermineFinalInterestRate"
                dependency_type: "data_dependency"
                dependency_reason: "Requires final_risk_adjusted_interest_rate as the approved_interest_rate for the loan, potentially recalculated after final LTV is established, as the interest rate is a critical output of the decision process that must be locked and disclosed to the borrower."
            - dependent_api: "CalculateLoanToValueRatio"
                dependency_type: "data_dependency"
                dependency_reason: "Requires loan_to_value_ratio to evaluate collateral coverage adequacy against credit-score-stratified LTV policy limits, determine mortgage insurance requirements, and potentially adjust approved_loan_amount downward if requested amount exceeds maximum supportable by appraised value."
            - dependent_api: "EvaluateCompensatingFactors"
                dependency_type: "data_dependency"
                dependency_reason: "Requires compensating_factors_present Boolean to determine if applications with DTI between 43-50% or other marginal risk metrics can be approved through offsetting positive characteristics, as compensating factors enable appropriate credit access for non-traditional but strong borrower profiles."
            - dependent_api: "AnalyzeCreditHistory"
                dependency_type: "data_dependency"
                dependency_reason: "Requires bankruptcy_seasoning_compliant and foreclosure_history_compliant Booleans to ensure borrowers with prior adverse credit events have satisfied mandatory waiting periods before new mortgage eligibility, as these are hard policy requirements that supersede other positive factors."
            - dependent_api: "ConductPropertyValuation"
                dependency_type: "data_dependency"
                dependency_reason: "Requires property_condition rating to determine if property is acceptable collateral (C3 or better) or requires mandatory repairs before closing (C4) or is unfinanceable without major rehabilitation (C5), as collateral quality directly impacts loan security and foreclosure recovery prospects."
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ExecuteFairLendingMonitoring"
            description: "Conducts automated fair lending compliance surveillance by capturing demographic information collected per Home Mortgage Disclosure Act requirements including applicant ethnicity, race, sex, and Census tract geocode derived from property_address, performing statistical analysis comparing approval rates across demographic cohorts after controlling for legitimate risk factors (credit score, DTI, LTV) to identify statistically significant disparities indicating potential disparate treatment or disparate impact violations of Equal Credit Opportunity Act, analyzing final_risk_adjusted_interest_rate spreads across demographic groups to detect pricing discrimination after risk adjustment, evaluating conditions_count patterns to identify inconsistent application of underwriting standards across protected classes, setting fair_lending_audit_flag to true for randomly selected applications (2-5% sample) or algorithmically flagged applications showing pricing or decisioning anomalies requiring detailed Fair Lending analyst file review, and documenting surveillance results for quarterly fair lending committee reporting, annual regulatory fair lending examination preparation, and continuous process improvement to ensure non-discriminatory credit access consistent with Regulation B and prudential regulatory guidance on fair lending risk management."
            purpose: "Ensure institutional compliance with fair lending laws prohibiting discrimination in credit decisions and pricing based on race, ethnicity, sex, or other protected characteristics, enable early detection of potential fair lending risks through ongoing statistical monitoring before they become systemic issues, and maintain audit trails documenting fair lending due diligence for regulatory examination defense."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "final_decision"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "approved_interest_rate"
                data_type: "float"
                source: "other_api"
                required: false
            - name: "conditions_count"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "representative_credit_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "debt_to_income_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "loan_to_value_ratio"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "property_address"
                data_type: "string"
                source: "user_input"
                required: true
            output_parameters:
            - name: "fair_lending_audit_flag"
                data_type: "boolean"
                destination: "storage"
            - name: "demographic_data_complete"
                data_type: "boolean"
                destination: "storage"
            - name: "pricing_variance_alert"
                data_type: "boolean"
                destination: "storage"
            - name: "decisioning_consistency_alert"
                data_type: "boolean"
                destination: "storage"
            - name: "fair_lending_monitoring_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "RenderFinalLoanDecision"
                dependency_type: "state_dependency"
                dependency_reason: "Can only conduct fair lending monitoring after final_decision is rendered, as the compliance surveillance analyzes approval/decline patterns, pricing consistency, and conditions imposed across demographic groups, requiring the completed underwriting decision as input to the statistical discrimination detection algorithms."
            complexity_level: "medium"
            business_criticality: "medium"

        - api_name: "TrackLoanPerformanceMetrics"
            description: "Calculates operational and business performance metrics by computing turnaround_time_days as calendar days between application_timestamp and final_decision_timestamp to measure process efficiency against institutional target of ≤25 days for W-2 wage earner applications and ≤35 days for self-employed applications, records first_pass_yield Boolean based on conditions_count equaling zero (true for clean approval, false if conditions exist) to measure operational quality and file preparation completeness, tracks funded_status Boolean indicating whether approved loan ultimately closed and was disbursed, records funding_date when loan funds were advanced to complete the transaction, calculates pull-through rate at portfolio level as percentage of approved applications resulting in funded loans (target 75-85%), monitors early payment default rates during first 12 months post-origination segmented by credit_score_grade to validate probability_of_default_score model accuracy through comparison of predicted versus actual delinquency rates, feeds performance data into continuous improvement processes for cycle time reduction, underwriting model recalibration, and origination process optimization while providing management reporting on volume, decisioning patterns, operational efficiency, and portfolio quality metrics supporting strategic planning and resource allocation decisions."
            purpose: "Enable data-driven operational management through systematic tracking of process efficiency metrics (turnaround time, first pass yield), business performance indicators (pull-through rate, approval rates), and portfolio quality measures (early payment default correlation to underwriting predictions), supporting continuous improvement initiatives, model validation requirements, and executive reporting on mortgage origination performance."
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "application_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "final_decision"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "final_decision_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: true
            - name: "conditions_count"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "approved_loan_amount"
                data_type: "float"
                source: "other_api"
                required: false
            output_parameters:
            - name: "turnaround_time_days"
                data_type: "integer"
                destination: "storage"
            - name: "first_pass_yield"
                data_type: "boolean"
                destination: "storage"
            - name: "funded_status"
                data_type: "boolean"
                destination: "storage"
            - name: "funding_date"
                data_type: "date"
                destination: "storage"
            - name: "performance_metrics_calculated_timestamp"
                data_type: "datetime"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "RenderFinalLoanDecision"
                dependency_type: "temporal_dependency"
                dependency_reason: "Requires final_decision_timestamp to calculate turnaround_time_days by measuring elapsed time from application submission to decision rendering, and requires conditions_count to determine first_pass_yield status, as these metrics can only be computed after the underwriting decision process completes."
            complexity_level: "low"
            business_criticality: "medium"
        </api_dependency_graph>
        '''

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_file_contents, additional_context, sample_data, api_graph)

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
