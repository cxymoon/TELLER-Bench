import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI
def load_yaml_prompt(file_path, sop_title, sop_file_contents, additional_context, sample_schema, n_samples, direct_graph):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text"]
    text = text.replace("{{sop_title}}", sop_title)
    text = text.replace("{{sop_file_contents}}", sop_file_contents)
    text = text.replace("{{additional_context}}", additional_context)
    text = text.replace("{{sample_schema}}", sample_schema)
    text = text.replace("{{n_samples}}", n_samples)
    text = text.replace("{{direct_graph}}",direct_graph)
    return text


def main():
    # === 输入参数 ===
    sop_title = "Banking Invest Process"
    sop_file_contents = '''
        <SOP>

        1. Purpose

        The purpose of this Standard Operating Procedure is to establish a comprehensive, multi-layered framework for the execution of retail wealth management investment transactions within a banking and financial services environment, encompassing the full lifecycle from initial client risk profiling and investor suitability assessment through to the final execution and settlement of investment product subscriptions, including but not limited to open-ended mutual fund units and complex structured financial instruments such as Dual Currency Investments and Principal Protected Notes. This SOP ensures strict adherence to all applicable regulatory mandates governing investor protection, Know Your Customer (KYC) requirements, Anti-Money Laundering (AML) screening protocols, suitability and appropriateness assessments as prescribed under MiFID II equivalents and local banking supervisory frameworks, while simultaneously optimizing Assets Under Management (AUM) growth and fee-based revenue generation through systematic, auditable, and compliant sales processes. The procedure integrates automated risk-based decisioning engines, real-time compliance validation checkpoints, and multi-system orchestration across Wealth Management Systems (WMS), Customer Relationship Management (CRM) platforms, Sales Compliance Recording Systems (SCRS), and Risk Engines to achieve Straight-Through Processing (STP) rates exceeding industry benchmarks while maintaining zero-tolerance policies for miss-selling incidents and regulatory breaches.

        2. Scope

        This Standard Operating Procedure applies comprehensively to all Relationship Managers (RMs), Investment Product Specialists (IPSs), Compliance Officers, and Back Office Operations personnel who are directly or indirectly involved in the origination, processing, validation, execution, and post-trade settlement of investment product transactions across all channels including Branch, Phone Banking, Mobile Application, and Online Banking platforms within the organization's retail wealth management business unit. The scope explicitly encompasses three distinct but interconnected sub-business processes: (A) Risk Profiling and Investor Suitability Assessment, which establishes the foundational eligibility and investment parameters for each client; (B) Mutual Fund Subscription (Lump Sum), covering the direct purchase of open-ended mutual fund units through immediate debit from client settlement accounts; and (C) Structured Product Enrollment, addressing the sales, disclosure, and booking procedures for complex financial instruments requiring enhanced due diligence, mandatory audio-video recording, and regulatory cooling-off periods. The SOP governs all transactions originating from clients identified by unique client_id values within the system, processed by authorized relationship_manager_id personnel, and recorded with distinct transaction_id identifiers for complete audit trail purposes. Geographically, this procedure applies across all regions and branch_code locations where the organization maintains retail wealth management operations, with specific regional regulatory adaptations documented in supplementary annexes where local supervisory requirements impose additional or variant obligations beyond the baseline standards articulated herein.

        3. Definitions

        **Risk Tolerance Questionnaire** refers to the regulatory-mandated, multi-dimensional assessment instrument administered to each client to systematically capture quantitative and qualitative data pertaining to the client's age bracket, income stability characteristics, declared annual income (client_annual_income), total net worth (client_net_worth), investment experience measured in years (client_investment_experience_years), liquidity requirements categorized as Low, Medium, or High (client_liquidity_needs), investment time horizon preferences, psychological loss tolerance thresholds, and prior exposure to various asset classes, with all responses recorded in structured JSON format within the risk_questionnaire_responses field and subsequently processed through a proprietary weighted scoring algorithm to generate a risk_questionnaire_total_score ranging from zero to one hundred points.

        **Calculated Risk Profile** denotes the system-generated investor categorization output derived from the risk_questionnaire_total_score through algorithmic mapping logic whereby scores falling within the range of zero to thirty-five points inclusive are categorized as "Conservative," scores from thirty-six to sixty-five points inclusive are designated as "Balanced," and scores from sixty-six to one hundred points inclusive are classified as "Aggressive," with the resulting classification stored in the calculated_risk_profile field and serving as the primary reference parameter for all subsequent suitability matching operations and product recommendation constraints.

        **Product Risk Rating** signifies the standardized risk classification assigned to each investment product identified by product_code and product_name, expressed through a five-tier taxonomy comprising "Conservative," "Moderate," "Balanced," "Growth," and "Aggressive" designations, with the rating determined through quantitative analysis of historical volatility metrics, asset class composition, leverage factors, currency exposure, liquidity characteristics, and structural complexity, and permanently associated with each product record in the product master database to enable automated suitability comparison operations.

        **Suitability Check** represents the automated, real-time validation process executed by the Risk Engine middleware wherein the product_risk_rating of the selected investment instrument is algorithmically compared against the client's calculated_risk_profile using a hierarchical matching matrix such that a product rated "Conservative" is deemed suitable for all client risk profiles, "Moderate" products are suitable for "Balanced" and "Aggressive" profiles, "Balanced" products are suitable for "Balanced" and "Aggressive" profiles, "Growth" products are suitable only for "Aggressive" profiles, and "Aggressive" products are exclusively suitable for "Aggressive" client profiles, with the comparison outcome recorded as a boolean value in the suitability_check_passed field.

        **Suitability Mismatch** occurs when the hierarchical risk comparison logic determines that the product_risk_rating exceeds the permissible threshold relative to the client's calculated_risk_profile, triggering the suitability_mismatch_flag to assume a TRUE boolean state and initiating a mandatory exception workflow requiring the generation of a mismatch_waiver_document_id, explicit client acknowledgment of elevated risk exposure through documented signature, and supervisor_approval_required flag activation necessitating authorization from a supervisor_id personnel possessing appropriate delegation authority before transaction progression is permitted.

        **Net Asset Value (NAV)** constitutes the per-unit market valuation of a mutual fund investment product calculated by the fund administrator at the close of each trading day by dividing the total market value of all securities held within the fund portfolio, minus all liabilities and accrued expenses, by the total number of outstanding fund units, with the most recent NAV figure stored in the current_nav field and utilized as the basis for determining estimated_units to be allocated to the client through the mathematical operation of dividing the net_investment_amount by the current_nav with results typically expressed to two decimal places.

        **Front-End Load** refers to the sales charge or commission assessed at the point of initial investment purchase, expressed as a percentage of the gross subscription_amount in the front_end_load_percentage field, with the absolute dollar value of this charge computed as the product of subscription_amount multiplied by front_end_load_percentage divided by one hundred and recorded in the front_end_load_amount field, thereby reducing the actual capital deployed into the investment product which is represented by the net_investment_amount calculated as subscription_amount minus front_end_load_amount.

        **Annual Management Fee** represents the ongoing administrative and portfolio management charge levied by the fund house or investment manager, expressed as an annualized percentage rate in the annual_management_fee_percentage field, with the estimated first-year dollar impact calculated as net_investment_amount multiplied by annual_management_fee_percentage divided by one hundred and recorded in the annual_management_fee_amount field to provide clients with transparent disclosure of the recurring cost burden associated with the investment product over the holding period.

        **Straight-Through Processing (STP)** designates the operational paradigm wherein a transaction progresses from initial order entry by the Relationship Manager through all validation checkpoints, compliance screening layers, execution operations, and booking confirmations without requiring any manual intervention, data correction, exception handling, or workflow rerouting by Back Office Operations personnel, with achievement of this state indicated by the straight_through_processing_flag assuming a TRUE boolean value and serving as a key performance metric for operational efficiency measurement.

        **Cooling-Off Period** signifies a regulatory-mandated temporal interval during which a client who has authorized the purchase of a complex or high-risk structured product retains the unconditional right to cancel the transaction without incurring penalties, surrender charges, or adverse financial consequences, with the duration specified in hours within the cooling_off_period_hours field (typically twenty-four or forty-eight hours depending on product category and jurisdiction), and the transaction maintained in a "Pending_Cooling_Off" status within the transaction_status field until the designated period expires or the client exercises their cancellation right.

        **Audio-Video Recording (VR)** refers to the mandatory electronic capture, encryption, and archival of the complete sales interaction between the Relationship Manager and the client for transactions involving complex structured products where audio_video_recording_required assumes a TRUE boolean value, with each recording session assigned a unique audio_video_recording_id identifier and stored within the Sales Compliance Recording System (SCRS) for subsequent compliance review, quality assurance sampling, and regulatory audit purposes, ensuring complete documentation of risk disclosure script delivery and client verbal acknowledgment of understanding.

        **Trade Reference Number** constitutes the unique system-generated alphanumeric identifier assigned to each successfully executed investment transaction at the moment of order booking, stored in the trade_reference_number field, serving as the primary reference key for all downstream settlement operations, client confirmation communications, portfolio accounting entries, and regulatory transaction reporting obligations.

        **Contract Note Number** represents the specialized confirmation document identifier generated exclusively for structured product transactions processed through Treasury or Global Markets desks, stored in the contract_note_number field, containing comprehensive details including notional amount booked, strike price parameters, value date specifications, payoff scenario matrices, and counterparty information necessary for client record-keeping and regulatory compliance.

        4. Input

        The initiation of any investment transaction workflow governed by this Standard Operating Procedure requires the complete availability and validation of the following input data elements and contextual parameters drawn from multiple source systems and client interaction touchpoints. First, the transaction must be associated with a valid and active client_id representing a customer whose identity has been previously verified through government-issued identification documents and whose profile data is maintained within the Customer Relationship Management (CRM) system, including but not limited to client_name, client_age measured in integer years, client_annual_income declared in United States Dollar equivalents, client_net_worth representing total liquid and illiquid asset valuations, and client_investment_experience_years quantifying the duration of prior involvement with capital markets instruments. Second, the transaction must be originated and sponsored by an authorized relationship_manager_id whose relationship_manager_name, delegation authorities, and product certification credentials are recorded in the human resources and access control systems, with the initial transaction_timestamp capturing the precise date and time of workflow initiation in ISO 8601 format with timezone designation. Third, the client's settlement account infrastructure must be established with a valid settlement_account_number associated with the client_id and containing a verifiable available_balance in United States Dollar denomination sufficient to support the intended subscription_amount inclusive of all applicable fees and charges. Fourth, for transactions involving investment products, the system must have access to complete product master data including product_code, product_name, product_risk_rating, product_category classification (Mutual_Fund, Bond, Structured_Note, Dual_Currency_Investment, or Principal_Protected_Note), current_nav for unit-based instruments, front_end_load_percentage applicable to the specific client relationship tier, annual_management_fee_percentage as stipulated in the product prospectus, and disclosure_documents_presented list containing references to Fund_Fact_Sheet, Prospectus, Key Information Document (KID), and any supplementary risk disclosure materials. Fifth, the client's current compliance and eligibility status must be retrievable including the previous_risk_profile classification if any prior assessment exists, the previous_risk_profile_expiry_date indicating when the existing assessment loses validity, the aml_sanctions_screening_status reflecting the outcome of most recent sanctions list screening operations, the source_of_wealth_documented boolean flag indicating whether SOW documentation is on file, and the monitoring_threshold_amount parameter defining the trigger point for enhanced AML scrutiny which is standardly configured at one hundred thousand United States Dollars but may be adjusted based on client segment or regional regulatory requirements. Sixth, for structured products specifically, the input data must include the audio_video_recording_required boolean flag determined by product classification rules, the cooling_off_period_required flag indicating regulatory mandate applicability, and the cooling_off_period_hours duration specification. Seventh, the transaction must be tagged with operational metadata including branch_code identifying the originating business location, channel designation (Branch, Phone_Banking, Mobile_App, or Online_Banking), region code for geographic reporting purposes, and assignment of a compliance_officer_id responsible for oversight and exception review. Eighth, system access to market_data_source platforms such as Bloomberg or Reuters must be confirmed operational to enable real-time NAV verification, pricing validation, and market condition assessment. Finally, the client_authorization_method infrastructure must be configured and functional whether utilizing One-Time Password (OTP) delivery systems, biometric signature capture devices, physical signature verification protocols, or digital signature certificate frameworks to enable legally binding transaction consent capture with associated client_authorization_timestamp recording.

        5. Main Procedure

        5.1 Client Risk Profile Assessment and Establishment Phase

        5.1.1 Upon initiation of an investment transaction request or commencement of a new client investment relationship, the Relationship Manager designated by relationship_manager_id must first verify that the client identified by client_id possesses a current and valid risk profile by interrogating the CRM system to retrieve the previous_risk_profile classification and associated previous_risk_profile_expiry_date timestamp, comparing the expiry date against the current transaction_timestamp to determine temporal validity whereby if the previous_risk_profile_expiry_date is null indicating no prior assessment exists or if the previous_risk_profile_expiry_date precedes the current transaction_timestamp indicating expiry of the existing assessment, then the RM must initiate a complete Risk Tolerance Questionnaire administration workflow, whereas if a valid and unexpired previous_risk_profile exists then the RM may proceed directly to product selection and suitability validation stages while documenting the continued reliance on the existing risk categorization in the transaction audit trail.

        5.1.2 For scenarios requiring new or refreshed risk assessment, the Relationship Manager must administer the standardized Risk Tolerance Questionnaire instrument through either digital tablet-based presentation in branch environments or secure web-form delivery for remote channel transactions, ensuring the client personally completes all mandatory questions without RM influence or suggestion that might compromise assessment objectivity, with the questionnaire systematically collecting structured responses covering age bracket classification derived from client_age such that clients aged eighteen to thirty years receive zero baseline points, thirty-one to forty years receive five points, forty-one to fifty years receive ten points, fifty-one to sixty years receive fifteen points, and sixty-one years or greater receive twenty points reflecting increasing conservative orientation with age; income stability evaluation based on client_annual_income and employment sector whereby salaried employees in stable industries receive fifteen points, self-employed or commission-based income sources receive ten points, and retired individuals with fixed income streams receive twenty points; investment time horizon assessment whereby horizons less than one year receive zero points, one to three years receive ten points, three to five years receive fifteen points, five to ten years receive twenty points, and greater than ten years receive twenty-five points reflecting capacity to weather market volatility; loss tolerance psychological evaluation presented through scenario-based questions whereby willingness to accept potential loss of zero to five percent over one year receives five points, five to fifteen percent tolerance receives fifteen points, fifteen to thirty percent tolerance receives twenty-five points, and acceptance of greater than thirty percent loss potential receives thirty points; and prior investment experience scoring wherein client_investment_experience_years of zero years receives zero points, one to three years receives five points, four to seven years receives ten points, eight to fifteen years receives fifteen points, and greater than fifteen years receives twenty points, with all individual question responses recorded in JSON format within the risk_questionnaire_responses field maintaining complete granular detail for audit and quality review purposes.

        5.1.3 Upon completion of questionnaire data capture, the Risk Engine component of the Wealth Management System must aggregate all individual question scores to compute the risk_questionnaire_total_score as an integer summation ranging from zero to one hundred points, subsequently applying deterministic mapping logic to derive the calculated_risk_profile classification such that total scores in the range zero through thirty-five points inclusive result in "Conservative" classification reflecting low risk tolerance, preference for capital preservation, and suitability primarily for fixed income and money market instruments; scores in the range thirty-six through sixty-five points inclusive result in "Balanced" classification indicating moderate risk tolerance, acceptance of measured volatility, and suitability for diversified portfolios combining equity and fixed income exposures; and scores in the range sixty-six through one hundred points inclusive result in "Aggressive" classification signifying high risk tolerance, focus on capital appreciation over income generation, and suitability for concentrated equity positions, alternative investments, and complex structured products, with the calculated_risk_profile value recorded in the designated field and simultaneously triggering system updates to the client's master profile record in the CRM database.

        5.1.4 The system must automatically present the calculated_risk_profile classification to the client through the RM's interface along with a comprehensive narrative description explaining the characteristics, typical investment strategies, and product categories associated with the determined profile, requiring explicit client acknowledgment through one of the configured client_authorization_method protocols such that if the client agrees with the calculated classification they must provide affirmative consent via OTP validation, biometric signature capture, physical signature on printed assessment summary, or digital signature application with the client_authorization_timestamp precisely recorded, whereas if the client disputes the calculated classification expressing belief that their risk tolerance is either higher or lower than the system determination, the RM must initiate a documented re-evaluation workflow involving completion of a supplementary questionnaire with additional scenario-based questions, consultation with an Investment Product Specialist referenced by IPS identifier for expert second opinion, and mandatory submission of written justification narrative explaining the basis for override request which is subject to compliance_officer_id review and approval before an adjusted calculated_risk_profile may be recorded in the system with all override documentation forming part of the permanent client file.

        5.1.5 Following successful client acceptance and authorization of the calculated_risk_profile, the system must automatically compute the risk_profile_expiry_date by adding a duration of exactly seven hundred thirty days (equivalent to two calendar years) to the current transaction_timestamp to establish the validity period for the assessment, update the client's master record in the CRM with the new calculated_risk_profile value replacing any previous_risk_profile designation, set the risk_profile_expiry_date in the client profile, and most critically modify the client's investable_status field to assume the value "Active" thereby removing any systemic blocks that would prevent investment transaction processing and enabling the client to proceed with product purchases within their suitability parameters, while also triggering automated calendar reminders to the assigned relationship_manager_id at intervals of sixty days prior to expiry, thirty days prior to expiry, and at expiry date to ensure proactive risk profile renewal before eligibility lapses and transactions become blocked.

        5.2 Investment Product Selection and Suitability Validation Phase

        5.2.1 With a valid and active calculated_risk_profile established and the investable_status confirmed as "Active," the Relationship Manager must proceed to product recommendation and selection activities by first retrieving from the product master database all investment instruments whose product_risk_rating falls within the permissible range aligned with the client's calculated_risk_profile utilizing the hierarchical suitability matrix logic whereby clients with "Conservative" calculated profiles may only be presented products rated "Conservative" or "Moderate," clients with "Balanced" calculated profiles may be presented products rated "Conservative," "Moderate," or "Balanced," and clients with "Aggressive" calculated profiles may be presented products across all rating categories including "Conservative," "Moderate," "Balanced," "Growth," and "Aggressive," with the RM particularly focusing recommendation efforts on products generating optimal fee-based revenue through front_end_load_percentage and annual_management_fee_percentage structures while remaining within client suitability constraints and appropriateness for the client's stated investment objectives, time horizon, and current portfolio composition.

        5.2.2 Upon identifying a specific target product characterized by product_code, product_name, product_category, and product_risk_rating, the Relationship Manager must conduct a comprehensive product presentation session with the client encompassing mandatory disclosure of all material information contained within the regulatory-required disclosure_documents_presented list, specifically including the Fund Fact Sheet containing one-page summary of investment objective, asset allocation strategy, historical performance metrics for one-year, three-year, five-year, and since-inception periods with explicit disclaimer that past performance does not guarantee future results, the complete Prospectus or offering document detailing investment policies, restrictions, risk factors, management team biographies, and financial statements, and the Key Information Document (KID) formatted in standardized layout presenting objectives, risk and reward profile using the Synthetic Risk and Reward Indicator (SRRI) scale, cost disclosures showing reduction in yield due to charges, and performance scenarios under stressed, unfavorable, moderate, and favorable market conditions, with the RM required to explicitly draw client attention to the product_risk_rating classification, the front_end_load_percentage expressed both as percentage rate and absolute dollar amount calculated as subscription_amount multiplied by front_end_load_percentage divided by one hundred, the annual_management_fee_percentage similarly expressed in both percentage and estimated annual dollar impact, and specific sections of the risk disclosure content addressing market risk, credit risk for fixed income products, currency risk for internationally-invested funds, liquidity risk, and concentration risk where applicable.

        5.2.3 For transactions involving structured products where product_category is designated as "Structured_Note," "Dual_Currency_Investment," or "Principal_Protected_Note," additional procedural requirements beyond standard mutual fund disclosure become mandatory, specifically requiring the Relationship Manager to conduct a detailed payoff scenario analysis presentation utilizing market data sourced from the designated market_data_source platform such as Bloomberg or Reuters to demonstrate to the client through numerical examples and graphical illustrations how the structured product's return profile varies under different market conditions, for instance explaining that a Dual Currency Investment may deliver a five percent coupon if the underlying exchange rate remains above a specified strike level at maturity but could result in principal repayment in a depreciated foreign currency if the strike is breached thereby crystallizing foreign exchange loss, and for any structured product transaction where audio_video_recording_required assumes TRUE boolean value indicating regulatory mandate for sales process documentation, the RM must activate the Sales Compliance Recording System (SCRS) to capture complete audio and video recording of the presentation and client interaction, with the system generating a unique audio_video_recording_id identifier that becomes permanently associated with the transaction_id and stored in encrypted format on secure archival storage infrastructure for minimum retention periods typically extending seven years to satisfy regulatory audit and examination requirements.

        5.2.4 Following completion of all mandatory disclosure presentations and client review of documentation, the Relationship Manager must proceed to formal order entry by accessing the Wealth Management System's transaction module and creating a new transaction record populated with transaction_id generated through system-assigned sequential alphanumeric format (e.g., INV_YYYYMMDDNNNNN), capturing the current transaction_timestamp with precision to the second in UTC format, associating the client_id and relationship_manager_id with the transaction, specifying the transaction_type as "Risk_Profiling" if this is solely a profile establishment transaction, "Mutual_Fund_Subscription" for open-ended fund purchases, or "Structured_Product_Enrollment" for complex instrument bookings, recording the selected product_code and confirming automatic population of related product_name, product_risk_rating, product_category, current_nav for NAV-based funds, front_end_load_percentage, and annual_management_fee_percentage from the product master database, capturing the client's desired subscription_amount in United States Dollar value, identifying the designated settlement_account_number from which funds will be debited, and tagging the transaction with branch_code, channel, region, and assigned compliance_officer_id based on transaction value thresholds and risk characteristics of the product.

        5.2.5 At the moment of order entry completion but prior to final client authorization and execution, the Wealth Management System must invoke its integrated Risk Engine middleware to perform automated suitability validation by executing a comparative evaluation between the product_risk_rating of the selected investment instrument and the client's established calculated_risk_profile using the hierarchical matching algorithm such that the suitability_check_passed field assumes TRUE boolean value if the product rating is equal to or lower than the client profile on the Conservative-Moderate-Balanced-Growth-Aggressive hierarchy (specifically: Conservative products pass for all profiles; Moderate products pass for Balanced and Aggressive profiles; Balanced products pass for Balanced and Aggressive profiles; Growth products pass only for Aggressive profiles; Aggressive products pass only for Aggressive profiles), whereas if the product_risk_rating is hierarchically superior to the client's calculated_risk_profile indicating a suitability violation, the suitability_check_passed field assumes FALSE boolean value and the suitability_mismatch_flag is set to TRUE triggering immediate workflow diversion to an exception handling process that displays a hard-block error message to the Relationship Manager preventing order progression unless the RM initiates a formal Risk Mismatch Waiver procedure requiring generation of a mismatch_waiver_document_id identifier, electronic or physical presentation of a standardized waiver form to the client explicitly stating in clear and unambiguous language that the recommended product exceeds the client's assessed risk tolerance and may result in investment losses greater than the client indicated willingness to accept during risk profiling, capture of client signature acknowledging and accepting the elevated risk exposure and confirming understanding that they are proceeding against the bank's standard suitability recommendation, and critically the activation of supervisor_approval_required flag to TRUE boolean state necessitating routing of the transaction to a supervisory personnel identified by supervisor_id possessing appropriate delegation authority who must review the complete risk assessment documentation, product disclosure records, waiver acknowledgment, and RM justification narrative before either approving the transaction which permits continuation to execution or rejecting the transaction which terminates the workflow and requires the RM to recommend an alternative suitable product to the client.

        5.3 Pre-Trade Compliance and Financial Validation Phase

        5.3.1 For all transactions regardless of product type where the suitability_check_passed field confirms TRUE boolean value either through direct matching or through approved supervisor override following mismatch waiver procedures, the system must proceed to execute comprehensive pre-trade compliance screening operations beginning with Anti-Money Laundering (AML) sanctions list verification by interfacing with the bank's enterprise AML monitoring platform to submit the client_id, client_name, and transaction details for real-time screening against all relevant sanctions lists including OFAC (Office of Foreign Assets Control) Specially Designated Nationals (SDN) list, EU consolidated sanctions list, UN Security Council sanctions designations, and national sanctions registries applicable to the bank's operating jurisdictions, with the screening outcome recorded in the aml_sanctions_screening_status field assuming values of "Clear" if no matches are detected indicating permission to proceed, "Flagged" if potential name matches or watchlist hits are identified requiring immediate escalation to the compliance_officer_id for manual review and disposition decision before transaction can advance, or maintaining "Pending" status during the interval while screening systems process the query with typical resolution timeframes under five seconds for unambiguous clear results.

        5.3.2 Concurrent with AML sanctions screening, the system must evaluate whether the transaction subscription_amount exceeds the configured monitoring_threshold_amount parameter which is standardly established at one hundred thousand United States Dollars (100000.00) for retail wealth management transactions, setting the transaction_threshold_exceeded field to TRUE boolean value when subscription_amount is strictly greater than monitoring_threshold_amount, and when this threshold breach condition is detected, the system must automatically verify that the source_of_wealth_documented field associated with the client_id assumes TRUE boolean value indicating that current and valid Source of Wealth (SOW) documentation such as employment contracts, business ownership records, inheritance documentation, real estate sale proceeds verification, or investment portfolio liquidation statements have been collected and verified by the compliance team and are stored in the document management system, such that if source_of_wealth_documented is FALSE when transaction_threshold_exceeded is TRUE, the system must generate a hard-block preventing transaction execution and display to the Relationship Manager explicit instruction that SOW documentation must be obtained, submitted for compliance review, and approved before the transaction can proceed with the workflow entering suspended state pending documentation completion.

        5.3.3 Following successful completion of AML screening yielding "Clear" status and confirmation of SOW documentation adequacy for threshold-exceeding transactions, the system must proceed to funds availability verification by executing a real-time balance inquiry against the Core Banking System using the settlement_account_number to retrieve the current available_balance in United States Dollar equivalents, comparing this available_balance against the required subscription_amount to determine sufficiency such that the funds_availability_verified field assumes TRUE boolean value only when available_balance is greater than or equal to subscription_amount thereby confirming that adequate liquid funds exist to settle the transaction without creating overdraft conditions or requiring supplementary funding arrangements, whereas if available_balance is less than subscription_amount indicating insufficient funds, the funds_availability_verified field assumes FALSE boolean value triggering generation of an insufficient funds notification to the Relationship Manager who must either arrange for the client to transfer additional funds into the settlement_account_number before proceeding or reduce the subscription_amount to fit within the available_balance constraint with the revised amount requiring fresh suitability validation if the reduction is substantial enough to change fee structures or minimum investment thresholds.

        5.3.4 Aggregating all component validation outcomes, the system must compute an overall pre_trade_compliance_passed field assuming TRUE boolean value only when all of the following conditions are simultaneously satisfied: suitability_check_passed equals TRUE (either directly or through approved override), aml_sanctions_screening_status equals "Clear," if transaction_threshold_exceeded equals TRUE then source_of_wealth_documented must equal TRUE, and funds_availability_verified equals TRUE, such that failure of any single validation component results in pre_trade_compliance_passed assuming FALSE boolean value and preventing workflow advancement to execution stages, with the system generating detailed diagnostic messages identifying which specific validation component failed to enable the Relationship Manager to take corrective action such as obtaining missing SOW documentation, waiting for funds transfer completion, or selecting an alternative product that passes suitability requirements.

        5.3.5 For mutual fund transactions specifically where product_category equals "Mutual_Fund" and pre_trade_compliance_passed confirms TRUE boolean value, the system must proceed to calculate precise financial metrics beginning with computation of the front_end_load_amount expressed in United States Dollars by multiplying the subscription_amount by the product's front_end_load_percentage and dividing by one hundred with results rounded to two decimal places using standard banker's rounding convention (round half to even), subsequently calculating the net_investment_amount representing the actual capital deployed into the fund by subtracting the front_end_load_amount from the subscription_amount, utilizing the current_nav value retrieved from the product master database (which itself is updated nightly through automated feeds from fund administrators or sourced real-time from market_data_source platforms during trading hours) to compute the estimated_units that will be allocated to the client's portfolio by dividing the net_investment_amount by the current_nav with precision extended to four decimal places to ensure accurate unit allocation given that mutual fund units are typically fractional, and additionally calculating an estimated annual_management_fee_amount by multiplying the net_investment_amount by the annual_management_fee_percentage and dividing by one hundred to project the first year's ongoing fee burden, with all calculated financial metrics recorded in their respective fields and presented to the client in a comprehensive Fee Impact Sheet disclosure summary enabling transparent understanding of all costs associated with the investment prior to final authorization.

        5.4 Client Authorization and Transaction Execution Phase

        5.4.1 With all pre-trade compliance validations successfully passed yielding pre_trade_compliance_passed equals TRUE and all financial calculations completed including front_end_load_amount, net_investment_amount, estimated_units, and annual_management_fee_amount determined and disclosed, the Relationship Manager must present the complete transaction summary to the client for final review and authorization, utilizing the Wealth Management System's transaction preview interface to display a consolidated summary screen showing the product_name, subscription_amount, front_end_load_amount with explicit notation of the percentage and dollar impact, net_investment_amount representing actual capital invested, estimated_units based on current_nav with clear disclaimer that final unit allocation will be determined based on the official NAV calculated at the close of business on the transaction date which may differ from the estimated NAV displayed, annual_management_fee_amount with explanation of ongoing fee structure, settlement_account_number from which funds will be debited, expected settlement_date typically following T+1 (trade date plus one business day) or T+2 settlement cycles depending on product category and fund house processing rules, and all relevant risk disclosures and terms and conditions, requiring the client to explicitly confirm their understanding and acceptance of all displayed terms before authorization is granted.

        5.4.2 Upon client confirmation of transaction review and expression of intent to proceed, the system must invoke the configured client_authorization_method framework to capture legally binding consent whereby if the method is designated as "OTP" (One-Time Password), the system generates a six-digit numerical code transmitted to the client's registered mobile telephone number via SMS or banking application push notification requiring the client to manually enter the received code into the transaction interface within a validity window of three hundred seconds (five minutes) before code expiration; if the method is "Biometric" the system interfaces with fingerprint scanner, facial recognition camera, or iris scanning hardware to capture the client's biometric signature comparing the captured template against the reference template stored during account opening with matching threshold typically set at ninety-five percent confidence level or higher; if the method is "Physical_Signature" the client must apply their handwritten signature to a printed transaction document or deal ticket with the RM verifying signature consistency against the specimen signature on file; or if the method is "Digital_Signature" the client utilizes their personal digital certificate issued under Public Key Infrastructure (PKI) framework to cryptographically sign the transaction data creating a non-repudiable authorization record, with the successful completion of authorization via any of these methods recorded by capturing the precise client_authorization_timestamp in ISO 8601 format with timezone and millisecond precision establishing the exact moment of legally binding consent.

        5.4.3 Immediately following successful client authorization capture, the system must evaluate whether the transaction involves a structured product requiring a regulatory cooling-off period by examining the cooling_off_period_required boolean field, such that for transactions where cooling_off_period_required equals FALSE (typically mutual fund and standard bond purchases), the system proceeds directly to immediate execution operations, whereas for transactions where cooling_off_period_required equals TRUE (typically structured notes, dual currency investments, and principal protected notes in jurisdictions mandating investor protection cooling-off rights), the system must divert the transaction to a deferred execution workflow by setting the transaction_status field to "Pending_Cooling_Off," recording the cooling_off_period_hours duration (commonly twenty-four or forty-eight hours depending on product complexity and local regulations), calculating a cooling-off expiry timestamp by adding the cooling_off_period_hours to the client_authorization_timestamp thereby establishing the earliest moment at which execution may proceed, and placing the transaction into an automated batch queue monitored by scheduled system jobs that execute at hourly intervals to identify all transactions whose cooling-off expiry timestamp has passed and whose status indicates no client cancellation request has been received, at which point the system automatically updates transaction_status from "Pending_Cooling_Off" to "Ready_for_Execution" and routes the transaction to the appropriate execution subsystem.

        5.4.4 For transactions where cooling-off requirements are not applicable or where the cooling-off period has successfully elapsed without client cancellation, the system must proceed to transaction execution operations which vary based on product_category such that for product_category equal to "Mutual_Fund," the Wealth Management System interfaces with the fund house order routing gateway to transmit an electronic purchase order containing the client_id mapped to the fund house's investor account number, product_code mapped to the fund's ISIN or local fund identifier, net_investment_amount as the subscription value, estimated_units for reference purposes with notation that final allocation will be NAV-based, settlement_account_number details for fund receipt, and transaction_timestamp establishing trade date, with the fund house's automated order management system acknowledging receipt and returning a preliminary order confirmation that the system stores as the trade_reference_number serving as the primary tracking identifier for all downstream settlement and reconciliation activities; whereas for product_category equal to "Structured_Note," "Dual_Currency_Investment," or "Principal_Protected_Note," the Wealth Management System creates a structured product order record and transmits this via secure messaging protocols to the bank's Treasury or Global Markets desk which functions as the internal counterparty or arranger for such instruments, with the Treasury desk's trading personnel reviewing the order parameters including notional amount, tenor, strike prices, barrier levels, and underlying asset specifications before manually confirming acceptance and execution which triggers generation of a contract_note_number representing the formal trade confirmation documentation that includes all economic terms, value_date establishing when the structured note becomes effective, pricing parameters, and payoff formulas that the system associates with the transaction record.

        5.4.5 Concurrent with order transmission to external fund houses or internal Treasury desks, the system must execute settlement account debit operations by generating an immediate debit instruction to the Core Banking System specifying the settlement_account_number, the full subscription_amount (not merely net_investment_amount, as the gross amount including fees must be debited), a descriptive narration field containing transaction_id and product_name for client statement clarity, and value date designation matching the transaction execution date, with the Core Banking System returning a debit confirmation reference number that validates successful fund reservation or movement from available balance to blocked or transit status pending final settlement with the fund house or product issuer, and upon successful debit confirmation, the system updates the available_balance field in the transaction record to reflect the post-transaction balance status, sets the transaction_status field to "Executed" indicating successful completion of all execution operations, records the execution_timestamp capturing the exact date and time when execution operations concluded, calculates the settlement_date based on product-specific settlement cycles (T+1 for most mutual funds, T+2 for certain offshore funds, T+0 or T+1 for structured products depending on Treasury desk processing), and critically evaluates whether the entire transaction lifecycle from initiation through execution occurred without any manual intervention, exception handling, workflow rerouting, or data correction by setting the straight_through_processing_flag to TRUE only when zero manual touchpoints occurred or to FALSE if any Operations or Compliance personnel intervened at any stage thereby providing key operational efficiency metrics.

        5.5 Post-Execution Compliance Review and Documentation Phase

        5.5.1 Following successful transaction execution with transaction_status updated to "Executed" and all primary processing operations completed, the system must automatically route a subset of transactions to mandatory post-execution compliance review based on risk-based selection criteria whereby all transactions where audio_video_recording_required equals TRUE indicating complex product sales must be queued for compliance review, all transactions where suitability_mismatch_flag equals TRUE indicating risk override scenarios must be reviewed, all transactions where transaction_threshold_exceeded equals TRUE indicating large-value orders must be sampled, and additionally a statistically random sample of standard transactions equating to typically five percent of daily transaction volume must be selected to ensure ongoing quality monitoring of routine sales practices, with all selected transactions routed to the assigned compliance_officer_id whose workqueue interface displays the transaction details, provides direct playback access to audio_video_recording_id files where applicable, presents the complete audit trail of system decision points and validation outcomes, and enables the compliance officer to review for potential process violations, documentation deficiencies, inadequate risk disclosure, or signs of client pressure or miss-selling behaviors.

        5.5.2 The assigned Compliance Officer executing post-trade review must systematically evaluate each transaction against a standardized compliance checklist covering verification that disclosure_documents_presented list includes all mandatory materials appropriate to the product_category, confirmation that client_authorization_timestamp exists and authorization method was properly executed, validation that for any suitability_mismatch_flag equals TRUE scenario the associated mismatch_waiver_document_id is present and supervisor_approval_timestamp confirms supervisory review occurred, assessment of audio_video_recording_id content where applicable to verify that the Relationship Manager delivered the mandatory risk disclosure script in clear and comprehensible language without minimizing downside scenarios and that the client verbally acknowledged understanding without evidence of confusion or hesitation that might suggest insufficient comprehension, and evaluation of overall transaction coherence relative to client_age, client_annual_income, client_net_worth, and stated investment objectives to identify potentially unsuitable sales that passed automated screening but exhibit qualitative concerns, with the outcome of this review process documented through entry of compliance_review_notes text field containing detailed observations and conclusions, and critically the setting of compliance_defect_flag to TRUE boolean value if any material deficiencies are identified that constitute process violations requiring corrective action or to FALSE if the transaction is determined fully compliant with all regulatory and internal policy requirements.

        5.5.3 For any transaction where compliance_defect_flag assumes TRUE boolean value indicating identification of process violations or documentation deficiencies, the Compliance Officer must initiate corrective action workflows which may include requiring the Relationship Manager to obtain supplementary documentation from the client such as additional risk acknowledgment forms or missing SOW records, scheduling a mandatory follow-up client contact to address disclosure gaps or confirm understanding of product risks that may not have been adequately communicated during initial sale, imposing transaction reversal or cancellation where severe suitability violations or material misrepresentation is identified within allowable reversal timeframes typically extending three to five business days post-execution, generating formal incident reports that are escalated to senior management and legal/compliance committees documenting the nature of the defect and remediation actions taken, and where patterns of defects are associated with specific relationship_manager_id personnel, triggering performance management interventions ranging from mandatory retraining on product knowledge and sales process requirements to suspension of investment sales privileges pending demonstration of competency improvement, with all corrective actions tracked through case management systems and recorded in the transaction_id audit trail to establish complete regulatory accountability.

        5.5.4 Parallel to compliance review processes, the Operations or Back Office team identified by operational processing queues must execute post-execution reconciliation and settlement monitoring activities by comparing the trade_reference_number or contract_note_number generated during execution against subsequent confirmation messages received from fund houses, custodian banks, or internal Treasury systems to verify that external parties have accurately booked the transaction with matching details including client account references, product identifiers, subscription amounts, unit allocations based on official NAV calculations, and settlement_date expectations, with any discrepancies between internally recorded transaction parameters and external confirmations flagged as exceptions requiring investigation and resolution through formal breaks management processes that may involve contacting fund house operations teams, submitting transaction amendment requests, or in severe cases unwinding and rebooking transactions to correct errors, and once full reconciliation is achieved confirming that the estimated_units calculated during pre-execution based on preliminary current_nav align within acceptable tolerance ranges (typically plus or minus zero point five percent) with the final units allocated by the fund house based on official end-of-day NAV calculations, the Operations team updates the transaction record with final unit quantities and confirms that settlement accounting entries have been accurately posted to both the client's investment portfolio custody account and the bank's omnibus accounts with external fund houses or custodians.

        5.5.5 As the culminating step of the post-execution phase, the system must generate comprehensive client confirmation documentation by producing a trade confirmation letter or contract note that includes transaction_id for reference, client_name and client_id for identification, product_name and product_code for clarity on what was purchased, subscription_amount showing the total funds committed, front_end_load_amount transparently disclosing upfront fees charged, net_investment_amount showing capital actually deployed, estimated_units allocated pending final NAV confirmation with explicit notation that final unit allocation may vary slightly, current_nav used for estimation purposes, settlement_date indicating when the transaction will be finalized, settlement_account_number debited, trade_reference_number or contract_note_number for tracking purposes, comprehensive fee disclosures reiterating both front_end_load_percentage and annual_management_fee_percentage with projected annual_management_fee_amount, and regulatory disclosures including cooling-off rights if applicable, complaint procedures, and investor protection scheme information, with this confirmation document delivered to the client through their preferred communication channel whether postal mail, secure email, mobile application notification, or online banking message center within the regulatory-mandated timeframe typically requiring delivery within one business day of execution, and the system records the confirmation delivery timestamp and method to demonstrate compliance with client communication obligations, simultaneously calculating the total_processing_time_minutes field by computing the elapsed time in minutes between the original transaction_timestamp when the process initiated and the execution_timestamp when execution operations completed, providing a quantitative metric of processing efficiency that feeds into operational dashboards tracking cycle times and identifying process bottlenecks requiring optimization attention.

        6. Output

        Upon successful completion of all procedural stages encompassing Risk Profile Assessment and Establishment, Investment Product Selection and Suitability Validation, Pre-Trade Compliance and Financial Validation, Client Authorization and Transaction Execution, and Post-Execution Compliance Review and Documentation, the Banking Invest Process produces a comprehensive set of deterministic outputs recorded within the transaction data structure and associated systems enabling full auditability and operational measurement. The primary quantitative outputs include the calculated_risk_profile classification assuming one of three discrete values ("Conservative," "Balanced," or "Aggressive") derived through algorithmic scoring of risk_questionnaire_total_score with the corresponding risk_profile_expiry_date timestamp establishing validity period, the investable_status field updated to "Active" enabling future transaction processing, the suitability_check_passed boolean definitively indicating whether product selection aligns with client risk parameters, the pre_trade_compliance_passed boolean aggregating all validation outcomes, the precise financial calculations comprising front_end_load_amount expressed in United States Dollars with two decimal precision, net_investment_amount representing actual invested capital, estimated_units calculated to four decimal precision through NAV-based division, and annual_management_fee_amount projecting ongoing cost burden. Transaction execution outputs include the system-generated trade_reference_number or contract_note_number serving as permanent tracking identifiers, the definitive transaction_status value progressing through state transitions from "Pending" through "Pending_Cooling_Off" if applicable to final "Executed" state for successful completions or "Cancelled" or "Failed" states for unsuccessful outcomes, the execution_timestamp providing precise temporal record of transaction consummation, the settlement_date establishing when final settlement will occur typically following T+1 or T+2 cycles, and the straight_through_processing_flag boolean indicating whether zero-touch processing was achieved contributing to STP rate metrics calculated as the percentage of transactions with this flag set to TRUE. Compliance and quality outputs encompass the compliance_defect_flag boolean identifying transactions requiring corrective action, the detailed compliance_review_notes text field documenting officer observations, the supervisor_approval_timestamp for escalated transactions requiring management authorization, and the audio_video_recording_id providing immutable evidence of sales process execution for complex products. Operational efficiency metrics output includes the total_processing_time_minutes integer value enabling calculation of average cycle time performance indicators, the funds_availability_verified boolean confirming settlement readiness, and the available_balance post-transaction value enabling liquidity monitoring. Risk control outputs definitively record the suitability_mismatch_flag boolean and associated mismatch_waiver_document_id for exception tracking, the aml_sanctions_screening_status confirming compliance with sanctions regulations, the source_of_wealth_documented boolean validating enhanced due diligence for high-value transactions, and the transaction_threshold_exceeded flag identifying transactions subject to enhanced monitoring. Collectively these outputs enable computation of key performance indicators including Sales Conversion Rate calculated as (count of transactions with transaction_status equals "Executed") divided by (count of all initiated transactions) multiplied by one hundred percent, Straight-Through Processing Rate calculated as (count of transactions with straight_through_processing_flag equals TRUE) divided by (count of transactions with transaction_status equals "Executed") multiplied by one hundred percent, Suitability Pass Rate calculated as (count of transactions with suitability_check_passed equals TRUE and suitability_mismatch_flag equals FALSE) divided by (count of all transactions) multiplied by one hundred percent, Average Ticket Size calculated as sum of all subscription_amount values divided by count of executed transactions, Compliance Defect Ratio calculated as (count of transactions with compliance_defect_flag equals TRUE) divided by (count of transactions reviewed by compliance) multiplied by one hundred percent, and Average Processing Time calculated as mean of all total_processing_time_minutes values across executed transactions, with all metrics stored in data warehouse structures enabling trend analysis, performance benchmarking against industry standards, identification of process improvement opportunities, and regulatory reporting demonstrating adherence to investor protection obligations and operational excellence standards mandated by banking supervisory authorities.

        </SOP>'''
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
    n_samples = '20'
    direct_graph = '''
        <api_dependency_graph>
        ```yaml
        apis:
        - api_name: "ValidateApplicationData"
            description: "Validates application data format, completeness, and internal consistency including OCR cross-reference for digital applications"
            purpose: "Ensure data quality and consistency before initiating expensive external verification processes, preventing invalid data from propagating downstream"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_date_of_birth"
                data_type: "date"
                source: "user_input"
                required: true
            - name: "applicant_ssn_tax_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_residential_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_email"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "applicant_phone"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "id_document_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "id_document_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "id_document_expiry_date"
                data_type: "date"
                source: "user_input"
                required: true
            - name: "application_data_entry_method"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "id_ocr_extracted_data"
                data_type: "JSON"
                source: "user_input"
                required: false
            - name: "customer_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "business_type"
                data_type: "enum"
                source: "user_input"
                required: true
            output_parameters:
            - name: "data_validation_status"
                data_type: "enum"
                destination: "other_api"
            - name: "data_consistency_score"
                data_type: "float"
                destination: "other_api"
            - name: "validation_error_details"
                data_type: "string"
                destination: "storage"
            - name: "manual_review_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "manual_review_reason"
                data_type: "string"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "ExecuteCIPVerification"
            description: "Orchestrates comprehensive Customer Identification Program verification including document authentication, biometric liveness checks, facial matching, and credit bureau identity validation"
            purpose: "Satisfy regulatory CIP requirements under 31 CFR 1020.220 by establishing reasonable belief in customer identity through documentary and non-documentary methods"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_date_of_birth"
                data_type: "date"
                source: "other_api"
                required: true
            - name: "applicant_ssn_tax_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_residential_address"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "id_document_type"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "id_document_number"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "id_document_issuing_country"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "biometric_liveness_check_completed"
                data_type: "boolean"
                source: "user_input"
                required: false
            - name: "application_channel"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "data_consistency_score"
                data_type: "float"
                source: "other_api"
                required: true
            output_parameters:
            - name: "cip_verification_status"
                data_type: "enum"
                destination: "other_api"
            - name: "facial_match_confidence_score"
                data_type: "float"
                destination: "other_api"
            - name: "credit_bureau_validation_status"
                data_type: "enum"
                destination: "other_api"
            - name: "document_authenticity_score"
                data_type: "float"
                destination: "storage"
            - name: "manual_review_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "manual_review_reason"
                data_type: "string"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "ValidateApplicationData"
                dependency_type: "data_dependency"
                dependency_reason: "Requires validated and consistent application data as input to avoid processing invalid customer information through expensive external verification services"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ScreenWatchlists"
            description: "Performs OFAC sanctions screening and watchlist matching against Specially Designated Nationals List, global sanctions databases, and PEP lists using fuzzy matching algorithms"
            purpose: "Ensure regulatory compliance with OFAC requirements under 31 CFR Chapter V by identifying and blocking sanctioned individuals and entities"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_date_of_birth"
                data_type: "date"
                source: "other_api"
                required: false
            - name: "applicant_residential_address"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_ssn_tax_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "cip_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "business_legal_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "beneficial_owners"
                data_type: "list[JSON]"
                source: "user_input"
                required: false
            output_parameters:
            - name: "watchlist_screening_result"
                data_type: "enum"
                destination: "other_api"
            - name: "watchlist_match_details"
                data_type: "JSON"
                destination: "storage"
            - name: "similarity_scores"
                data_type: "list[float]"
                destination: "storage"
            - name: "manual_review_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "manual_review_reason"
                data_type: "string"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "ExecuteCIPVerification"
                dependency_type: "state_dependency"
                dependency_reason: "Only screens applications with verified identities (cip_verification_status = 'verified') to avoid wasting resources on invalid applications and ensure screening accuracy"
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "AssessRiskProfile"
            description: "Calculates comprehensive risk assessment including ChexSystems deposit account history score retrieval and multi-factor fraud risk scoring based on behavioral analytics, device fingerprinting, and data consistency analysis"
            purpose: "Quantify deposit account risk and fraud probability to support automated decisioning and manual review prioritization"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_ssn_tax_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_date_of_birth"
                data_type: "date"
                source: "other_api"
                required: true
            - name: "applicant_residential_address"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_employment_status"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "applicant_annual_income"
                data_type: "integer"
                source: "user_input"
                required: true
            - name: "initial_deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "customer_type"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "cip_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "credit_bureau_validation_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "facial_match_confidence_score"
                data_type: "float"
                source: "other_api"
                required: false
            - name: "id_ocr_extracted_data"
                data_type: "JSON"
                source: "user_input"
                required: false
            - name: "data_consistency_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "application_channel"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "business_industry"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "business_monthly_transaction_volume_estimate"
                data_type: "integer"
                source: "user_input"
                required: false
            output_parameters:
            - name: "chexsystems_score"
                data_type: "integer"
                destination: "other_api"
            - name: "fraud_risk_score"
                data_type: "float"
                destination: "other_api"
            - name: "due_diligence_level"
                data_type: "enum"
                destination: "other_api"
            - name: "risk_factor_breakdown"
                data_type: "JSON"
                destination: "storage"
            - name: "manual_review_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "manual_review_reason"
                data_type: "string"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "ExecuteCIPVerification"
                dependency_type: "data_dependency"
                dependency_reason: "Requires CIP outputs including credit_bureau_validation_status and facial_match_confidence_score as input factors for fraud risk calculation, and must verify identity before accessing ChexSystems consumer reports per FCRA requirements"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ExecuteAutoDecision"
            description: "Synthesizes all risk assessment metrics through multi-factor decision matrix to render automated approval, decline, or manual review referral decision based on institutional risk appetite thresholds"
            purpose: "Enable straight-through processing for low-risk applications while routing complex cases to compliance officers, optimizing operational efficiency while maintaining risk controls"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "cip_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "watchlist_screening_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "chexsystems_score"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "fraud_risk_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "credit_bureau_validation_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "facial_match_confidence_score"
                data_type: "float"
                source: "other_api"
                required: false
            - name: "due_diligence_level"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "customer_type"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "application_channel"
                data_type: "enum"
                source: "system_state"
                required: true
            output_parameters:
            - name: "auto_decision_result"
                data_type: "enum"
                destination: "other_api"
            - name: "composite_risk_score"
                data_type: "float"
                destination: "storage"
            - name: "decision_rationale"
                data_type: "string"
                destination: "storage"
            - name: "manual_review_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "manual_review_reason"
                data_type: "string"
                destination: "other_api"
            - name: "final_approval_status"
                data_type: "enum"
                destination: "other_api"
            - name: "decline_reason"
                data_type: "string"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ExecuteCIPVerification"
                dependency_type: "data_dependency"
                dependency_reason: "Requires cip_verification_status, credit_bureau_validation_status, and facial_match_confidence_score as critical decision inputs"
            - dependent_api: "ScreenWatchlists"
                dependency_type: "data_dependency"
                dependency_reason: "Requires watchlist_screening_result as mandatory compliance gate - confirmed matches trigger automatic decline"
            - dependent_api: "AssessRiskProfile"
                dependency_type: "data_dependency"
                dependency_reason: "Requires chexsystems_score, fraud_risk_score, and due_diligence_level to calculate composite risk and apply decision thresholds"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ProcessManualReview"
            description: "Facilitates compliance officer adjudication for applications referred for manual review, including case assignment, investigation workflow, documentation capture, and final approval decision rendering"
            purpose: "Enable expert human judgment for ambiguous risk scenarios including watchlist potential matches, elevated fraud scores, and enhanced due diligence cases requiring discretionary approval authority"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "auto_decision_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "manual_review_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "manual_review_reason"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "watchlist_screening_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "fraud_risk_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "chexsystems_score"
                data_type: "integer"
                source: "other_api"
                required: false
            - name: "due_diligence_level"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "documents_submitted"
                data_type: "list[JSON]"
                source: "user_input"
                required: true
            - name: "beneficial_owners"
                data_type: "list[JSON]"
                source: "user_input"
                required: false
            - name: "compliance_officer_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "final_approval_status"
                data_type: "enum"
                destination: "other_api"
            - name: "decline_reason"
                data_type: "string"
                destination: "storage"
            - name: "compliance_review_timestamp"
                data_type: "datetime"
                destination: "storage"
            - name: "compliance_review_notes"
                data_type: "string"
                destination: "storage"
            - name: "compliance_officer_id"
                data_type: "string"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "ExecuteAutoDecision"
                dependency_type: "state_dependency"
                dependency_reason: "Only executes when auto_decision_result equals 'refer_for_manual_review' or manual_review_required is true - creates conditional workflow branch"
            - dependent_api: "ScreenWatchlists"
                dependency_type: "data_dependency"
                dependency_reason: "Requires watchlist_screening_result and match details for potential match disambiguation investigations"
            - dependent_api: "AssessRiskProfile"
                dependency_type: "data_dependency"
                dependency_reason: "Requires fraud_risk_score, chexsystems_score, and due_diligence_level to understand risk factors requiring officer assessment"
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "CreateBankAccount"
            description: "Generates new account in Core Banking System of record including account number assignment, routing number association, product configuration, and initial account setup"
            purpose: "Establish the formal contractual banking relationship and system of record for deposit transactions following successful approval determination"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "final_approval_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_date_of_birth"
                data_type: "date"
                source: "other_api"
                required: true
            - name: "applicant_ssn_tax_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_residential_address"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_email"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_phone"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "account_type_requested"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "customer_type"
                data_type: "enum"
                source: "system_state"
                required: true
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "business_legal_name"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "business_ein"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "authorized_signers"
                data_type: "list[JSON]"
                source: "user_input"
                required: false
            output_parameters:
            - name: "account_number_generated"
                data_type: "string"
                destination: "other_api"
            - name: "routing_number"
                data_type: "string"
                destination: "other_api"
            - name: "account_creation_timestamp"
                data_type: "datetime"
                destination: "other_api"
            - name: "account_activation_status"
                data_type: "enum"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "ExecuteAutoDecision"
                dependency_type: "state_dependency"
                dependency_reason: "Cannot create account until final_approval_status equals 'approved' - either from auto-decision or manual review completion"
            - dependent_api: "ProcessManualReview"
                dependency_type: "state_dependency"
                dependency_reason: "For manually reviewed cases, requires final_approval_status from compliance officer adjudication before proceeding to account creation"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "ProcessAccountFunding"
            description: "Executes initial deposit processing based on specified funding method including cash/check posting, external ACH transfer initiation, or debit card authorization and updates account activation status"
            purpose: "Complete account funding to transition account from pending to active status and enable full transaction capabilities"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "account_number_generated"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "initial_deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "initial_deposit_method"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "external_bank_account_linked"
                data_type: "boolean"
                source: "user_input"
                required: false
            - name: "account_activation_status"
                data_type: "enum"
                source: "other_api"
                required: true
            output_parameters:
            - name: "account_funding_completed"
                data_type: "boolean"
                destination: "other_api"
            - name: "account_activation_status"
                data_type: "enum"
                destination: "storage"
            - name: "funding_transaction_id"
                data_type: "string"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CreateBankAccount"
                dependency_type: "data_dependency"
                dependency_reason: "Requires account_number_generated to post deposit transactions and update account balance - cannot fund non-existent account"
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "ProvisionAncillaryProducts"
            description: "Coordinates issuance of debit cards, online banking credential creation, check stock ordering, and welcome kit assembly for newly created accounts"
            purpose: "Enable full account functionality by provisioning payment instruments, digital access, and transaction tools requested during account opening"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "account_number_generated"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "routing_number"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_full_name"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_residential_address"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "applicant_email"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "account_type_requested"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "cross_sell_products_accepted"
                data_type: "list[string]"
                source: "user_input"
                required: true
            - name: "customer_type"
                data_type: "enum"
                source: "system_state"
                required: true
            output_parameters:
            - name: "debit_card_number_issued"
                data_type: "string"
                destination: "storage"
            - name: "online_banking_credentials_created"
                data_type: "boolean"
                destination: "storage"
            - name: "temporary_checks_issued"
                data_type: "boolean"
                destination: "storage"
            - name: "welcome_kit_issued"
                data_type: "boolean"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CreateBankAccount"
                dependency_type: "data_dependency"
                dependency_reason: "Requires account_number_generated and routing_number for debit card embossing, check printing, and online banking account association"
            complexity_level: "medium"
            business_criticality: "medium"

        - api_name: "CalculateProcessMetrics"
            description: "Computes process performance metrics including total processing time, first time yield, rework count, abandonment analysis, and updates all completion status fields"
            purpose: "Enable operational performance monitoring, SLA compliance tracking, and continuous improvement analytics by capturing comprehensive process efficiency metrics"
            input_parameters:
            - name: "application_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "application_start_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "account_creation_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: true
            - name: "account_funding_completed"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "final_approval_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "auto_decision_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "manual_review_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "customer_consent_forms_signed"
                data_type: "list[string]"
                source: "user_input"
                required: true
            - name: "application_abandoned"
                data_type: "boolean"
                source: "system_state"
                required: true
            - name: "abandonment_stage"
                data_type: "string"
                source: "system_state"
                required: false
            output_parameters:
            - name: "total_processing_time_minutes"
                data_type: "integer"
                destination: "storage"
            - name: "first_time_yield"
                data_type: "boolean"
                destination: "storage"
            - name: "rework_count"
                data_type: "integer"
                destination: "storage"
            - name: "welcome_kit_issued"
                data_type: "boolean"
                destination: "storage"
            - name: "application_abandoned"
                data_type: "boolean"
                destination: "storage"
            - name: "abandonment_stage"
                data_type: "string"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "CreateBankAccount"
                dependency_type: "data_dependency"
                dependency_reason: "Requires account_creation_timestamp to calculate total_processing_time_minutes by comparing with application_start_timestamp"
            - dependent_api: "ProcessAccountFunding"
                dependency_type: "data_dependency"
                dependency_reason: "Requires account_funding_completed status to determine final account_activation_status for metrics reporting"
            - dependent_api: "ExecuteAutoDecision"
                dependency_type: "data_dependency"
                dependency_reason: "Requires auto_decision_result and manual_review_required to calculate auto-approval rate and manual review percentage metrics"
            complexity_level: "low"
            business_criticality: "medium"
        ```
        </api_dependency_graph>
        '''
    yaml_file = "/Invest/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Dataset_Generation.yaml"
    system_prompt = "You are a SOP (Standard Operating Procedure) complete process format data generator. ***Please note that the format of each data entry in the dataset you generate should be exactly the same. Each data entry should be wrapped in [], with each field value conforming to the structure in the schema, each wrapped in '', and separated by English commas. **The input and output fields in each data entry must not be those not present in the schema**; only synthesize the field values within the <Existing_fields></Existing_fields> tags in the schema. For fields wrapped in <Field_to_be_obtained></Field_to_be_obtained>, only write the key, and leave the value empty. ***"



    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, sop_file_contents, additional_context, sample_schema, n_samples, direct_graph)

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
