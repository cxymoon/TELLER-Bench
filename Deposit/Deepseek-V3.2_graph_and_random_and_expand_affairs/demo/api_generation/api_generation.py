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

        This Standard Operating Procedure establishes the comprehensive, technically rigorous framework governing the end-to-end execution of banking deposit business handling processes across all transaction modalities, including but not limited to retail demand deposits, term deposit certificates, corporate bulk deposits, and physical cash counter operations. The procedure ensures strict adherence to regulatory mandates codified under Title 31 of the United States Code Section 5313 pertaining to Currency Transaction Reports, the Customer Identification Program requirements enumerated in 31 CFR 1020.220, and Anti-Money Laundering protocols enforced by the Financial Crimes Enforcement Network, while simultaneously optimizing operational efficiency through systematic validation checkpoints, multi-layered risk assessment algorithms, and real-time fraud detection mechanisms integrated within the Core Banking System infrastructure. The ultimate objective is to establish a defensible, auditable, and repeatable process that transforms raw deposit transaction data into verifiable financial instrument postings, generates immutable audit trails, and produces quantifiable outputs including updated ledger balances, compliance verification statuses, and transaction reference identifiers that can withstand regulatory scrutiny during periodic examinations conducted by federal banking supervisory authorities.

        2. Scope

        This Standard Operating Procedure applies comprehensively to all personnel operating within branch banking environments, including but not limited to Bank Tellers, Customer Service Representatives, Personal Bankers, Relationship Managers, Branch Managers with supervisory authority limits, Compliance Officers responsible for Know Your Customer analytical review, and System Administrators maintaining the operational integrity of Core Banking System platforms and peripheral transaction processing devices. The procedural framework encompasses all deposit transaction types enumerated within the deposit_type field specification, specifically including RETAIL_DEMAND transactions processed through BRANCH_TELLER and ATM channels, TERM_DEPOSIT instruments requiring maturity date calculations and Annual Percentage Yield determinations, CORPORATE_DEPOSIT operations involving complex authorization hierarchies, CASH_COUNTER transactions necessitating physical currency validation through discriminator hardware, and MOBILE_DEPOSIT operations requiring digital image capture validation. The geographic scope extends to all branch locations identified by unique branch_code designators operating under the institution's Federal Deposit Insurance Corporation charter, and temporal applicability commences immediately upon procedural publication and remains in force until superseded by subsequent regulatory-driven revisions or internal policy amendments. This procedure explicitly excludes wire transfer deposit crediting mechanisms governed by Federal Reserve Regulation J, automated clearing house transactions subject to NACHA operating rules, and correspondent banking deposit relationships managed under separate institutional agreements.

        3. Definitions

        The Customer Information File, universally abbreviated as CIF and represented within system architecture as the customer_id field, constitutes the authoritative master data repository containing all identifying characteristics, demographic attributes, relationship hierarchies, and historical interaction records associated with a unique banking relationship entity, serving as the foundational linkage mechanism across all account numbers, product holdings, and transaction histories maintained within the Core Banking System's relational database structure. The Annual Percentage Yield, denoted as APY and stored within the interest_rate_apy field, represents the normalized rate of return expressed as a percentage that accounts for the compounding frequency of interest crediting over a standardized twelve-month period, calculated according to the formula APY equals the quantity of one plus the periodic rate raised to the power of the number of compounding periods per year, minus one, and multiplied by one hundred to express as percentage terms. The Currency Transaction Report, mandated under 31 USC 5313 and triggered when the ctr_filing_required boolean flag evaluates to true based on deposit_amount exceeding the ctr_threshold_amount value of ten thousand United States dollars, constitutes a regulatory submission to the Financial Crimes Enforcement Network documenting the particulars of currency transactions that meet statutory reporting thresholds, thereby enabling pattern analysis for potential money laundering or terrorist financing activities. The Anti-Money Laundering risk score, quantified within the aml_risk_score integer field on a scale ranging from zero representing negligible risk to one hundred representing maximum risk exposure, represents the algorithmic output of probabilistic modeling that evaluates transaction characteristics, customer behavioral patterns, geographic risk factors, and historical compliance incident correlations to generate a composite risk assessment requiring manual review when exceeding predetermined institutional tolerance thresholds. The sanctions screening process, documented through the aml_screening_result enumeration and sanctions_screening_timestamp datetime fields, encompasses the automated comparison of customer identifying information including customer_name, customer_tax_id, and customer_address against consolidated watchlists maintained by the Office of Foreign Assets Control Specially Designated Nationals list, United Nations Security Council sanctions committees, Interpol notices, and other governmental restricted party databases, with results categorized as CLEAR indicating no matches, WATCHLIST_HIT indicating potential name correspondence requiring investigation, or MANUAL_REVIEW_REQUIRED indicating fuzzy matching algorithms detected sufficient similarity to warrant human analytical intervention. The General Ledger account, representing the double-entry accounting construct within which term deposit principal amounts are maintained separately from transactional demand deposit balances, utilizes distinct account classification codes to enable regulatory reporting segregation between interest-bearing and non-interest-bearing deposit categories as required under Federal Reserve Regulation D reserve requirement calculations. The fraud detection score, expressed as a floating-point value within the fraud_detection_score field ranging from zero point zero representing negligible fraud probability to one point zero representing maximum fraud likelihood, derives from machine learning models trained on historical fraud incident patterns that evaluate real-time transaction attributes including deposit_amount deviation from customer baseline behavior, deposit_channel consistency with established patterns, geographic location anomalies detected through branch_code analysis, temporal clustering of transactions, and device fingerprinting when applicable to electronic channels. The processing time measurement, captured in the processing_time_seconds integer field, represents the elapsed duration in seconds between the transaction_timestamp marking process initiation and the final system commit operation that generates the transaction_reference_id, serving as a key performance indicator for operational efficiency monitoring and staff productivity assessment against service level agreement targets.

        4. Input

        The initiation of the deposit business handling process mandates the availability of a complete and validated data set encompassing all fields enumerated within the Existing_fields specification of the transactional schema. The transaction_id serves as the immutable unique identifier establishing the atomic unit of processing and must conform to the alphanumeric format convention incorporating temporal encoding to ensure global uniqueness across the enterprise transaction repository. The customer_id derived from the Customer Information File must reference a pre-existing and active customer record within the Core Banking System master data tables, having previously undergone initial account establishment procedures that satisfied Customer Identification Program requirements. The account_number designating the target ledger account for deposit crediting must correspond to an account classification enumerated within the account_type field that permits deposit transactions according to product rules engine configurations, with validation confirming the account maintains an account_status of ACTIVE rather than DORMANT, FROZEN, or CLOSED states that would prohibit transaction posting. The deposit_amount expressed in floating-point precision must be accompanied by the corresponding deposit_currency code, with the system defaulting to United States Dollar denomination unless explicitly overridden for foreign currency transactions requiring real-time exchange rate application from treasury management systems. The funding_source_type enumeration determines the validation pathway, with CASH designations requiring accompanying cash_denomination_breakdown JSON structure detailing the count of each bill denomination to enable currency discriminator reconciliation and counterfeit_detection_result validation, whereas INTERNAL_TRANSFER designations mandate population of the funding_source_account field to enable sufficient funds verification against the source ledger balance. For term deposit transactions characterized by deposit_type of TERM_DEPOSIT, the term_length_months integer field must be populated with a value corresponding to available product offerings maintained within the interest rate matrix, triggering automatic calculation of the maturity_date through the addition of the specified month quantity to the transaction_timestamp and the retrieval of the corresponding interest_rate_apy from the treasury-approved rate table indexed by term length and current market conditions. The customer identification documentation fields including customer_id_type, customer_id_number, customer_address, customer_tax_id, customer_phone, and customer_email must be populated with data of sufficient quality and completeness to satisfy Know Your Customer regulatory standards, with the kyc_verification_status field indicating VERIFIED status as a prerequisite for transaction processing except in cases where PENDING status is permissible subject to subsequent verification completion within regulatory timeframes. The teller_id or banker employee identifier must be authenticated against active directory credentials to ensure proper authorization levels, audit trail attribution, and segregation of duties compliance, while the branch_code must correspond to a valid operating location within the institution's branch network to enable proper general ledger coding and regional performance attribution. The previous_account_balance field provides the baseline ledger balance against which the new_account_balance will be calculated through arithmetic addition of the deposit_amount subject to successful transaction approval, and this baseline value must be retrieved from the authoritative Core Banking System ledger in real-time immediately prior to transaction processing to prevent race conditions in concurrent transaction scenarios.

        5. Main Procedure

        5.1 Customer Identification and Know Your Customer Verification Execution

        Upon receipt of all input data elements as specified in Section 4, the system shall instantiate the Know Your Customer verification workflow by first validating the completeness and format conformity of the customer_id_type and customer_id_number fields against regulatory acceptable identification document standards, wherein PASSPORT designations must conform to International Civil Aviation Organization machine-readable zone specifications, DRIVERS_LICENSE must include state issuing authority validation, NATIONAL_ID must align with jurisdiction-specific formats, and SSN must satisfy Social Security Administration nine-digit format with hyphen placement at positions four and seven. The verification subsystem shall then execute a multi-source identity validation query against third-party data aggregators to confirm the correlation between the customer_name, customer_address, customer_tax_id, and the presented identification credentials, calculating a confidence score based on exact match criteria for name spelling, address street number and postal code concordance, and tax identifier checksum validation algorithms. In scenarios where the confidence score falls below the institutional threshold of eighty-five percent, the kyc_verification_status field shall be programmatically set to REQUIRES_REVIEW and the transaction shall be routed to the Compliance Officer queue for manual adjudication, whereas confidence scores at or above eighty-five percent shall permit automatic progression to kyc_verification_status of VERIFIED with the kyc_verification_timestamp populated with the current system datetime in ISO 8601 format with Zulu time zone designation. Concurrently with identity verification, the system shall extract the customer's historical banking relationship data from the Customer Relationship Management system to assess cross-selling opportunity alignment with the cross_sell_products_offered list field, algorithmically recommending products such as CREDIT_CARD based on deposit_amount thresholds exceeding five thousand dollars or PERSONAL_LOAN based on term deposit establishment indicating savings capacity, thereby optimizing the revenue potential of the customer interaction while maintaining primary focus on compliant transaction processing. The Customer Identification Program verification must also encompass beneficial ownership determination for CORPORATE_DEPOSIT transaction types, requiring the collection and validation of identifying information for all natural persons owning twenty-five percent or more equity interest and at least one individual exercising significant managerial control, with this extended verification documented through supplementary fields in the customer master record and subject to the same confidence scoring methodology applied to individual retail customers.

        5.2 Anti-Money Laundering and Sanctions Screening Protocol Implementation

        Following successful completion of Know Your Customer verification procedures as evidenced by kyc_verification_status field attaining VERIFIED enumeration, the system shall immediately invoke the Anti-Money Laundering screening engine by transmitting the customer_name, customer_address, customer_tax_id, and associated aliases or doing-business-as names to the consolidated sanctions database interface. The screening algorithm shall perform fuzzy matching logic employing Levenshtein distance calculations with a maximum edit distance of two characters to identify potential matches against the Office of Foreign Assets Control Specially Designated Nationals list, the United Nations Security Council consolidated sanctions list, the European Union financial sanctions targets, and proprietary watchlists maintained by the institution's Compliance department reflecting Politically Exposed Persons and high-risk jurisdiction nationals. When the fuzzy matching algorithm returns zero candidate matches, the aml_screening_result field shall be set to CLEAR and the sanctions_screening_timestamp shall be recorded, permitting immediate progression to subsequent processing stages; however, when one or more candidate matches are detected, the aml_screening_result shall be set to WATCHLIST_HIT and the transaction shall enter a suspended state pending Compliance Officer review to determine whether the match represents a true positive requiring transaction blocking and Suspicious Activity Report filing or a false positive permissible to override with documented rationale. Concurrent with watchlist screening, the system shall calculate the aml_risk_score integer value by executing a proprietary risk scoring model that evaluates multiple risk dimensions including but not limited to the deposit_amount as a percentage of the customer's historical average deposit size, the deposit_channel consistency with established behavioral patterns, the geographic risk rating associated with the branch_code location, the presence of cash as the funding_source_type which elevates risk relative to electronic transfer mechanisms, and the recency and frequency of previous transactions within a rolling thirty-day observation window. The risk scoring algorithm applies weighted coefficients to each risk dimension derived from institution-specific calibration against historical Suspicious Activity Report filings and regulatory enforcement actions, with the composite score calculated as the sum of the products of each dimension value and its corresponding weight, normalized to the zero-to-one-hundred integer scale through percentile transformation based on the current customer population distribution. When the calculated aml_risk_score exceeds the institutional threshold value of seventy-five, the system shall automatically set the aml_screening_result to MANUAL_REVIEW_REQUIRED regardless of watchlist matching results, thereby triggering human analytical review by a Compliance Officer who must document their risk assessment determination and either approve transaction continuation with enhanced monitoring or reject the transaction with appropriate customer notification and regulatory reporting if circumstances warrant.

        5.3 Currency Transaction Report Threshold Evaluation and Filing Determination

        In parallel with Anti-Money Laundering screening execution, the system shall perform real-time evaluation of Currency Transaction Report filing obligations by comparing the deposit_amount field value against the ctr_threshold_amount statutory limit of ten thousand United States dollars as codified in 31 USC 5313. When the deposit_amount equals or exceeds the ctr_threshold_amount and the funding_source_type is enumerated as CASH, the ctr_filing_required boolean flag shall be set to true, triggering the automatic generation of Financial Crimes Enforcement Network Form 112 with data population from the customer identifying fields including customer_name, customer_address, customer_tax_id, customer_id_type, and customer_id_number, along with transaction-specific details including transaction_timestamp, deposit_amount, branch_code, and teller_id for reporting attribution. The system shall additionally execute aggregation logic to detect potential structuring activities by querying the transaction database for all cash deposits attributed to the same customer_id within the preceding rolling twelve-month period, calculating the cumulative sum of deposit_amount values, and flagging for Compliance review any patterns indicating multiple transactions individually below the reporting threshold but collectively suggesting intentional avoidance of Currency Transaction Report filing obligations, such as deposit frequency clustering or amount patterns consistently ranging between nine thousand and nine thousand nine hundred ninety-nine dollars. For non-cash funding sources including CHECK, INTERNAL_TRANSFER, WIRE, and ACH enumerations, the ctr_filing_required flag shall default to false as these mechanisms create inherent audit trails through the banking system that satisfy Bank Secrecy Act transparency objectives without necessitating supplementary reporting; however, the system shall maintain algorithmic surveillance for unusual patterns in these transaction types that might indicate layering activities within money laundering typologies, particularly sequences of internal transfers followed by immediate cash withdrawals executed through different channels or locations. In scenarios where multiple deposits occurring on the same business day for the same customer_id collectively exceed the ctr_threshold_amount when individually below that threshold, the system shall apply aggregation rules mandated under 31 CFR 1010.311 to determine whether the transactions were conducted on behalf of the same person, evaluating factors including temporal proximity, branch_code consistency, and explicit customer statements documented in transaction_notes fields, with aggregated transactions triggering ctr_filing_required status and supplementary narrative explanation within the regulatory filing documenting the aggregation methodology and factual basis for person attribution.

        5.4 Physical Currency Validation and Counterfeit Detection Procedure

        For deposit transactions characterized by funding_source_type enumeration of CASH, the Bank Teller or Customer Service Representative must execute rigorous physical currency validation protocols utilizing certified currency discriminator hardware identified by the cash_counter_machine_id field. The teller shall manually organize the physical currency by denomination prior to machine processing, visually inspecting bills for obvious signs of alteration or reproduction including color inconsistencies, paper texture anomalies, missing or irregular security thread placement, watermark absence or misalignment, and microprinting degradation, with any bills exhibiting suspicious characteristics immediately segregated for supervisor evaluation and potential submission to United States Secret Service field offices for forensic analysis. The organized currency shall then be processed through the currency discriminator device in denomination-specific batches, with the machine executing multi-spectral imaging analysis encompassing ultraviolet fluorescence detection to validate embedded security fibers, infrared absorption spectroscopy to verify ink authenticity, magnetic sensing to detect ferrous content in genuine security threads, and dimensional measurement to confirm bill size conformity to Bureau of Engraving and Printing specifications of 156 millimeters by 66.3 millimeters with tolerances of plus or minus 0.5 millimeters. The currency discriminator shall generate both a count output enumerating the quantity of bills per denomination and a quality assessment categorizing each processed bill as GENUINE, SUSPECT, or REJECT based on the aggregated confidence level from the multi-spectral analysis battery. The teller must then reconcile the machine-generated denomination counts against the cash_denomination_breakdown JSON structure, verifying that the machine count matches the customer-declared or teller-estimated count within a tolerance of zero discrepancy, as any variance necessitates recount procedures and customer notification of the verified amount prior to transaction posting. When the currency discriminator categorizes any bills as SUSPECT or REJECT, the counterfeit_detection_result field shall be set to FAIL and the transaction processing shall immediately halt pending supervisor intervention, with the Branch Manager required to perform secondary validation using portable counterfeit detection devices employing alternative detection methodologies including size comparison templates, density measurement through weight verification, and manual tactile assessment of intaglio printing depth, ultimately determining whether to accept the currency with enhanced documentation, reject the bills and return them to the customer with law enforcement notification if counterfeiting is confirmed, or escalate to regional security management for adjudication in ambiguous cases. Only upon successful validation of all currency denominations with counterfeit_detection_result achieving PASS enumeration may the teller proceed to update the cash_drawer_total_after_transaction field by arithmetically adding the validated deposit_amount to the pre-transaction drawer balance, with this addition recorded in the teller balancing system to enable end-of-day reconciliation against general ledger cash vault accounts and daily settlement reporting to treasury management functions.

        5.5 Real-Time Fraud Detection Scoring and Behavioral Analytics Evaluation

        Contemporaneously with compliance screening and currency validation workflows, the system shall invoke real-time fraud detection algorithms to calculate the fraud_detection_score floating-point value through the evaluation of transaction behavioral characteristics against established customer baseline patterns and institutional fraud typology libraries. The fraud scoring engine shall retrieve the customer's historical transaction profile from the Customer Relationship Management data warehouse, calculating statistical measures including the mean and standard deviation of deposit_amount values over the trailing ninety-day period, the modal deposit_channel and deposit_type distributions, the geographic entropy of branch_code utilization reflecting branch location diversity, and the temporal patterns of transaction_timestamp clustering by hour-of-day and day-of-week dimensions. The current transaction's characteristics shall then be compared against these baseline measures through z-score calculations that quantify the number of standard deviations by which the current deposit_amount exceeds the historical mean, with z-scores exceeding 3.0 indicating statistically significant deviation that elevates fraud probability. Additional fraud indicators evaluated by the scoring algorithm include first-time utilization of previously unused deposit_channel values, geographic impossibility scenarios where sequential transactions attributed to the same customer_id occur at branch_code locations separated by distances exceeding feasible travel velocities within the elapsed time between transaction_timestamp values, and pattern matches against known fraud typologies such as account takeover sequences characterized by channel switching from established patterns to MOBILE_APP or ONLINE_BANKING channels followed by rapid liquidation attempts. The fraud detection engine shall aggregate these individual risk signals through a weighted logistic regression model trained on historical confirmed fraud cases, with model coefficients periodically recalibrated through machine learning retraining cycles incorporating newly identified fraud incidents and false positive feedback from fraud analyst case review outcomes. The resulting fraud_detection_score represents the predicted probability of fraudulent activity expressed on a zero-to-one scale, with institutional risk tolerance thresholds establishing decision rules wherein scores below 0.30 permit automatic transaction approval, scores between 0.30 and 0.70 trigger enhanced authentication requirements such as customer callback verification or one-time-password challenges, and scores exceeding 0.70 mandate transaction suspension pending fraud analyst investigation and customer contact to verify transaction legitimacy through out-of-band communication channels not susceptible to compromise by account takeover perpetrators.

        5.6 Supervisory Approval Routing and Authorization Limit Validation

        Based on the aggregated risk assessment outcomes from Know Your Customer verification, Anti-Money Laundering screening, Currency Transaction Report evaluation, counterfeit detection, and fraud scoring workflows, the system shall determine the necessity for supervisory approval by evaluating multiple conditional criteria against the supervisor_approval_required boolean flag. The approval routing logic shall first evaluate whether the deposit_amount exceeds the teller_id employee's individual transaction authority limit as configured in the employee authorization matrix, with standard teller authority limits typically established at five thousand dollars for RETAIL_DEMAND transactions and ten thousand dollars for TERM_DEPOSIT transactions, while Personal Bankers and Relationship Managers receive elevated authority limits of twenty-five thousand dollars and fifty thousand dollars respectively based on their enhanced training and relationship management responsibilities. When deposit_amount exceeds the applicable authority limit for the teller_id processing the transaction, the supervisor_approval_required flag shall be set to true regardless of other risk indicators. Additionally, the system shall set supervisor_approval_required to true when any of the following conditions evaluate as true: the aml_screening_result equals WATCHLIST_HIT or MANUAL_REVIEW_REQUIRED, the aml_risk_score exceeds seventy-five, the fraud_detection_score exceeds 0.70, the counterfeit_detection_result equals FAIL, the kyc_verification_status equals REQUIRES_REVIEW, or the ctr_filing_required flag equals true in conjunction with transaction_notes field containing keywords such as "unusual," "suspicious," or "multiple parties" that indicate teller concern warranting management awareness. When supervisor_approval_required evaluates to true, the system shall route the transaction to the Branch Manager queue with notification including all relevant risk scores, screening results, and teller observations, suspending further processing until the Branch Manager with supervisor_id credentials reviews the transaction details, performs supplementary verification procedures as deemed necessary including customer interview, source of funds documentation review, or consultation with regional compliance management, and either approves the transaction by entering their supervisor_id and triggering supervisor_approval_timestamp population with current system datetime, or rejects the transaction by documenting the rejection rationale in the approval_rejection_reason field and initiating customer notification protocols. The supervisor approval interface must present the Branch Manager with comprehensive contextual information including the customer's complete relationship profile, historical transaction patterns, current account balances across all products, any previous Suspicious Activity Report filings or fraud incidents, and comparative analytics showing how the current transaction's characteristics deviate from both individual customer baselines and broader customer segment norms, thereby enabling informed risk-based decision making that balances fraud prevention, compliance obligations, and customer service quality considerations. In scenarios where the Branch Manager's authority limit is also exceeded, typically established at one hundred thousand dollars for single transactions, the system shall implement tiered escalation to Regional Manager or Chief Risk Officer approval queues with corresponding increases in supervisor_id authority level requirements and documentation standards.

        5.7 Term Deposit Specific Processing and Interest Calculation Execution

        For transactions where the deposit_type field is enumerated as TERM_DEPOSIT, the system shall execute specialized processing logic to generate certificate instruments, calculate maturity parameters, and establish interest crediting schedules. The term_length_months integer value shall serve as the index key to retrieve the corresponding interest_rate_apy from the current rate matrix maintained by treasury management functions, with rate tables typically differentiating across term length buckets of 3-month, 6-month, 12-month, 24-month, 36-month, and 60-month offerings, and further stratified by deposit_amount tier thresholds wherein principal amounts exceeding twenty-five thousand dollars, fifty thousand dollars, and one hundred thousand dollars qualify for progressively enhanced rate premiums reflecting the institution's liquidity preference and cost of funds optimization objectives. The system shall calculate the maturity_date by applying calendar month addition logic to the transaction_timestamp, incrementing the month component by the term_length_months value while preserving the day-of-month component, with special handling for month-end scenarios where the origination date falls on day 29, 30, or 31 of months lacking equivalent calendar dates requiring adjustment to the final day of the destination month in accordance with modified following business day convention to prevent maturity dates from falling on non-business days when interest payments cannot be processed. The calculated_interest_at_maturity field shall be computed using the compound interest formula A equals P multiplied by the quantity of one plus the quotient of r divided by n, raised to the power of the product of n multiplied by t, where P represents the deposit_amount principal, r represents the interest_rate_apy expressed in decimal form, n represents the compounding frequency per year derived from product configuration specifications, and t represents the term_length_months divided by twelve to express the term in years, with the final calculated_interest_at_maturity value determined by subtracting the original principal P from the maturity amount A. The system shall generate a unique cd_certificate_number following the alphanumeric format convention CD_CERT concatenated with the current year and a sequential identifier, populating this value in both the transaction record and a separate certificate of deposit instrument table that maintains the contractual terms including principal, rate, maturity date, and automatic renewal instructions if elected by the customer. The term deposit contract generation must also populate the early_withdrawal_penalty_rate field based on product terms, typically specified as a fraction of the interest that would have been earned, such as ninety days of interest for terms less than twelve months or one hundred eighty days of interest for longer terms, with this penalty provision disclosed to the customer through Truth in Savings Act-compliant documentation presented via the receipt_delivery_method channel. The system shall establish automated calendar triggers based on the maturity_date to generate customer notifications at thirty days, fourteen days, and seven days prior to maturity, informing the customer of impending maturity and presenting renewal options, automatic rollover elections, or liquidation preferences to optimize deposit retention and minimize operational friction associated with maturity processing.

        5.8 Account Balance Update and General Ledger Posting Execution

        Upon successful completion of all validation, screening, approval, and product-specific processing workflows, the system shall execute the atomic transaction commit operation that updates the account ledger and generates final transaction outputs. The new_account_balance field shall be calculated through arithmetic precision addition of the deposit_amount to the previous_account_balance retrieved from the authoritative Core Banking System ledger table, with this calculation performed using decimal data type arithmetic to prevent floating-point rounding errors that could accumulate across transaction sequences and result in balance discrepancies during reconciliation procedures. The ledger posting operation must implement double-entry accounting principles by simultaneously debiting the appropriate asset account based on funding_source_type, wherein CASH funding debits the branch vault general ledger account identified by concatenating branch_code with the cash asset account code, INTERNAL_TRANSFER funding debits the funding_source_account customer ledger, and CHECK funding debits the check clearing suspense account pending subsequent check collection and Federal Reserve settlement, while the corresponding credit entry posts to the customer's deposit account general ledger identified by the account_number and account_type combination that determines the appropriate interest-bearing or non-interest-bearing classification required for regulatory reserve requirement reporting under Federal Reserve Regulation D. The system shall generate a globally unique transaction_reference_id following the format convention REF concatenated with the transaction_timestamp formatted as YYYYMMDDHHmmss and a six-digit sequential identifier, with this reference number serving as the immutable pointer to the complete transaction record including all validation results, approval chains, and audit trail elements that establish the evidentiary foundation for transaction legitimacy during subsequent examination procedures. The account_status field shall be confirmed or updated to ACTIVE enumeration, with dormancy timers reset based on the transaction_timestamp to prevent automated account closure procedures that trigger when no customer-initiated transactions occur within statutorily defined dormancy periods varying by state escheatment laws but typically ranging from three to five years. For newly established accounts where this deposit transaction represents the initial funding event, the system shall trigger ancillary provisioning workflows including debit_card_issued flag activation that initiates card production batch processing with the debit_card_number assigned through Luhn algorithm-compliant card number generation and linked to the account_number for point-of-sale and ATM transaction authorization, online_banking_enabled flag activation that provisions user credentials and establishes security question protocols, mobile_app_enabled flag activation that generates device registration tokens, and checkbook_ordered flag processing that submits printing orders to the check stock vendor with appropriate customer_name and customer_address imprinting specifications. The system shall record the processing_time_seconds by calculating the elapsed duration between the transaction_timestamp and the current system time at the moment of final ledger commit, capturing this performance metric for service level agreement monitoring and process optimization analysis that identifies bottlenecks in validation or approval workflows warranting remediation through system enhancement or procedural refinement.

        5.9 Compliance Verification Aggregation and Final Approval Determination

        The system shall perform a comprehensive compliance verification aggregation by evaluating the boolean state of all mandatory control checkpoints prior to final transaction approval, implementing a logical AND operation across multiple condition predicates. The compliance_check_passed boolean flag shall be set to true only when all of the following conditions simultaneously evaluate as true: the kyc_verification_status equals VERIFIED, the aml_screening_result equals CLEAR or has received documented Compliance Officer override approval in the case of WATCHLIST_HIT or MANUAL_REVIEW_REQUIRED results, the counterfeit_detection_result equals PASS or NOT_APPLICABLE based on funding_source_type, the fraud_detection_score is below the institutional tolerance threshold of 0.70 or has received fraud analyst override approval with documented rationale, and either supervisor_approval_required equals false or when true has been satisfied through valid supervisor_id entry and supervisor_approval_timestamp population. When compliance_check_passed evaluates to true, the transaction_approved boolean flag shall be set to true, the approval_rejection_reason field shall remain null, and the system shall proceed with ledger posting and output generation as specified in Section 5.8; conversely, when compliance_check_passed evaluates to false due to any control checkpoint failure, the transaction_approved flag shall be set to false, the approval_rejection_reason field shall be populated with a structured text string identifying the specific control failure condition such as "Failed AML screening - WATCHLIST_HIT without override approval," "Counterfeit currency detected during validation," "Fraud score exceeds threshold without analyst approval," or "KYC verification incomplete - customer documentation insufficient," and the system shall initiate transaction rejection protocols including customer notification via the receipt_delivery_method channel, reversal of any preliminary ledger entries made during processing, return of physical currency to the customer when applicable, and generation of audit trail entries documenting the rejection decision for subsequent compliance monitoring and quality assurance review. The compliance verification aggregation must also evaluate regulatory filing completeness, confirming that when ctr_filing_required equals true, the Currency Transaction Report has been successfully submitted to the Financial Crimes Enforcement Network BSA E-Filing system with confirmation number received and stored in supplementary audit tables, as failure to timely file required reports constitutes a separate violation necessitating self-disclosure to regulatory authorities and potential civil monetary penalties under Bank Secrecy Act enforcement provisions.

        5.10 Receipt Generation and Transaction Documentation Finalization

        Upon successful transaction approval as evidenced by transaction_approved boolean flag evaluating to true, the system shall generate comprehensive transaction documentation for customer delivery and internal audit retention. The receipt_number shall be generated following the format convention RCPT concatenated with the transaction_timestamp formatted as YYYYMMDD, the branch_code, and a three-digit sequential counter representing the transaction's position in the branch's daily processing sequence, thereby creating a receipt numbering schema that enables rapid retrieval of transaction records based on branch location and date parameters during research or dispute resolution procedures. The receipt generation engine shall retrieve and format all relevant transaction details including the customer_name, account_number rendered in masked format displaying only the final four digits to protect account confidentiality, the deposit_amount formatted with currency symbol and comma thousand separators, the new_account_balance similarly formatted, the transaction_reference_id for customer record keeping and future inquiry reference, the transaction_timestamp formatted in human-readable twelve-hour clock format with meridiem indicator and time zone designation, and for term deposit transactions the additional elements of cd_certificate_number, maturity_date formatted as full month name and four-digit year, interest_rate_apy formatted as percentage with two decimal precision, and calculated_interest_at_maturity formatted as currency value. The receipt layout must incorporate all mandatory Truth in Savings Act disclosures for interest-bearing accounts including the Annual Percentage Yield expressed as APY followed by the numeric value, frequency of rate changes if variable rate products, balance computation method used for interest calculation, minimum balance requirements to obtain advertised APY, effect of fees on earnings, and features of time deposit accounts including term length, maturity date, and early withdrawal penalties expressed both as interest forfeiture period and potential principal reduction scenarios for significant early withdrawal events. The system shall execute receipt delivery via the receipt_delivery_method specified during transaction processing, wherein PRINTED enumeration triggers physical receipt output to the teller thermal printer device, EMAIL enumeration transmits formatted PDF attachment to the customer_email address with appropriate email security protocols including transport layer encryption and sender policy framework validation, SMS enumeration sends abbreviated transaction confirmation containing the transaction_reference_id and new_account_balance to the customer_phone number via secure short message service gateway, and MOBILE_APP enumeration pushes notification through the institution's mobile application programming interface to the customer's authenticated device with receipt details accessible through in-app document repository. Concurrent with customer receipt generation, the system shall populate the transaction_notes field with any teller-entered observations, customer requests, or service quality notations such as "Customer requested expedited card delivery," "Discussed certificate of deposit laddering strategy," or "Referred customer to mortgage department for home loan consultation," thereby capturing contextual information valuable for relationship management, service quality monitoring, and future customer interaction personalization. The system shall also update the customer's profile within the Customer Relationship Management platform to reflect the new deposit relationship, incrementing product holdings counts, recalculating total relationship value through aggregation of all deposit and loan balances, and triggering next-best-action recommendation algorithms that suggest appropriate follow-up activities such as wealth management consultations for customers exceeding one hundred thousand dollars in total deposits or insurance cross-selling opportunities based on life event indicators detected through transaction pattern analysis.

        5.11 Audit Trail Generation and Regulatory Record Retention Establishment

        The system shall create comprehensive audit trail documentation capturing the complete transaction lifecycle from initiation through final approval and posting, maintaining immutable records in compliance with Bank Secrecy Act requirements for minimum five-year retention of deposit account records and transaction documentation. The audit trail data structure shall incorporate all fields from both the Existing_fields and Field_to_be_obtained specifications, along with supplementary metadata including the Internet Protocol address of the workstation or mobile device initiating the transaction when applicable to electronic channels, browser user agent strings and device fingerprints for online and mobile transactions enabling device-based fraud pattern detection, the system version identifiers of the Core Banking System and peripheral applications involved in transaction processing to support incident investigation and system change correlation analysis, and the complete chain of approval events with corresponding supervisor_id values and supervisor_approval_timestamp entries creating a verifiable authorization hierarchy that demonstrates adherence to segregation of duties principles. The audit trail must preserve the pre-transaction state snapshot including the previous_account_balance value and account_status enumeration to enable point-in-time balance reconstruction and discrepancy investigation, while also maintaining complete records of all validation check results including the specific watchlist entries triggering WATCHLIST_HIT results during sanctions screening, the dimensional scores contributing to the aggregated aml_risk_score and fraud_detection_score calculations to enable model performance analysis and false positive rate optimization, and the individual currency denomination counts from the cash_denomination_breakdown JSON structure to support cash handling error investigation and teller performance assessment. The system shall implement blockchain-inspired data integrity controls through cryptographic hashing of the complete audit trail record upon transaction finalization, with the resulting hash value stored in tamper-evident distributed ledger infrastructure that enables verification of record authenticity and detection of any post-transaction data modification attempts that would invalidate the hash chain continuity. The audit trail storage architecture must support efficient retrieval through multiple index dimensions including transaction_id primary key access, customer_id indexing for relationship-level transaction history queries, transaction_timestamp chronological indexing for time-series analysis and regulatory examination period-based extraction, and teller_id indexing for employee performance monitoring and quality assurance sampling, with query performance requirements mandating maximum response times below two seconds for single transaction retrieval and below thirty seconds for complex multi-dimensional aggregation queries spanning annual reporting periods. The system shall establish automated data lifecycle management protocols that transition audit records through progressively cost-optimized storage tiers based on age and access frequency, maintaining current year records in high-performance solid-state storage for real-time operational access, migrating prior year records to standard magnetic disk storage for regulatory examination support, archiving records beyond two years old to lower-cost object storage platforms while preserving search index metadata, and ultimately transitioning records exceeding the regulatory retention period to long-term archive on write-once-read-many media for potential legal hold scenarios while purging from active operational databases to optimize system performance and storage costs.

        6. Output

        The successful completion of the deposit business handling process shall generate a comprehensive set of definite, quantifiable outputs that establish verifiable evidence of transaction execution and regulatory compliance. The transaction_approved boolean output shall definitively indicate either true for successful deposit processing or false for rejected transactions, with the corresponding approval_rejection_reason field providing null value for approved transactions or specific rejection rationale text string for declined transactions, enabling categorical classification of transaction outcomes for success rate monitoring and rejection reason trend analysis. The new_account_balance output shall represent the precise post-transaction ledger balance expressed as a floating-point numeric value with two decimal precision reflecting United States Dollar cent-level accuracy, calculated through the arithmetic sum of the previous_account_balance plus the deposit_amount for successful transactions or remaining unchanged at the previous_account_balance value for rejected transactions, providing customers with definitive confirmation of their current available funds subject to funds availability schedule overlays required by Federal Reserve Regulation CC for check deposits. The transaction_reference_id output shall constitute a unique alphanumeric identifier string following the format REF_YYYYMMDDHHmmssnnnnnn where the nnnnnn component represents a six-digit sequential number, serving as the permanent reference pointer for customer inquiries, dispute resolution procedures, and regulatory examination transaction sampling. For term deposit transactions specifically, the cd_certificate_number output shall be generated as a unique identifier string following format CD_CERT_YYYY_nnnnnn, and the calculated_interest_at_maturity output shall be expressed as a precise floating-point numeric value representing the total interest earnings in United States Dollars that will accrue over the full term calculated using the formula I equals P multiplied by the quantity one plus the quotient of r divided by n raised to the power of n multiplied by t, then subtracting P, where this calculated value provides customers with definitive projection of investment returns assuming no early withdrawal events. The account_status output shall be definitively set to the enumerated value ACTIVE for newly established accounts or existing accounts receiving deposits, confirming the account's operational availability for subsequent transaction processing and automated payment authorization. The processing_time_seconds output shall quantify the total elapsed duration as an integer value representing the number of seconds between transaction_timestamp and the system commit timestamp, with typical values ranging from 60 seconds for straightforward retail demand deposits to 300 seconds for complex term deposit transactions requiring rate lookup, maturity calculation, and certificate generation, providing operational performance metrics for service level agreement compliance assessment and process improvement prioritization. The cash_drawer_total_after_transaction output shall represent the precise floating-point numeric value of total currency held in the teller's cash drawer following the deposit transaction, calculated by adding the deposit_amount to the pre-transaction drawer total for cash deposits or remaining unchanged for non-cash funding sources, enabling teller balancing reconciliation at end-of-day through comparison against the sum of all transaction-level drawer impacts and the opening cash allocation. The fraud_detection_score output shall be expressed as a floating-point value between 0.0 and 1.0 with four decimal precision, representing the probabilistic assessment of fraudulent activity risk derived from behavioral analytics algorithms, with values such as 0.0847 indicating low risk, 0.4523 indicating moderate risk warranting enhanced monitoring, and 0.8912 indicating high risk requiring investigation. The compliance_check_passed output shall be a boolean value definitively indicating true when all regulatory validation checkpoints including Know Your Customer verification, Anti-Money Laundering screening, sanctions list checking, and fraud scoring have been successfully satisfied, or false when any compliance control has failed, providing aggregated compliance status for audit documentation and regulatory examination evidence. The final_transaction_fee output shall quantify any charges assessed to the customer as a floating-point numeric value, typically 0.00 for standard deposit transactions but potentially reflecting non-zero amounts for specialized services such as currency exchange fees for foreign currency deposits, expedited availability fees for next-day check clearing, or account research fees for complex deposit source verification requirements. The receipt_number output shall be generated as a unique alphanumeric string following format RCPT_YYYYMMDD_branchcode_nnn where the nnn component represents a three-digit sequence number, providing customers with a reference identifier for transaction confirmation and enabling branch-level daily reconciliation through sequential receipt number gap analysis to detect missing or voided transactions requiring investigation.

        </SOP>'''
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
    sample_data = '''
        <data>
        ['transaction_id:TXN_DEP_20250418_001234', 'customer_id:CIF_789456123', 'account_number:4532100012345678', 'transaction_timestamp:2025-04-18T10:23:45Z', 'deposit_type:CASH_COUNTER', 'account_type:SAVINGS', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_NY_5012', 'teller_id:EMP_45823', 'customer_name:John Michael Smith', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:D1234567', 'customer_address:123 Main Street, New York, NY 10001', 'customer_tax_id:123-45-6789', 'customer_phone:+1-212-555-0123', 'customer_email:john.smith@email.com', 'deposit_amount:5000.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 50, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:2.50', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:1234.56', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T10:15:32Z', 'aml_screening_result:CLEAR', 'aml_risk_score:15', 'sanctions_screening_timestamp:2025-04-18T10:16:05Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR5012_01', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-1234', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["CREDIT_CARD", "PERSONAL_LOAN"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Customer requested expedited card delivery', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_002156', 'customer_id:CIF_445789321', 'account_number:4532100098765432', 'transaction_timestamp:2025-04-18T11:45:23Z', 'deposit_type:TERM_DEPOSIT', 'account_type:CD_1Y', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_CA_3045', 'teller_id:EMP_67234', 'customer_name:Sarah Elizabeth Johnson', 'customer_id_type:PASSPORT', 'customer_id_number:P87654321', 'customer_address:456 Oak Avenue, Los Angeles, CA 90001', 'customer_tax_id:987-65-4321', 'customer_phone:+1-310-555-7890', 'customer_email:sarah.johnson@email.com', 'deposit_amount:25000.00', 'deposit_currency:USD', 'funding_source_type:INTERNAL_TRANSFER', 'funding_source_account:4532100087654321', 'cash_denomination_breakdown:', 'term_length_months:12', 'interest_rate_apy:4.25', 'maturity_date:', 'early_withdrawal_penalty_rate:0.5', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:0.00', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T11:38:12Z', 'aml_screening_result:CLEAR', 'aml_risk_score:22', 'sanctions_screening_timestamp:2025-04-18T11:39:45Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:True', 'supervisor_id:EMP_12034', 'supervisor_approval_timestamp:2025-04-18T11:47:56Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:False', 'debit_card_number:', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["WEALTH_MANAGEMENT", "INVESTMENT_ADVISORY"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:Customer interested in CD laddering strategy', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_003789', 'customer_id:CIF_223344556', 'account_number:4532100055667788', 'transaction_timestamp:2025-04-18T14:32:18Z', 'deposit_type:CASH_COUNTER', 'account_type:CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_TX_7823', 'teller_id:EMP_89234', 'customer_name:Michael Robert Davis', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:TX9876543', 'customer_address:789 Pine Street, Houston, TX 77001', 'customer_tax_id:456-78-9012', 'customer_phone:+1-713-555-4567', 'customer_email:michael.davis@email.com', 'deposit_amount:15000.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 150, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:0.25', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:10.00', 'previous_account_balance:3456.78', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T14:28:45Z', 'aml_screening_result:CLEAR', 'aml_risk_score:45', 'sanctions_screening_timestamp:2025-04-18T14:29:12Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR7823_02', 'supervisor_approval_required:True', 'supervisor_id:EMP_34567', 'supervisor_approval_timestamp:2025-04-18T14:35:23Z', 'ctr_filing_required:True', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-7788', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:True', 'cross_sell_products_offered:["MORTGAGE", "AUTO_LOAN"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Large cash deposit - CTR filed per regulatory requirements', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_004523', 'customer_id:CIF_998877665', 'account_number:4532100044332211', 'transaction_timestamp:2025-04-18T09:15:34Z', 'deposit_type:RETAIL_DEMAND', 'account_type:SAVINGS', 'deposit_channel:ATM', 'branch_code:BR_FL_4512', 'teller_id:', 'customer_name:Jennifer Anne Wilson', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:F8765432', 'customer_address:321 Beach Boulevard, Miami, FL 33101', 'customer_tax_id:321-54-9876', 'customer_phone:+1-305-555-8901', 'customer_email:jennifer.wilson@email.com', 'deposit_amount:750.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 7, "20": 2, "10": 1, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:2.75', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:2345.67', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T09:12:23Z', 'aml_screening_result:CLEAR', 'aml_risk_score:8', 'sanctions_screening_timestamp:2025-04-18T09:13:01Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-2211', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:[]', 'customer_signature_captured:False', 'signature_capture_method:NOT_CAPTURED', 'receipt_delivery_method:PRINTED', 'transaction_notes:ATM deposit processed successfully', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_005678', 'customer_id:CIF_556677889', 'account_number:4532100099887766', 'transaction_timestamp:2025-04-18T15:48:22Z', 'deposit_type:CORPORATE_DEPOSIT', 'account_type:BUSINESS_CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_IL_6789', 'teller_id:EMP_23456', 'customer_name:TechCorp Solutions LLC', 'customer_id_type:NATIONAL_ID', 'customer_id_number:EIN36-7894523', 'customer_address:1000 Corporate Drive, Chicago, IL 60601', 'customer_tax_id:36-7894523', 'customer_phone:+1-312-555-2345', 'customer_email:finance@techcorp.com', 'deposit_amount:125000.00', 'deposit_currency:USD', 'funding_source_type:CHECK', 'funding_source_account:', 'cash_denomination_breakdown:', 'term_length_months:', 'interest_rate_apy:1.50', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:5000.00', 'monthly_maintenance_fee:25.00', 'previous_account_balance:87654.32', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T15:42:11Z', 'aml_screening_result:CLEAR', 'aml_risk_score:35', 'sanctions_screening_timestamp:2025-04-18T15:43:56Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:True', 'supervisor_id:EMP_78901', 'supervisor_approval_timestamp:2025-04-18T15:52:45Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-7766', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:True', 'cross_sell_products_offered:["BUSINESS_LINE_OF_CREDIT", "MERCHANT_SERVICES"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:Large corporate check deposit - payroll funding', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_006234', 'customer_id:CIF_112233445', 'account_number:4532100012121212', 'transaction_timestamp:2025-04-18T13:22:11Z', 'deposit_type:CASH_COUNTER', 'account_type:CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_WA_8901', 'teller_id:EMP_56789', 'customer_name:David Lee Anderson', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:WA1234567', 'customer_address:555 Market Street, Seattle, WA 98101', 'customer_tax_id:789-01-2345', 'customer_phone:+1-206-555-6789', 'customer_email:david.anderson@email.com', 'deposit_amount:9800.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 98, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:0.15', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:12.00', 'previous_account_balance:456.89', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T13:18:34Z', 'aml_screening_result:CLEAR', 'aml_risk_score:52', 'sanctions_screening_timestamp:2025-04-18T13:19:23Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR8901_01', 'supervisor_approval_required:True', 'supervisor_id:EMP_45678', 'supervisor_approval_timestamp:2025-04-18T13:26:45Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-1212', 'online_banking_enabled:True', 'mobile_app_enabled:False', 'checkbook_ordered:True', 'cross_sell_products_offered:["SAVINGS_ACCOUNT"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Cash deposit just below CTR threshold - monitoring for structuring', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_007890', 'customer_id:CIF_667788990', 'account_number:4532100033445566', 'transaction_timestamp:2025-04-18T10:55:47Z', 'deposit_type:TERM_DEPOSIT', 'account_type:CD_6M', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_MA_2341', 'teller_id:EMP_12345', 'customer_name:Patricia Marie Thompson', 'customer_id_type:PASSPORT', 'customer_id_number:P12345678', 'customer_address:777 Commonwealth Avenue, Boston, MA 02215', 'customer_tax_id:234-56-7890', 'customer_phone:+1-617-555-3456', 'customer_email:patricia.thompson@email.com', 'deposit_amount:10000.00', 'deposit_currency:USD', 'funding_source_type:INTERNAL_TRANSFER', 'funding_source_account:4532100099998888', 'cash_denomination_breakdown:', 'term_length_months:6', 'interest_rate_apy:3.50', 'maturity_date:', 'early_withdrawal_penalty_rate:0.5', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:0.00', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T10:52:12Z', 'aml_screening_result:CLEAR', 'aml_risk_score:18', 'sanctions_screening_timestamp:2025-04-18T10:52:56Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:False', 'debit_card_number:', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["IRA_ACCOUNT", "RETIREMENT_PLANNING"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:6-month CD opening for short-term savings goal', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_008456', 'customer_id:CIF_334455667', 'account_number:4532100055443322', 'transaction_timestamp:2025-04-18T16:12:33Z', 'deposit_type:CASH_COUNTER', 'account_type:SAVINGS', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_AZ_5678', 'teller_id:EMP_67890', 'customer_name:Robert James Martinez', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:AZ7654321', 'customer_address:888 Desert Road, Phoenix, AZ 85001', 'customer_tax_id:567-89-0123', 'customer_phone:+1-602-555-7890', 'customer_email:robert.martinez@email.com', 'deposit_amount:3200.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 32, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:2.85', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:8765.43', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T16:08:45Z', 'aml_screening_result:CLEAR', 'aml_risk_score:12', 'sanctions_screening_timestamp:2025-04-18T16:09:34Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR5678_01', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-3322', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["CREDIT_CARD"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Regular savings deposit', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_009123', 'customer_id:CIF_889900112', 'account_number:4532100077665544', 'transaction_timestamp:2025-04-18T12:34:56Z', 'deposit_type:MOBILE_DEPOSIT', 'account_type:CHECKING', 'deposit_channel:MOBILE_APP', 'branch_code:BR_NC_3456', 'teller_id:', 'customer_name:Emily Grace Thompson', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:NC5432109', 'customer_address:123 University Drive, Charlotte, NC 28202', 'customer_tax_id:890-12-3456', 'customer_phone:+1-704-555-0123', 'customer_email:emily.thompson@email.com', 'deposit_amount:1250.00', 'deposit_currency:USD', 'funding_source_type:CHECK', 'funding_source_account:', 'cash_denomination_breakdown:', 'term_length_months:', 'interest_rate_apy:0.10', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:100.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:345.67', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T12:32:11Z', 'aml_screening_result:CLEAR', 'aml_risk_score:6', 'sanctions_screening_timestamp:2025-04-18T12:32:45Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-5544', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["STUDENT_LOAN"]', 'customer_signature_captured:False', 'signature_capture_method:NOT_CAPTURED', 'receipt_delivery_method:MOBILE_APP', 'transaction_notes:Mobile check deposit - paycheck', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_010789', 'customer_id:CIF_445566778', 'account_number:4532100011223344', 'transaction_timestamp:2025-04-18T08:45:12Z', 'deposit_type:TERM_DEPOSIT', 'account_type:CD_5Y', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_PA_7890', 'teller_id:EMP_34567', 'customer_name:William Charles Brown', 'customer_id_type:SSN', 'customer_id_number:901-23-4567', 'customer_address:999 Liberty Street, Philadelphia, PA 19101', 'customer_tax_id:901-23-4567', 'customer_phone:+1-215-555-4567', 'customer_email:william.brown@email.com', 'deposit_amount:75000.00', 'deposit_currency:USD', 'funding_source_type:WIRE', 'funding_source_account:', 'cash_denomination_breakdown:', 'term_length_months:60', 'interest_rate_apy:5.00', 'maturity_date:', 'early_withdrawal_penalty_rate:0.5', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:0.00', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T08:40:23Z', 'aml_screening_result:CLEAR', 'aml_risk_score:28', 'sanctions_screening_timestamp:2025-04-18T08:41:12Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:True', 'supervisor_id:EMP_90123', 'supervisor_approval_timestamp:2025-04-18T08:48:34Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:False', 'debit_card_number:', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["WEALTH_MANAGEMENT", "TRUST_SERVICES"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:5-year CD for retirement planning - maximum rate', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_011456', 'customer_id:CIF_778899001', 'account_number:4532100066778899', 'transaction_timestamp:2025-04-18T14:08:27Z', 'deposit_type:CASH_COUNTER', 'account_type:MONEY_MARKET', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_CO_4567', 'teller_id:EMP_78901', 'customer_name:Amanda Nicole Garcia', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:CO8765432', 'customer_address:456 Mountain View, Denver, CO 80202', 'customer_tax_id:012-34-5678', 'customer_phone:+1-303-555-2345', 'customer_email:amanda.garcia@email.com', 'deposit_amount:18500.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 185, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:3.25', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:2500.00', 'monthly_maintenance_fee:15.00', 'previous_account_balance:12345.67', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T14:04:15Z', 'aml_screening_result:WATCHLIST_HIT', 'aml_risk_score:78', 'sanctions_screening_timestamp:2025-04-18T14:05:23Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR4567_01', 'supervisor_approval_required:True', 'supervisor_id:EMP_23456', 'supervisor_approval_timestamp:2025-04-18T14:15:45Z', 'ctr_filing_required:True', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-8899', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:True', 'cross_sell_products_offered:[]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:AML watchlist hit - false positive confirmed after manual review - compliance override approved', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_012234', 'customer_id:CIF_223344556', 'account_number:4532100022334455', 'transaction_timestamp:2025-04-18T11:20:45Z', 'deposit_type:RETAIL_DEMAND', 'account_type:CHECKING', 'deposit_channel:ONLINE_BANKING', 'branch_code:BR_MI_8901', 'teller_id:', 'customer_name:Christopher Alan White', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:MI3456789', 'customer_address:222 Lake Shore Drive, Detroit, MI 48201', 'customer_tax_id:345-67-8901', 'customer_phone:+1-313-555-6789', 'customer_email:christopher.white@email.com', 'deposit_amount:500.00', 'deposit_currency:USD', 'funding_source_type:ACH', 'funding_source_account:', 'cash_denomination_breakdown:', 'term_length_months:', 'interest_rate_apy:0.05', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:25.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:125.34', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T11:18:56Z', 'aml_screening_result:CLEAR', 'aml_risk_score:5', 'sanctions_screening_timestamp:2025-04-18T11:19:23Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-4455', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:[]', 'customer_signature_captured:False', 'signature_capture_method:NOT_CAPTURED', 'receipt_delivery_method:EMAIL', 'transaction_notes:ACH transfer from external bank account', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_013789', 'customer_id:CIF_556677889', 'account_number:4532100055667788', 'transaction_timestamp:2025-04-18T09:42:18Z', 'deposit_type:CASH_COUNTER', 'account_type:CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_OR_1234', 'teller_id:EMP_45678', 'customer_name:Michelle Lauren Harris', 'customer_id_type:PASSPORT', 'customer_id_number:P98765432', 'customer_address:111 Pioneer Square, Portland, OR 97201', 'customer_tax_id:678-90-1234', 'customer_phone:+1-503-555-8901', 'customer_email:michelle.harris@email.com', 'deposit_amount:4500.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 40, "50": 10, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:0.20', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:8.00', 'previous_account_balance:234.56', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T09:38:34Z', 'aml_screening_result:CLEAR', 'aml_risk_score:25', 'sanctions_screening_timestamp:2025-04-18T09:39:12Z', 'counterfeit_detection_result:FAIL', 'cash_counter_machine_id:CCM_BR1234_01', 'supervisor_approval_required:True', 'supervisor_id:EMP_67890', 'supervisor_approval_timestamp:2025-04-18T09:50:23Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-7788', 'online_banking_enabled:True', 'mobile_app_enabled:False', 'checkbook_ordered:True', 'cross_sell_products_offered:[]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Counterfeit detection failed - two $100 bills rejected after secondary validation - bills returned to customer - transaction processed with verified amount only', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_014567', 'customer_id:CIF_998877665', 'account_number:4532100099887766', 'transaction_timestamp:2025-04-18T15:33:42Z', 'deposit_type:CASH_COUNTER', 'account_type:SAVINGS', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_NV_5678', 'teller_id:EMP_89012', 'customer_name:Daniel Joseph Clark', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:NV6543210', 'customer_address:333 Strip Boulevard, Las Vegas, NV 89101', 'customer_tax_id:789-01-2345', 'customer_phone:+1-702-555-0123', 'customer_email:daniel.clark@email.com', 'deposit_amount:22000.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 220, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:2.90', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:5678.90', 'kyc_verification_status:REQUIRES_REVIEW', 'kyc_verification_timestamp:2025-04-18T15:29:45Z', 'aml_screening_result:MANUAL_REVIEW_REQUIRED', 'aml_risk_score:85', 'sanctions_screening_timestamp:2025-04-18T15:30:23Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR5678_02', 'supervisor_approval_required:True', 'supervisor_id:EMP_12345', 'supervisor_approval_timestamp:', 'ctr_filing_required:True', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-7766', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:[]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:High-risk transaction - KYC requires review - AML manual review required - multiple red flags detected', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_015890', 'customer_id:CIF_334455667', 'account_number:4532100033445566', 'transaction_timestamp:2025-04-18T13:45:28Z', 'deposit_type:TERM_DEPOSIT', 'account_type:CD_1Y', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_GA_2345', 'teller_id:EMP_23456', 'customer_name:Jessica Lynn Robinson', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:GA9876543', 'customer_address:444 Peachtree Street, Atlanta, GA 30303', 'customer_tax_id:456-78-9012', 'customer_phone:+1-404-555-3456', 'customer_email:jessica.robinson@email.com', 'deposit_amount:50000.00', 'deposit_currency:USD', 'funding_source_type:INTERNAL_TRANSFER', 'funding_source_account:4532100077889900', 'cash_denomination_breakdown:', 'term_length_months:12', 'interest_rate_apy:4.50', 'maturity_date:', 'early_withdrawal_penalty_rate:0.5', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:0.00', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T13:41:12Z', 'aml_screening_result:CLEAR', 'aml_risk_score:20', 'sanctions_screening_timestamp:2025-04-18T13:41:56Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:True', 'supervisor_id:EMP_56789', 'supervisor_approval_timestamp:2025-04-18T13:49:34Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:False', 'debit_card_number:', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["INVESTMENT_ACCOUNT", "FINANCIAL_PLANNING"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:High-balance CD - premium rate tier applied', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_016234', 'customer_id:CIF_112233445', 'account_number:4532100011223344', 'transaction_timestamp:2025-04-18T10:18:55Z', 'deposit_type:RETAIL_DEMAND', 'account_type:CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_TN_6789', 'teller_id:EMP_34567', 'customer_name:Brandon Keith Walker', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:TN2345678', 'customer_address:666 Music Row, Nashville, TN 37203', 'customer_tax_id:123-45-6789', 'customer_phone:+1-615-555-7890', 'customer_email:brandon.walker@email.com', 'deposit_amount:2800.00', 'deposit_currency:USD', 'funding_source_type:CHECK', 'funding_source_account:', 'cash_denomination_breakdown:', 'term_length_months:', 'interest_rate_apy:0.15', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:100.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:567.89', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T10:15:23Z', 'aml_screening_result:CLEAR', 'aml_risk_score:10', 'sanctions_screening_timestamp:2025-04-18T10:16:01Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-3344', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:True', 'cross_sell_products_offered:["CREDIT_CARD"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Payroll check deposit', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_017678', 'customer_id:CIF_889900112', 'account_number:4532100088990011', 'transaction_timestamp:2025-04-18T14:52:16Z', 'deposit_type:CASH_COUNTER', 'account_type:MONEY_MARKET', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_VA_3456', 'teller_id:EMP_78901', 'customer_name:Stephanie Rose Miller', 'customer_id_type:PASSPORT', 'customer_id_number:P23456789', 'customer_address:777 Capitol Avenue, Richmond, VA 23219', 'customer_tax_id:234-56-7890', 'customer_phone:+1-804-555-2345', 'customer_email:stephanie.miller@email.com', 'deposit_amount:35000.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 350, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:3.40', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:2500.00', 'monthly_maintenance_fee:15.00', 'previous_account_balance:45678.12', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T14:47:34Z', 'aml_screening_result:CLEAR', 'aml_risk_score:68', 'sanctions_screening_timestamp:2025-04-18T14:48:23Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR3456_01', 'supervisor_approval_required:True', 'supervisor_id:EMP_90123', 'supervisor_approval_timestamp:2025-04-18T14:58:45Z', 'ctr_filing_required:True', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-0011', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:True', 'cross_sell_products_offered:["PRIVATE_BANKING", "INVESTMENT_MANAGEMENT"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:Large cash deposit - business proceeds - CTR filed - elevated AML risk score but cleared after enhanced due diligence', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_018901', 'customer_id:CIF_667788990', 'account_number:4532100066778899', 'transaction_timestamp:2025-04-18T11:07:39Z', 'deposit_type:RETAIL_DEMAND', 'account_type:SAVINGS', 'deposit_channel:ATM', 'branch_code:BR_MN_4567', 'teller_id:', 'customer_name:Kevin Michael Taylor', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:MN8765432', 'customer_address:888 Lakeview Drive, Minneapolis, MN 55401', 'customer_tax_id:567-89-0123', 'customer_phone:+1-612-555-6789', 'customer_email:kevin.taylor@email.com', 'deposit_amount:1500.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 15, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:2.65', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:5.00', 'previous_account_balance:3456.78', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T11:05:12Z', 'aml_screening_result:CLEAR', 'aml_risk_score:9', 'sanctions_screening_timestamp:2025-04-18T11:05:56Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-8899', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:[]', 'customer_signature_captured:False', 'signature_capture_method:NOT_CAPTURED', 'receipt_delivery_method:PRINTED', 'transaction_notes:ATM cash deposit - standard transaction', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_019567', 'customer_id:CIF_445566778', 'account_number:4532100044556677', 'transaction_timestamp:2025-04-18T16:23:47Z', 'deposit_type:TERM_DEPOSIT', 'account_type:CD_1Y', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_OH_5678', 'teller_id:EMP_12345', 'customer_name:Rachel Anne Martinez', 'customer_id_type:DRIVERS_LICENSE', 'customer_id_number:OH5432109', 'customer_address:555 River Road, Cincinnati, OH 45201', 'customer_tax_id:890-12-3456', 'customer_phone:+1-513-555-8901', 'customer_email:rachel.martinez@email.com', 'deposit_amount:15000.00', 'deposit_currency:USD', 'funding_source_type:INTERNAL_TRANSFER', 'funding_source_account:4532100099887766', 'cash_denomination_breakdown:', 'term_length_months:12', 'interest_rate_apy:4.00', 'maturity_date:', 'early_withdrawal_penalty_rate:0.5', 'minimum_balance_requirement:1000.00', 'monthly_maintenance_fee:0.00', 'previous_account_balance:0.00', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T16:20:23Z', 'aml_screening_result:CLEAR', 'aml_risk_score:16', 'sanctions_screening_timestamp:2025-04-18T16:21:01Z', 'counterfeit_detection_result:NOT_APPLICABLE', 'cash_counter_machine_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'supervisor_approval_timestamp:', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:False', 'debit_card_number:', 'online_banking_enabled:True', 'mobile_app_enabled:True', 'checkbook_ordered:False', 'cross_sell_products_offered:["HOME_EQUITY_LOAN"]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:EMAIL', 'transaction_notes:1-year CD for emergency fund - standard rate', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        ['transaction_id:TXN_DEP_20250418_020234', 'customer_id:CIF_223344556', 'account_number:4532100022334455', 'transaction_timestamp:2025-04-18T12:56:32Z', 'deposit_type:CASH_COUNTER', 'account_type:CHECKING', 'deposit_channel:BRANCH_TELLER', 'branch_code:BR_WI_7890', 'teller_id:EMP_67890', 'customer_name:Anthony James Lewis', 'customer_id_type:SSN', 'customer_id_number:678-90-1234', 'customer_address:333 Wisconsin Avenue, Milwaukee, WI 53202', 'customer_tax_id:678-90-1234', 'customer_phone:+1-414-555-0123', 'customer_email:anthony.lewis@email.com', 'deposit_amount:8750.00', 'deposit_currency:USD', 'funding_source_type:CASH', 'funding_source_account:', 'cash_denomination_breakdown:{"100": 87, "50": 1, "20": 0, "10": 0, "5": 0, "1": 0}', 'term_length_months:', 'interest_rate_apy:0.10', 'maturity_date:', 'early_withdrawal_penalty_rate:', 'minimum_balance_requirement:500.00', 'monthly_maintenance_fee:8.00', 'previous_account_balance:1234.56', 'kyc_verification_status:VERIFIED', 'kyc_verification_timestamp:2025-04-18T12:52:45Z', 'aml_screening_result:CLEAR', 'aml_risk_score:38', 'sanctions_screening_timestamp:2025-04-18T12:53:34Z', 'counterfeit_detection_result:PASS', 'cash_counter_machine_id:CCM_BR7890_01', 'supervisor_approval_required:True', 'supervisor_id:EMP_45678', 'supervisor_approval_timestamp:2025-04-18T13:01:12Z', 'ctr_filing_required:False', 'ctr_threshold_amount:10000.00', 'debit_card_issued:True', 'debit_card_number:****-****-****-4455', 'online_banking_enabled:True', 'mobile_app_enabled:False', 'checkbook_ordered:True', 'cross_sell_products_offered:[]', 'customer_signature_captured:True', 'signature_capture_method:ELECTRONIC_PAD', 'receipt_delivery_method:PRINTED', 'transaction_notes:Cash deposit near CTR threshold - monitoring for potential structuring patterns', 'transaction_approved:', 'approval_rejection_reason:', 'new_account_balance:', 'transaction_reference_id:', 'cd_certificate_number:', 'calculated_interest_at_maturity:', 'account_status:', 'processing_time_seconds:', 'cash_drawer_total_after_transaction:', 'fraud_detection_score:', 'compliance_check_passed:', 'final_transaction_fee:', 'receipt_number:'],
        </data>'''
    yaml_file = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_API_Generation.yaml"
    system_prompt = "You are an SOP-related API documentation generator. Every input parameter of the APIs you generate must be either already present in the sample_data or obtainable through calculations from other API functions. You must implement the APIs in the API dependency graph using their original names."
    api_graph = '''
        <api_dependency_graph>
        apis:
        - api_name: "KYC_Verification_API"
            description: "Executes comprehensive Know Your Customer identity verification workflows including multi-source identity validation, confidence scoring algorithms, third-party data aggregator queries, and documentation completeness assessment to satisfy Customer Identification Program regulatory requirements under 31 CFR 1020.220"
            purpose: "Validate customer identity credentials, verify correlation between presented identification documents and customer master data, calculate identity confidence scores, and determine kyc_verification_status to gate subsequent compliance processing workflows"
            input_parameters:
            - name: "customer_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_name"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_id_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "customer_id_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_tax_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_phone"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_email"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "kyc_verification_status"
                data_type: "enum"
                destination: "other_api"
            - name: "kyc_verification_timestamp"
                data_type: "datetime"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "AML_Sanctions_Screening_API"
            description: "Performs automated Anti-Money Laundering screening through fuzzy matching algorithms against consolidated sanctions databases including OFAC SDN list, UN Security Council sanctions, and proprietary watchlists, calculates composite AML risk scores using weighted multi-dimensional risk modeling, and determines screening result classification requiring manual review escalation when risk thresholds are exceeded"
            purpose: "Screen customer identifying information against governmental and institutional watchlists, calculate probabilistic AML risk scores based on transaction characteristics and customer behavioral patterns, and output aml_screening_result and aml_risk_score to inform approval routing decisions"
            input_parameters:
            - name: "customer_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_name"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_address"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "customer_tax_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "kyc_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "deposit_channel"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "funding_source_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "aml_screening_result"
                data_type: "enum"
                destination: "other_api"
            - name: "aml_risk_score"
                data_type: "integer"
                destination: "other_api"
            - name: "sanctions_screening_timestamp"
                data_type: "datetime"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "KYC_Verification_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires kyc_verification_status to equal VERIFIED before executing AML screening per SOP Section 5.2 statement 'Following successful completion of Know Your Customer verification procedures as evidenced by kyc_verification_status field attaining VERIFIED enumeration'"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "CTR_Evaluation_API"
            description: "Executes real-time Currency Transaction Report filing obligation evaluation by comparing deposit amounts against statutory thresholds codified in 31 USC 5313, performs aggregation queries to detect potential structuring patterns across multiple transactions within rolling twelve-month windows, and determines ctr_filing_required boolean flag triggering FinCEN Form 112 generation workflows"
            purpose: "Assess CTR filing requirements based on deposit amount thresholds and funding source type, detect structuring activity patterns through transaction aggregation analysis, and output ctr_filing_required flag to inform compliance verification and regulatory reporting obligations"
            input_parameters:
            - name: "customer_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "funding_source_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "ctr_threshold_amount"
                data_type: "float"
                source: "system_state"
                required: true
            output_parameters:
            - name: "ctr_filing_required"
                data_type: "boolean"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "low"
            business_criticality: "high"

        - api_name: "Currency_Validation_API"
            description: "Interfaces with certified currency discriminator hardware to execute multi-spectral imaging analysis including ultraviolet fluorescence detection, infrared absorption spectroscopy, magnetic sensing, and dimensional measurement for physical currency authentication, performs denomination count reconciliation against teller-declared cash_denomination_breakdown, and calculates updated cash_drawer_total_after_transaction for teller balancing workflows"
            purpose: "Validate authenticity of physical currency through counterfeit detection mechanisms, reconcile machine-counted denominations against declared amounts, update teller cash drawer totals, and output counterfeit_detection_result and cash_drawer_total_after_transaction for compliance verification and operational reconciliation"
            input_parameters:
            - name: "funding_source_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "cash_denomination_breakdown"
                data_type: "JSON"
                source: "user_input"
                required: false
            - name: "cash_counter_machine_id"
                data_type: "string"
                source: "system_state"
                required: false
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "counterfeit_detection_result"
                data_type: "enum"
                destination: "other_api"
            - name: "cash_drawer_total_after_transaction"
                data_type: "float"
                destination: "storage"
            direct_dependencies: []
            complexity_level: "low"
            business_criticality: "medium"

        - api_name: "Fraud_Detection_Scoring_API"
            description: "Executes real-time behavioral analytics fraud detection through machine learning models trained on historical fraud incident patterns, retrieves customer transaction profile baselines from CRM data warehouse, calculates z-score deviations for deposit amounts and behavioral characteristics, evaluates geographic impossibility scenarios and channel switching patterns, and generates probabilistic fraud_detection_score through weighted logistic regression aggregation of individual risk signals"
            purpose: "Calculate real-time fraud risk probability scores based on transaction behavioral deviations from customer baseline patterns, detect account takeover indicators and geographic anomalies, and output fraud_detection_score to inform approval routing and enhanced authentication requirement determination"
            input_parameters:
            - name: "customer_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "account_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "deposit_channel"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "deposit_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "fraud_detection_score"
                data_type: "float"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "Supervisor_Approval_Routing_API"
            description: "Evaluates multiple conditional criteria against employee authorization matrices to determine supervisor_approval_required flag status, assesses whether deposit amounts exceed teller authority limits, aggregates risk indicators from KYC, AML, CTR, counterfeit detection, and fraud scoring workflows, implements tiered escalation logic routing transactions to Branch Manager or Regional Manager approval queues with comprehensive risk context presentation, and captures supervisor_id and supervisor_approval_timestamp upon approval execution"
            purpose: "Determine supervisory approval necessity based on transaction amount thresholds and aggregated risk assessment outcomes, route high-risk or high-value transactions to appropriate authorization levels, capture approval chain documentation, and output supervisor_approval_required flag with corresponding approval metadata for compliance verification"
            input_parameters:
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "teller_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "kyc_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "aml_screening_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "aml_risk_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "fraud_detection_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "counterfeit_detection_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "ctr_filing_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "transaction_notes"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "supervisor_approval_required"
                data_type: "boolean"
                destination: "other_api"
            - name: "supervisor_id"
                data_type: "string"
                destination: "other_api"
            - name: "supervisor_approval_timestamp"
                data_type: "datetime"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "KYC_Verification_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires kyc_verification_status as input to evaluate whether KYC status equals REQUIRES_REVIEW triggering supervisor approval per SOP Section 5.6 conditional criteria"
            - dependent_api: "AML_Sanctions_Screening_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires aml_screening_result and aml_risk_score to evaluate whether AML results equal WATCHLIST_HIT or MANUAL_REVIEW_REQUIRED or risk score exceeds 75 threshold per SOP Section 5.6"
            - dependent_api: "Fraud_Detection_Scoring_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires fraud_detection_score to evaluate whether score exceeds 0.70 threshold triggering mandatory supervisor approval per SOP Section 5.6"
            - dependent_api: "Currency_Validation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires counterfeit_detection_result to evaluate whether result equals FAIL necessitating Branch Manager intervention per SOP Section 5.6"
            - dependent_api: "CTR_Evaluation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires ctr_filing_required flag to determine if CTR filing combined with suspicious transaction notes triggers supervisor notification per SOP Section 5.6"
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "Term_Deposit_Processing_API"
            description: "Executes specialized term deposit product processing logic including retrieval of treasury-approved interest_rate_apy from rate matrix tables indexed by term_length_months and deposit_amount tier thresholds, calculates maturity_date through calendar month addition with business day adjustment logic, computes calculated_interest_at_maturity using compound interest formula with configurable compounding frequency, generates unique cd_certificate_number identifiers, and establishes automated maturity notification calendar triggers"
            purpose: "Process term deposit specific calculations and certificate generation for TERM_DEPOSIT transaction types, retrieve applicable interest rates from treasury rate tables, calculate maturity parameters and projected interest earnings, and output cd_certificate_number, maturity_date, and calculated_interest_at_maturity for customer disclosure and ledger posting"
            input_parameters:
            - name: "deposit_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "term_length_months"
                data_type: "integer"
                source: "user_input"
                required: false
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "interest_rate_apy"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "early_withdrawal_penalty_rate"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "cd_certificate_number"
                data_type: "string"
                destination: "other_api"
            - name: "maturity_date"
                data_type: "datetime"
                destination: "other_api"
            - name: "calculated_interest_at_maturity"
                data_type: "float"
                destination: "other_api"
            direct_dependencies: []
            complexity_level: "medium"
            business_criticality: "medium"

        - api_name: "Compliance_Aggregation_API"
            description: "Performs comprehensive compliance verification aggregation by implementing logical AND operation across all mandatory control checkpoint boolean states, evaluates whether kyc_verification_status equals VERIFIED, aml_screening_result equals CLEAR or has documented override approval, counterfeit_detection_result passes validation, fraud_detection_score falls below institutional tolerance thresholds, and supervisor approvals are satisfied when required, determines final compliance_check_passed boolean and transaction_approved status, and populates approval_rejection_reason field with structured rejection rationale when any control checkpoint fails"
            purpose: "Aggregate all compliance validation results through boolean logic evaluation, determine final transaction approval status based on comprehensive control checkpoint assessment, generate approval or rejection determination with detailed rationale documentation, and output compliance_check_passed, transaction_approved, and approval_rejection_reason for ledger posting gate control"
            input_parameters:
            - name: "kyc_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "aml_screening_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "aml_risk_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "counterfeit_detection_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "fraud_detection_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "supervisor_approval_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "supervisor_id"
                data_type: "string"
                source: "other_api"
                required: false
            - name: "supervisor_approval_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: false
            - name: "ctr_filing_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            output_parameters:
            - name: "compliance_check_passed"
                data_type: "boolean"
                destination: "other_api"
            - name: "transaction_approved"
                data_type: "boolean"
                destination: "other_api"
            - name: "approval_rejection_reason"
                data_type: "string"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "KYC_Verification_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires kyc_verification_status to evaluate whether status equals VERIFIED as mandatory condition for compliance_check_passed per SOP Section 5.9 logical AND operation"
            - dependent_api: "AML_Sanctions_Screening_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires aml_screening_result and aml_risk_score to evaluate whether screening equals CLEAR or has override approval as compliance condition per SOP Section 5.9"
            - dependent_api: "Currency_Validation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires counterfeit_detection_result to evaluate whether result equals PASS or NOT_APPLICABLE for compliance verification per SOP Section 5.9"
            - dependent_api: "Fraud_Detection_Scoring_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires fraud_detection_score to evaluate whether score below 0.70 threshold or has analyst override approval per SOP Section 5.9"
            - dependent_api: "Supervisor_Approval_Routing_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires supervisor_approval_required flag and corresponding approval metadata to evaluate whether approvals satisfied when required per SOP Section 5.9"
            - dependent_api: "CTR_Evaluation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires ctr_filing_required to evaluate regulatory filing completeness as part of compliance aggregation per SOP Section 5.9"
            complexity_level: "medium"
            business_criticality: "high"

        - api_name: "Account_Ledger_Posting_API"
            description: "Executes atomic transaction commit operation implementing double-entry accounting principles by simultaneously debiting appropriate asset accounts based on funding_source_type and crediting customer deposit account general ledger, calculates new_account_balance through decimal precision arithmetic addition of deposit_amount to previous_account_balance, generates globally unique transaction_reference_id, updates account_status to ACTIVE, triggers ancillary provisioning workflows for debit card issuance and online banking enablement, records processing_time_seconds performance metric, and ensures general ledger coding for regulatory reserve requirement reporting"
            purpose: "Execute final ledger posting with double-entry accounting integrity, update customer account balance with transaction amount, generate immutable transaction reference identifier, activate account status and ancillary service provisioning, and output new_account_balance, transaction_reference_id, account_status, and processing_time_seconds as definitive transaction completion evidence"
            input_parameters:
            - name: "account_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "previous_account_balance"
                data_type: "float"
                source: "system_state"
                required: true
            - name: "funding_source_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "funding_source_account"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "account_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "compliance_check_passed"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "transaction_approved"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "cd_certificate_number"
                data_type: "string"
                source: "other_api"
                required: false
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "debit_card_issued"
                data_type: "boolean"
                source: "user_input"
                required: false
            - name: "online_banking_enabled"
                data_type: "boolean"
                source: "user_input"
                required: false
            - name: "mobile_app_enabled"
                data_type: "boolean"
                source: "user_input"
                required: false
            - name: "checkbook_ordered"
                data_type: "boolean"
                source: "user_input"
                required: false
            output_parameters:
            - name: "new_account_balance"
                data_type: "float"
                destination: "other_api"
            - name: "transaction_reference_id"
                data_type: "string"
                destination: "other_api"
            - name: "account_status"
                data_type: "enum"
                destination: "other_api"
            - name: "processing_time_seconds"
                data_type: "integer"
                destination: "other_api"
            - name: "debit_card_number"
                data_type: "string"
                destination: "storage"
            - name: "final_transaction_fee"
                data_type: "float"
                destination: "other_api"
            direct_dependencies:
            - dependent_api: "Compliance_Aggregation_API"
                dependency_type: "state_dependency"
                dependency_reason: "Requires compliance_check_passed to equal true and transaction_approved to equal true before executing ledger posting per SOP Section 5.8 statement 'Upon successful completion of all validation, screening, approval, and product-specific processing workflows'"
            - dependent_api: "Term_Deposit_Processing_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires cd_certificate_number for term deposit transactions to link certificate instrument to ledger account per SOP Section 5.8 term deposit processing requirements"
            complexity_level: "high"
            business_criticality: "high"

        - api_name: "Receipt_Generation_API"
            description: "Generates comprehensive transaction documentation for customer delivery including receipt_number formatted with branch and date identifiers, retrieves and formats transaction details with currency symbols and comma separators, incorporates Truth in Savings Act mandatory disclosures for interest-bearing accounts including APY and early withdrawal penalties, executes receipt delivery via specified receipt_delivery_method including printed thermal output, email PDF transmission, SMS text notification, or mobile app push notification, updates customer profile in CRM with relationship value recalculation and next-best-action recommendations"
            purpose: "Generate formatted transaction receipts with regulatory disclosure compliance, deliver documentation through customer-selected channel, update CRM relationship data, and output receipt_number as customer reference identifier for inquiry and dispute resolution procedures"
            input_parameters:
            - name: "customer_name"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "account_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "new_account_balance"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "transaction_reference_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "receipt_delivery_method"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "customer_email"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "customer_phone"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "cd_certificate_number"
                data_type: "string"
                source: "other_api"
                required: false
            - name: "maturity_date"
                data_type: "datetime"
                source: "other_api"
                required: false
            - name: "interest_rate_apy"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "calculated_interest_at_maturity"
                data_type: "float"
                source: "other_api"
                required: false
            - name: "transaction_notes"
                data_type: "string"
                source: "user_input"
                required: false
            - name: "transaction_approved"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "minimum_balance_requirement"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "monthly_maintenance_fee"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "early_withdrawal_penalty_rate"
                data_type: "float"
                source: "system_state"
                required: false
            - name: "final_transaction_fee"
                data_type: "float"
                source: "other_api"
                required: true
            output_parameters:
            - name: "receipt_number"
                data_type: "string"
                destination: "ui_display"
            direct_dependencies:
            - dependent_api: "Account_Ledger_Posting_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires new_account_balance and transaction_reference_id from ledger posting to populate receipt with updated balance and reference number per SOP Section 5.10 receipt generation requirements"
            - dependent_api: "Term_Deposit_Processing_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires cd_certificate_number, maturity_date, and calculated_interest_at_maturity for term deposit receipt documentation per SOP Section 5.10 Truth in Savings Act disclosure requirements"
            - dependent_api: "Compliance_Aggregation_API"
                dependency_type: "state_dependency"
                dependency_reason: "Requires transaction_approved boolean to confirm successful approval before generating customer receipt per SOP Section 5.10 statement 'Upon successful transaction approval'"
            complexity_level: "low"
            business_criticality: "medium"

        - api_name: "Audit_Trail_Recording_API"
            description: "Creates comprehensive immutable audit trail documentation capturing complete transaction lifecycle from initiation through final approval and posting, incorporates all schema fields from Existing_fields and Field_to_be_obtained specifications along with supplementary metadata including IP addresses and device fingerprints, preserves pre-transaction state snapshots and validation check result details, implements blockchain-inspired cryptographic hashing for data integrity verification, establishes multi-dimensional indexing for efficient retrieval, and executes automated data lifecycle management transitioning records through cost-optimized storage tiers"
            purpose: "Generate complete audit trail records with cryptographic integrity controls, preserve transaction lifecycle documentation for regulatory examination and incident investigation, implement efficient retrieval indexing across multiple query dimensions, and establish compliance with Bank Secrecy Act five-year retention requirements for deposit account transaction records"
            input_parameters:
            - name: "transaction_id"
                data_type: "string"
                source: "system_state"
                required: true
            - name: "customer_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "account_number"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "transaction_timestamp"
                data_type: "datetime"
                source: "user_input"
                required: true
            - name: "deposit_type"
                data_type: "enum"
                source: "user_input"
                required: true
            - name: "deposit_amount"
                data_type: "float"
                source: "user_input"
                required: true
            - name: "teller_id"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "branch_code"
                data_type: "string"
                source: "user_input"
                required: true
            - name: "kyc_verification_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "kyc_verification_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: true
            - name: "aml_screening_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "aml_risk_score"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "sanctions_screening_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: true
            - name: "fraud_detection_score"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "counterfeit_detection_result"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "supervisor_approval_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "supervisor_id"
                data_type: "string"
                source: "other_api"
                required: false
            - name: "supervisor_approval_timestamp"
                data_type: "datetime"
                source: "other_api"
                required: false
            - name: "compliance_check_passed"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "transaction_approved"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "approval_rejection_reason"
                data_type: "string"
                source: "other_api"
                required: false
            - name: "new_account_balance"
                data_type: "float"
                source: "other_api"
                required: true
            - name: "transaction_reference_id"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "account_status"
                data_type: "enum"
                source: "other_api"
                required: true
            - name: "processing_time_seconds"
                data_type: "integer"
                source: "other_api"
                required: true
            - name: "receipt_number"
                data_type: "string"
                source: "other_api"
                required: true
            - name: "previous_account_balance"
                data_type: "float"
                source: "system_state"
                required: true
            - name: "ctr_filing_required"
                data_type: "boolean"
                source: "other_api"
                required: true
            - name: "cash_drawer_total_after_transaction"
                data_type: "float"
                source: "other_api"
                required: false
            output_parameters:
            - name: "audit_trail_hash"
                data_type: "string"
                destination: "storage"
            - name: "audit_record_id"
                data_type: "string"
                destination: "storage"
            direct_dependencies:
            - dependent_api: "KYC_Verification_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires kyc_verification_status and kyc_verification_timestamp for complete audit trail documentation per SOP Section 5.11 comprehensive lifecycle capture requirements"
            - dependent_api: "AML_Sanctions_Screening_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires aml_screening_result, aml_risk_score, and sanctions_screening_timestamp for audit trail validation check results per SOP Section 5.11"
            - dependent_api: "Fraud_Detection_Scoring_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires fraud_detection_score and dimensional score components for audit trail model performance analysis per SOP Section 5.11"
            - dependent_api: "Currency_Validation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires counterfeit_detection_result and cash_drawer_total_after_transaction for cash handling audit documentation per SOP Section 5.11"
            - dependent_api: "Supervisor_Approval_Routing_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires supervisor_approval_required, supervisor_id, and supervisor_approval_timestamp for authorization hierarchy audit chain per SOP Section 5.11"
            - dependent_api: "Compliance_Aggregation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires compliance_check_passed, transaction_approved, and approval_rejection_reason for final approval determination audit documentation per SOP Section 5.11"
            - dependent_api: "Account_Ledger_Posting_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires new_account_balance, transaction_reference_id, account_status, and processing_time_seconds for ledger posting results audit capture per SOP Section 5.11"
            - dependent_api: "Receipt_Generation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires receipt_number for complete transaction documentation audit trail per SOP Section 5.11 customer interaction record requirements"
            - dependent_api: "CTR_Evaluation_API"
                dependency_type: "data_dependency"
                dependency_reason: "Requires ctr_filing_required for regulatory filing obligation audit documentation per SOP Section 5.11 compliance record retention"
            complexity_level: "low"
            business_criticality: "high"
        </api_dependency_graph>
        '''

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_file_contents, additional_context, sample_data, api_graph)

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
    with open("api_response_output.txt", "w", encoding="utf-8") as file:
        file.write(full_response)

    print("完整回复已保存到 api_response_output.txt 文件")

if __name__ == "__main__":
    main()
