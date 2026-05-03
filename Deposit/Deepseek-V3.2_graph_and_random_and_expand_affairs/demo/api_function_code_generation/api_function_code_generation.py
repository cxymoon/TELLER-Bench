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

        ## 1. SOP Sections Relevant to These Tools

        The API tools correspond to the following SOP sections:

        - **Section 5.1 (KYC_Verification_API)**: Customer Identification and Know Your Customer Verification Execution
        - **Section 5.2 (AML_Sanctions_Screening_API)**: Anti-Money Laundering and Sanctions Screening Protocol Implementation
        - **Section 5.3 (CTR_Evaluation_API)**: Currency Transaction Report Threshold Evaluation and Filing Determination
        - **Section 5.4 (Currency_Validation_API)**: Physical Currency Validation and Counterfeit Detection Procedure
        - **Section 5.5 (Fraud_Detection_Scoring_API)**: Real-Time Fraud Detection Scoring and Behavioral Analytics Evaluation
        - **Section 5.6 (Supervisor_Approval_Routing_API)**: Supervisory Approval Routing and Authorization Limit Validation
        - **Section 5.7 (Term_Deposit_Processing_API)**: Term Deposit Specific Processing and Interest Calculation Execution
        - **Section 5.8 (Account_Ledger_Posting_API)**: Account Balance Update and General Ledger Posting Execution
        - **Section 5.9 (Compliance_Aggregation_API)**: Compliance Verification Aggregation and Final Approval Determination
        - **Section 5.10 (Receipt_Generation_API)**: Receipt Generation and Transaction Documentation Finalization
        - **Section 5.11 (Audit_Trail_Recording_API)**: Audit Trail Generation and Regulatory Record Retention Establishment

        ## 2. Input Parameters Based on SOP and Metadata

        Each API has specific input requirements:

        - **KYC_Verification_API**: Requires customer identification data including transaction_id, customer_id, customer_name, ID type/number, address, tax ID, phone, and email
        - **AML_Sanctions_Screening_API**: Requires transaction and customer data plus KYC verification status, deposit details, and funding source information
        - **CTR_Evaluation_API**: Requires transaction amount, funding source type, timestamp, and threshold configuration
        - **Currency_Validation_API**: Requires cash denomination breakdown and machine ID (conditional on funding_source_type=CASH)
        - **Fraud_Detection_Scoring_API**: Requires transaction behavioral attributes including amount, channel, type, location, and timing
        - **Supervisor_Approval_Routing_API**: Aggregates results from multiple prior APIs including KYC, AML, fraud scores, and counterfeit detection
        - **Term_Deposit_Processing_API**: Requires term length, deposit amount, APY rate, and penalty rates (conditional on deposit_type=TERM_DEPOSIT)
        - **Compliance_Aggregation_API**: Consolidates all compliance checkpoint results from preceding validations
        - **Account_Ledger_Posting_API**: Requires approved transaction data, account details, funding source, and compliance clearance
        - **Receipt_Generation_API**: Requires finalized transaction details including reference ID, balances, and delivery preferences
        - **Audit_Trail_Recording_API**: Captures comprehensive transaction lifecycle data from all preceding APIs

        ## 3. Constraints and Special Considerations

        Key constraints from the SOP:

        - **Sequential Dependencies**: Many APIs depend on outputs from prior APIs (e.g., AML screening requires KYC verification completion)
        - **Conditional Execution**: Some APIs only execute under specific conditions (Currency_Validation_API for cash, Term_Deposit_Processing_API for term deposits)
        - **Threshold-Based Decision Logic**: Multiple APIs implement threshold-based branching (fraud scores >0.70, AML scores >75, CTR amounts ≥$10,000)
        - **Regulatory Compliance Requirements**: All APIs must maintain audit trails, satisfy Bank Secrecy Act requirements, and support regulatory examination
        - **Error Handling Requirements**: Each API must implement specific error scenarios with appropriate HTTP status codes and retry logic
        - **Data Precision Requirements**: Financial calculations must use decimal precision to prevent rounding errors
        - **Timeout and Performance SLAs**: Processing time must be monitored, with typical transactions completing in 60-300 seconds

        ## 4. Alignment with Overall SOP Direction

        The API tools implement the SOP's comprehensive deposit handling framework by:

        - **Modularizing Compliance Controls**: Each API represents a discrete compliance or operational checkpoint
        - **Enabling Sequential Validation**: APIs execute in logical sequence matching SOP procedural steps
        - **Supporting Risk-Based Decision Making**: APIs calculate risk scores and route transactions based on thresholds
        - **Maintaining Audit Trails**: Final Audit_Trail_Recording_API captures complete lifecycle documentation
        - **Ensuring Regulatory Compliance**: APIs collectively implement KYC, AML, CTR, and fraud detection requirements
        - **Facilitating Supervisor Oversight**: Approval routing enables human intervention when risk thresholds are exceeded

        ## 5. Potential Edge Cases and Limitations

        Key edge cases identified:

        - **System Availability**: Multiple APIs depend on external systems (watchlist databases, CRM, currency discriminators) that may be unavailable
        - **Concurrent Transaction Conflicts**: Race conditions during balance updates require optimistic locking and retry logic
        - **Threshold Boundary Conditions**: Transactions exactly at decision thresholds (e.g., exactly $10,000 for CTR) require clear business rules
        - **Missing Historical Data**: New customers lack baseline patterns for fraud detection requiring fallback scoring methodologies
        - **Multi-Day Transaction Aggregation**: CTR structuring detection requires accurate transaction aggregation which may fail if historical data is incomplete
        - **Approval Timeout Scenarios**: Supervisor approval delays require escalation and queue management
        - **Data Format Variations**: ID document formats vary by jurisdiction requiring flexible validation logic
        - **Fuzzy Matching False Positives**: AML screening may generate false positive watchlist matches requiring manual review capacity

        ## 6. Tool Workflow Integration

        The tools form an integrated workflow:

        1. **KYC_Verification_API** initiates the compliance validation chain by verifying customer identity
        2. **AML_Sanctions_Screening_API** executes watchlist screening and risk scoring (depends on KYC completion)
        3. **CTR_Evaluation_API** runs in parallel to determine reporting obligations
        4. **Currency_Validation_API** validates physical cash (conditional execution)
        5. **Fraud_Detection_Scoring_API** calculates behavioral risk scores in parallel
        6. **Supervisor_Approval_Routing_API** aggregates risk indicators and determines approval requirements
        7. **Term_Deposit_Processing_API** calculates term-specific parameters (conditional execution)
        8. **Compliance_Aggregation_API** consolidates all validation results to determine final approval
        9. **Account_Ledger_Posting_API** executes financial posting only after compliance clearance
        10. **Receipt_Generation_API** produces customer documentation after successful posting
        11. **Audit_Trail_Recording_API** captures comprehensive lifecycle documentation for regulatory retention

        This sequential yet partially parallel architecture optimizes processing efficiency while maintaining strict compliance controls at each stage.

        </sop_tool_analysis>

        <SCHEMA>
        {
        "toolSpec": {
            "name": "KYC_Verification_API",
            "description": "Executes comprehensive Know Your Customer identity verification workflows including multi-source identity validation, confidence scoring algorithms, third-party data aggregator queries, and documentation completeness assessment to satisfy Customer Identification Program regulatory requirements under 31 CFR 1020.220. Validates customer identity credentials, verifies correlation between presented identification documents and customer master data, calculates identity confidence scores, and determines kyc_verification_status to gate subsequent compliance processing workflows.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, formatted as TXN_DEP_YYYYMMDD_nnnnnn, serving as the primary key for transaction tracking across all processing stages and audit trail linkage."
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer Information File identifier formatted as CIF_nnnnnnnnn, referencing the authoritative master data repository containing all customer identifying characteristics, demographic attributes, and relationship hierarchies within the Core Banking System."
                },
                "customer_name": {
                    "type": "string",
                    "description": "Full legal name of the customer as it appears on official identification documents, used for exact match and fuzzy matching validation against third-party data aggregators and identity verification services."
                },
                "customer_id_type": {
                    "type": "string",
                    "description": "Enumerated type of identification document presented by the customer for verification purposes. Valid values: PASSPORT (conforming to ICAO machine-readable zone specifications), DRIVERS_LICENSE (including state issuing authority), NATIONAL_ID (jurisdiction-specific formats), SSN (Social Security Number in 9-digit format with hyphens).",
                    "enum": ["PASSPORT", "DRIVERS_LICENSE", "NATIONAL_ID", "SSN"]
                },
                "customer_id_number": {
                    "type": "string",
                    "description": "Unique identifier number from the presented identification document, subject to format validation based on customer_id_type (e.g., PASSPORT requires alphanumeric format, DRIVERS_LICENSE requires state-specific format, SSN requires nnn-nn-nnnn format)."
                },
                "customer_address": {
                    "type": "string",
                    "description": "Complete residential or business address including street number, street name, city, state/province, and postal code, used for address concordance validation and geographic risk assessment in compliance screening workflows."
                },
                "customer_tax_id": {
                    "type": "string",
                    "description": "Tax identification number (TIN) such as Social Security Number or Employer Identification Number in format nnn-nn-nnnn or nn-nnnnnnn respectively, validated through checksum algorithms and cross-referenced with IRS databases where permissible."
                },
                "customer_phone": {
                    "type": "string",
                    "description": "Primary contact telephone number in international format +n-nnn-nnn-nnnn, used for customer callback verification during fraud investigation and multi-factor authentication workflows requiring out-of-band communication."
                },
                "customer_email": {
                    "type": "string",
                    "description": "Primary email address for customer communication and electronic receipt delivery, subject to format validation (RFC 5322 compliance) and domain verification to detect disposable email services indicative of fraud risk."
                }
                },
                "required": [
                "transaction_id",
                "customer_id",
                "customer_name",
                "customer_id_type",
                "customer_id_number",
                "customer_address",
                "customer_tax_id",
                "customer_phone",
                "customer_email"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "AML_Sanctions_Screening_API",
            "description": "Performs automated Anti-Money Laundering screening through fuzzy matching algorithms against consolidated sanctions databases including OFAC SDN list, UN Security Council sanctions, and proprietary watchlists. Calculates composite AML risk scores using weighted multi-dimensional risk modeling and determines screening result classification requiring manual review escalation when risk thresholds are exceeded. Screens customer identifying information against governmental and institutional watchlists, calculates probabilistic AML risk scores based on transaction characteristics and customer behavioral patterns.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, serving as the correlation key across all compliance screening workflows and enabling transaction-level audit trail reconstruction during regulatory examinations."
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer Information File identifier used to retrieve historical transaction patterns, previous AML screening results, and relationship context necessary for behavioral risk scoring algorithms and deviation analysis."
                },
                "customer_name": {
                    "type": "string",
                    "description": "Full legal name of the customer submitted to fuzzy matching algorithms employing Levenshtein distance calculations (maximum edit distance of 2 characters) for comparison against OFAC SDN list, UN sanctions, and proprietary watchlists."
                },
                "customer_address": {
                    "type": "string",
                    "description": "Complete address used for geographic risk assessment based on high-risk jurisdiction classifications (FATF blacklist countries, sanctioned regions) and cross-border transaction pattern analysis detecting structuring or layering typologies."
                },
                "customer_tax_id": {
                    "type": "string",
                    "description": "Tax identification number cross-referenced against watchlist databases and used in conjunction with name and address for multi-attribute matching to reduce false positive rates in sanctions screening algorithms."
                },
                "kyc_verification_status": {
                    "type": "string",
                    "description": "Current status of Know Your Customer verification process, serving as a mandatory prerequisite gate for AML screening execution. Must equal 'VERIFIED' to proceed with AML processing. Valid values: VERIFIED, REQUIRES_REVIEW, PENDING.",
                    "enum": ["VERIFIED", "REQUIRES_REVIEW", "PENDING"]
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the deposit amount in the specified currency, used to calculate z-score deviation from customer historical average deposit size as a risk factor in the composite AML risk scoring model."
                },
                "deposit_channel": {
                    "type": "string",
                    "description": "Channel through which the deposit transaction is being processed, evaluated for consistency with customer established behavioral patterns. Unexpected channel switching (e.g., from branch to mobile app) elevates risk scores. Valid values: BRANCH_TELLER, ATM, MOBILE_APP, ONLINE_BANKING.",
                    "enum": ["BRANCH_TELLER", "ATM", "MOBILE_APP", "ONLINE_BANKING"]
                },
                "branch_code": {
                    "type": "string",
                    "description": "Unique identifier for the branch location processing the transaction, formatted as BR_STATE_nnnn, used to assess geographic risk ratings and detect geographic impossibility scenarios (sequential transactions at distant locations exceeding feasible travel velocity)."
                },
                "funding_source_type": {
                    "type": "string",
                    "description": "Type of funding mechanism for the deposit, with CASH funding elevating AML risk scores relative to electronic transfer mechanisms due to reduced audit trail transparency. Valid values: CASH, CHECK, INTERNAL_TRANSFER, WIRE, ACH.",
                    "enum": ["CASH", "CHECK", "INTERNAL_TRANSFER", "WIRE", "ACH"]
                }
                },
                "required": [
                "transaction_id",
                "customer_id",
                "customer_name",
                "customer_address",
                "customer_tax_id",
                "kyc_verification_status",
                "deposit_amount",
                "deposit_channel",
                "branch_code",
                "funding_source_type"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CTR_Evaluation_API",
            "description": "Executes real-time Currency Transaction Report filing obligation evaluation by comparing deposit amounts against statutory thresholds codified in 31 USC 5313. Performs aggregation queries to detect potential structuring patterns across multiple transactions within rolling twelve-month windows and determines ctr_filing_required boolean flag triggering FinCEN Form 112 generation workflows. Assesses CTR filing requirements based on deposit amount thresholds and funding source type, detects structuring activity patterns through transaction aggregation analysis.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to correlate CTR filing determinations with the specific transaction record and enable subsequent CTR form population with transaction-specific details."
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer Information File identifier used to query historical transaction database for aggregation analysis across rolling twelve-month period, enabling detection of structuring patterns where multiple transactions individually below threshold collectively suggest reporting obligation avoidance."
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the current deposit transaction amount, compared against ctr_threshold_amount to determine immediate CTR filing requirement and aggregated with historical transactions to detect structuring patterns (e.g., frequent deposits in 9000-9999 range)."
                },
                "funding_source_type": {
                    "type": "string",
                    "description": "Type of funding mechanism for the deposit, with CTR filing obligations primarily applicable to CASH transactions per 31 USC 5313, while CHECK, WIRE, ACH, and INTERNAL_TRANSFER create inherent audit trails reducing BSA reporting requirements. Valid values: CASH, CHECK, INTERNAL_TRANSFER, WIRE, ACH.",
                    "enum": ["CASH", "CHECK", "INTERNAL_TRANSFER", "WIRE", "ACH"]
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking transaction initiation, used for temporal aggregation analysis to detect same-business-day multiple transactions requiring aggregation per 31 CFR 1010.311 and clustering patterns indicative of structuring."
                },
                "ctr_threshold_amount": {
                    "type": "number",
                    "description": "Statutory threshold amount above which Currency Transaction Reports must be filed, codified as $10,000.00 USD in 31 USC 5313, configurable to accommodate potential regulatory amendments or foreign currency equivalent calculations based on real-time exchange rates."
                }
                },
                "required": [
                "transaction_id",
                "customer_id",
                "deposit_amount",
                "funding_source_type",
                "transaction_timestamp",
                "ctr_threshold_amount"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Currency_Validation_API",
            "description": "Interfaces with certified currency discriminator hardware to execute multi-spectral imaging analysis including ultraviolet fluorescence detection, infrared absorption spectroscopy, magnetic sensing, and dimensional measurement for physical currency authentication. Performs denomination count reconciliation against teller-declared cash_denomination_breakdown and calculates updated cash_drawer_total_after_transaction for teller balancing workflows. Validates authenticity of physical currency through counterfeit detection mechanisms, reconciles machine-counted denominations against declared amounts, and updates teller cash drawer totals.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, linking currency validation results to the specific transaction record and enabling supervisor investigation of counterfeit detection incidents or count discrepancies."
                },
                "funding_source_type": {
                    "type": "string",
                    "description": "Type of funding mechanism for the deposit, with Currency_Validation_API conditionally executing only when funding_source_type equals CASH, as non-cash funding sources do not require physical currency authentication. Valid values: CASH, CHECK, INTERNAL_TRANSFER, WIRE, ACH.",
                    "enum": ["CASH", "CHECK", "INTERNAL_TRANSFER", "WIRE", "ACH"]
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Total numeric value of the cash deposit being validated, used to verify that the sum of denomination counts in cash_denomination_breakdown matches the declared deposit_amount within zero discrepancy tolerance, with any variance requiring recount procedures."
                },
                "cash_denomination_breakdown": {
                    "type": "object",
                    "description": "JSON object structure detailing the count of bills per denomination, with keys representing bill face values (100, 50, 20, 10, 5, 1) and values representing the count of each denomination. Used for reconciliation against currency discriminator machine count output.",
                    "properties": {
                    "100": {
                        "type": "integer",
                        "description": "Count of $100 bills in the cash deposit."
                    },
                    "50": {
                        "type": "integer",
                        "description": "Count of $50 bills in the cash deposit."
                    },
                    "20": {
                        "type": "integer",
                        "description": "Count of $20 bills in the cash deposit."
                    },
                    "10": {
                        "type": "integer",
                        "description": "Count of $10 bills in the cash deposit."
                    },
                    "5": {
                        "type": "integer",
                        "description": "Count of $5 bills in the cash deposit."
                    },
                    "1": {
                        "type": "integer",
                        "description": "Count of $1 bills in the cash deposit."
                    }
                    }
                },
                "cash_counter_machine_id": {
                    "type": "string",
                    "description": "Unique identifier for the specific currency discriminator hardware device processing the cash validation, formatted as CCM_BRnnnn_nn, enabling device-level performance tracking, counterfeit detection accuracy monitoring, and maintenance scheduling based on usage patterns."
                }
                },
                "required": [
                "transaction_id",
                "funding_source_type",
                "deposit_amount",
                "cash_denomination_breakdown",
                "cash_counter_machine_id"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Fraud_Detection_Scoring_API",
            "description": "Executes real-time behavioral analytics fraud detection through machine learning models trained on historical fraud incident patterns. Retrieves customer transaction profile baselines from CRM data warehouse, calculates z-score deviations for deposit amounts and behavioral characteristics, evaluates geographic impossibility scenarios and channel switching patterns, and generates probabilistic fraud_detection_score through weighted logistic regression aggregation of individual risk signals. Calculates real-time fraud risk probability scores based on transaction behavioral deviations from customer baseline patterns and detects account takeover indicators and geographic anomalies.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to correlate fraud detection scoring results with the transaction record and enable fraud analyst case investigation linking to comprehensive transaction details."
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer Information File identifier used to retrieve historical transaction profile from CRM data warehouse including statistical baseline measures (mean, standard deviation of deposit amounts, modal channels, temporal patterns) for behavioral deviation analysis."
                },
                "account_number": {
                    "type": "string",
                    "description": "Account number receiving the deposit, used to retrieve account-specific transaction history and detect account takeover patterns such as sudden changes in transaction behavior following credential compromise or social engineering attacks."
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the deposit amount, compared against customer historical mean and standard deviation to calculate z-score representing number of standard deviations from baseline, with z-scores exceeding 3.0 indicating statistically significant deviation elevating fraud probability."
                },
                "deposit_channel": {
                    "type": "string",
                    "description": "Channel through which the deposit transaction is being processed, evaluated for consistency with customer established patterns. First-time utilization of previously unused channels or switching from established patterns (e.g., branch to mobile) triggers fraud risk elevation. Valid values: BRANCH_TELLER, ATM, MOBILE_APP, ONLINE_BANKING.",
                    "enum": ["BRANCH_TELLER", "ATM", "MOBILE_APP", "ONLINE_BANKING"]
                },
                "deposit_type": {
                    "type": "string",
                    "description": "Type of deposit transaction being processed, evaluated against customer product holdings and typical transaction patterns. Unusual deposit types relative to customer profile (e.g., term deposit for transaction-focused customer) may indicate account takeover. Valid values: RETAIL_DEMAND, TERM_DEPOSIT, CORPORATE_DEPOSIT, CASH_COUNTER, MOBILE_DEPOSIT.",
                    "enum": ["RETAIL_DEMAND", "TERM_DEPOSIT", "CORPORATE_DEPOSIT", "CASH_COUNTER", "MOBILE_DEPOSIT"]
                },
                "branch_code": {
                    "type": "string",
                    "description": "Unique identifier for the branch location processing the transaction, formatted as BR_STATE_nnnn, used to calculate geographic entropy of customer location usage patterns and detect geographic impossibility scenarios where sequential transactions occur at locations separated by distances exceeding feasible travel velocities."
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking transaction initiation, used to calculate velocity metrics (transaction frequency within rolling time windows) and detect temporal clustering patterns inconsistent with customer baseline behavior (e.g., hour-of-day, day-of-week distributions)."
                }
                },
                "required": [
                "transaction_id",
                "customer_id",
                "account_number",
                "deposit_amount",
                "deposit_channel",
                "deposit_type",
                "branch_code",
                "transaction_timestamp"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Supervisor_Approval_Routing_API",
            "description": "Evaluates multiple conditional criteria against employee authorization matrices to determine supervisor_approval_required flag status. Assesses whether deposit amounts exceed teller authority limits, aggregates risk indicators from KYC, AML, CTR, counterfeit detection, and fraud scoring workflows. Implements tiered escalation logic routing transactions to Branch Manager or Regional Manager approval queues with comprehensive risk context presentation and captures supervisor_id and supervisor_approval_timestamp upon approval execution. Determines supervisory approval necessity based on transaction amount thresholds and aggregated risk assessment outcomes.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to route transaction to supervisor approval queue with complete transaction context and enable supervisor retrieval of transaction details for informed approval decision-making."
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the deposit amount, compared against teller authority limits retrieved from employee authorization matrix based on teller_id. Transactions exceeding authority limits (typically $5,000 for standard tellers, $10,000 for personal bankers) trigger supervisor_approval_required flag."
                },
                "teller_id": {
                    "type": "string",
                    "description": "Employee identifier for the teller or banker processing the transaction, formatted as EMP_nnnnn, used to retrieve individual transaction authority limits from employee authorization matrix and attribute transaction to specific employee for performance monitoring and quality assurance sampling."
                },
                "kyc_verification_status": {
                    "type": "string",
                    "description": "Status of Know Your Customer verification process, with values of REQUIRES_REVIEW triggering supervisor_approval_required flag regardless of amount, as incomplete customer identification documentation represents unacceptable compliance risk. Valid values: VERIFIED, REQUIRES_REVIEW, PENDING.",
                    "enum": ["VERIFIED", "REQUIRES_REVIEW", "PENDING"]
                },
                "aml_screening_result": {
                    "type": "string",
                    "description": "Result of Anti-Money Laundering sanctions screening, with values of WATCHLIST_HIT or MANUAL_REVIEW_REQUIRED triggering supervisor_approval_required flag to ensure Compliance Officer review of potential sanctions matches or elevated AML risk conditions. Valid values: CLEAR, WATCHLIST_HIT, MANUAL_REVIEW_REQUIRED.",
                    "enum": ["CLEAR", "WATCHLIST_HIT", "MANUAL_REVIEW_REQUIRED"]
                },
                "aml_risk_score": {
                    "type": "integer",
                    "description": "Integer value on 0-100 scale representing composite Anti-Money Laundering risk assessment, with scores exceeding institutional threshold of 75 triggering supervisor_approval_required flag to ensure human analytical review of elevated risk transactions by Compliance Officer."
                },
                "fraud_detection_score": {
                    "type": "number",
                    "description": "Floating-point value on 0.0-1.0 scale representing probabilistic fraud risk assessment, with scores exceeding institutional threshold of 0.70 triggering supervisor_approval_required flag to ensure fraud analyst investigation and customer verification through out-of-band communication channels."
                },
                "counterfeit_detection_result": {
                    "type": "string",
                    "description": "Result of physical currency validation through currency discriminator hardware, with FAIL status indicating counterfeit bills detected triggering supervisor_approval_required flag for Branch Manager intervention and potential law enforcement notification. Valid values: PASS, FAIL, NOT_APPLICABLE.",
                    "enum": ["PASS", "FAIL", "NOT_APPLICABLE"]
                },
                "ctr_filing_required": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether Currency Transaction Report filing is required per 31 USC 5313, with true values in conjunction with transaction_notes containing keywords ('unusual', 'suspicious', 'multiple parties') triggering supervisor_approval_required flag for enhanced management awareness of reportable transactions."
                },
                "transaction_notes": {
                    "type": "string",
                    "description": "Free-text field containing teller observations, customer requests, or service quality notations, algorithmically scanned for keywords ('unusual', 'suspicious', 'multiple parties', 'source of funds') that elevate supervisor_approval_required flag regardless of other risk indicators."
                }
                },
                "required": [
                "transaction_id",
                "deposit_amount",
                "teller_id",
                "kyc_verification_status",
                "aml_screening_result",
                "aml_risk_score",
                "fraud_detection_score",
                "counterfeit_detection_result",
                "ctr_filing_required",
                "transaction_notes"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Term_Deposit_Processing_API",
            "description": "Executes specialized term deposit product processing logic including retrieval of treasury-approved interest_rate_apy from rate matrix tables indexed by term_length_months and deposit_amount tier thresholds. Calculates maturity_date through calendar month addition with business day adjustment logic, computes calculated_interest_at_maturity using compound interest formula with configurable compounding frequency, generates unique cd_certificate_number identifiers, and establishes automated maturity notification calendar triggers. Processes term deposit specific calculations and certificate generation for TERM_DEPOSIT transaction types.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to link term deposit certificate generation results to the transaction record and establish audit trail correlation for regulatory examination of certificate issuance procedures."
                },
                "deposit_type": {
                    "type": "string",
                    "description": "Type of deposit transaction being processed, with Term_Deposit_Processing_API conditionally executing only when deposit_type equals TERM_DEPOSIT, as retail demand and other deposit types do not require certificate generation or maturity calculations. Valid values: RETAIL_DEMAND, TERM_DEPOSIT, CORPORATE_DEPOSIT, CASH_COUNTER, MOBILE_DEPOSIT.",
                    "enum": ["RETAIL_DEMAND", "TERM_DEPOSIT", "CORPORATE_DEPOSIT", "CASH_COUNTER", "MOBILE_DEPOSIT"]
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Principal amount of the term deposit in United States Dollars, used as index key along with term_length_months to retrieve appropriate interest_rate_apy from treasury rate matrix, with rate stratification typically occurring at $25,000, $50,000, and $100,000 tier thresholds offering progressively enhanced rate premiums."
                },
                "term_length_months": {
                    "type": "integer",
                    "description": "Duration of the term deposit in calendar months, used as primary index key to retrieve corresponding interest_rate_apy from rate matrix and calculate maturity_date through calendar month addition logic. Typical term offerings: 3, 6, 12, 24, 36, 60 months."
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking transaction execution time, serving as the origination date for maturity_date calculation through addition of term_length_months with modified following business day convention for month-end date handling."
                },
                "interest_rate_apy": {
                    "type": "number",
                    "description": "Annual Percentage Yield expressed as percentage value (e.g., 4.25 represents 4.25% APY), representing the treasury-approved interest rate for the specific term_length_months and deposit_amount tier combination, used in compound interest formula for calculated_interest_at_maturity computation."
                },
                "early_withdrawal_penalty_rate": {
                    "type": "number",
                    "description": "Penalty rate expressed as decimal fraction (e.g., 0.5 represents forfeiture of 50% of interest earned) applied to early withdrawal scenarios, disclosed to customer through Truth in Savings Act-compliant documentation and incorporated into certificate terms typically as 90 days interest for terms <12 months, 180 days for longer terms."
                }
                },
                "required": [
                "transaction_id",
                "deposit_type",
                "deposit_amount",
                "term_length_months",
                "transaction_timestamp",
                "interest_rate_apy",
                "early_withdrawal_penalty_rate"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Compliance_Aggregation_API",
            "description": "Performs comprehensive compliance verification aggregation by implementing logical AND operation across all mandatory control checkpoint boolean states. Evaluates whether kyc_verification_status equals VERIFIED, aml_screening_result equals CLEAR or has documented override approval, counterfeit_detection_result passes validation, fraud_detection_score falls below institutional tolerance thresholds, and supervisor approvals are satisfied when required. Determines final compliance_check_passed boolean and transaction_approved status and populates approval_rejection_reason field with structured rejection rationale when any control checkpoint fails.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to correlate aggregated compliance determination with transaction record and enable audit trail reconstruction of approval decision logic and control checkpoint evaluation results."
                },
                "kyc_verification_status": {
                    "type": "string",
                    "description": "Status of Know Your Customer verification process, with compliance_check_passed requiring kyc_verification_status to equal VERIFIED. Values of REQUIRES_REVIEW or PENDING result in compliance failure and transaction rejection with approval_rejection_reason 'KYC verification incomplete - customer documentation insufficient'. Valid values: VERIFIED, REQUIRES_REVIEW, PENDING.",
                    "enum": ["VERIFIED", "REQUIRES_REVIEW", "PENDING"]
                },
                "aml_screening_result": {
                    "type": "string",
                    "description": "Result of Anti-Money Laundering sanctions screening, with compliance_check_passed requiring aml_screening_result to equal CLEAR or have documented Compliance Officer override approval. WATCHLIST_HIT or MANUAL_REVIEW_REQUIRED without override results in rejection with approval_rejection_reason 'Failed AML screening - WATCHLIST_HIT without override approval'. Valid values: CLEAR, WATCHLIST_HIT, MANUAL_REVIEW_REQUIRED.",
                    "enum": ["CLEAR", "WATCHLIST_HIT", "MANUAL_REVIEW_REQUIRED"]
                },
                "aml_risk_score": {
                    "type": "integer",
                    "description": "Integer value on 0-100 scale representing composite Anti-Money Laundering risk assessment, evaluated as informational context for compliance decision but not as independent rejection criterion unless aml_screening_result indicates manual review requirement."
                },
                "counterfeit_detection_result": {
                    "type": "string",
                    "description": "Result of physical currency validation, with compliance_check_passed requiring counterfeit_detection_result to equal PASS or NOT_APPLICABLE (for non-cash transactions). FAIL status results in rejection with approval_rejection_reason 'Counterfeit currency detected during validation'. Valid values: PASS, FAIL, NOT_APPLICABLE.",
                    "enum": ["PASS", "FAIL", "NOT_APPLICABLE"]
                },
                "fraud_detection_score": {
                    "type": "number",
                    "description": "Floating-point value on 0.0-1.0 scale representing probabilistic fraud risk assessment, with compliance_check_passed requiring fraud_detection_score to be below institutional threshold of 0.70 or have documented fraud analyst override approval. Scores ≥0.70 without override result in rejection with approval_rejection_reason 'Fraud score exceeds threshold without analyst approval'."
                },
                "supervisor_approval_required": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether supervisory approval is required based on amount thresholds or risk indicators, with compliance_check_passed requiring either supervisor_approval_required to equal false or when true, valid supervisor_id and supervisor_approval_timestamp must be provided."
                },
                "supervisor_id": {
                    "type": "string",
                    "description": "Employee identifier for the supervisor who approved the transaction, formatted as EMP_nnnnn, required when supervisor_approval_required equals true. Empty string value when supervisor_approval_required is true results in rejection with approval_rejection_reason 'Required supervisor approval not obtained'."
                },
                "supervisor_approval_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking when supervisor approval was granted, required when supervisor_approval_required equals true. Empty string value when supervisor_approval_required is true results in compliance failure."
                },
                "ctr_filing_required": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether Currency Transaction Report filing is required per 31 USC 5313, with compliance_check_passed requiring either ctr_filing_required to equal false or when true, confirmation that CTR has been successfully submitted to FinCEN BSA E-Filing system with confirmation number stored in supplementary audit tables."
                }
                },
                "required": [
                "transaction_id",
                "kyc_verification_status",
                "aml_screening_result",
                "aml_risk_score",
                "counterfeit_detection_result",
                "fraud_detection_score",
                "supervisor_approval_required",
                "supervisor_id",
                "supervisor_approval_timestamp",
                "ctr_filing_required"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Account_Ledger_Posting_API",
            "description": "Executes atomic transaction commit operation implementing double-entry accounting principles by simultaneously debiting appropriate asset accounts based on funding_source_type and crediting customer deposit account general ledger. Calculates new_account_balance through decimal precision arithmetic addition of deposit_amount to previous_account_balance, generates globally unique transaction_reference_id, updates account_status to ACTIVE, triggers ancillary provisioning workflows for debit card issuance and online banking enablement, records processing_time_seconds performance metric, and ensures general ledger coding for regulatory reserve requirement reporting.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to establish immutable linkage between ledger posting entries and transaction record for audit trail reconstruction and general ledger reconciliation procedures during month-end close processes."
                },
                "account_number": {
                    "type": "string",
                    "description": "Account number receiving the deposit credit, validated to confirm account exists in Core Banking System, maintains account_status of ACTIVE rather than FROZEN/DORMANT/CLOSED, and permits deposit transactions according to product rules engine configuration for the account_type classification."
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the deposit amount in United States Dollars, added to previous_account_balance using decimal precision arithmetic (not floating-point) to prevent rounding errors that accumulate across transaction sequences and result in balance discrepancies during reconciliation procedures."
                },
                "previous_account_balance": {
                    "type": "number",
                    "description": "Baseline ledger balance retrieved from authoritative Core Banking System ledger in real-time immediately prior to transaction processing to prevent race conditions in concurrent transaction scenarios. Used in arithmetic addition with deposit_amount to calculate new_account_balance."
                },
                "funding_source_type": {
                    "type": "string",
                    "description": "Type of funding mechanism determining debit-side general ledger account for double-entry posting: CASH debits branch vault GL account, INTERNAL_TRANSFER debits funding_source_account customer ledger, CHECK debits check clearing suspense account. Valid values: CASH, CHECK, INTERNAL_TRANSFER, WIRE, ACH.",
                    "enum": ["CASH", "CHECK", "INTERNAL_TRANSFER", "WIRE", "ACH"]
                },
                "funding_source_account": {
                    "type": "string",
                    "description": "Source account number for INTERNAL_TRANSFER funding type, validated to confirm sufficient funds availability before posting. Empty string for other funding_source_type values where external funding mechanisms apply (cash, check, wire, ACH)."
                },
                "branch_code": {
                    "type": "string",
                    "description": "Unique identifier for the branch location processing the transaction, formatted as BR_STATE_nnnn, used to construct branch-specific general ledger account codes for CASH funding debit entries (concatenating branch_code with cash asset account code) and regional performance attribution reporting."
                },
                "account_type": {
                    "type": "string",
                    "description": "Product classification of the account receiving deposit, determining general ledger account code for credit entry and regulatory reserve requirement categorization (interest-bearing vs non-interest-bearing) under Federal Reserve Regulation D. Valid values: SAVINGS, CHECKING, MONEY_MARKET, CD_1Y, CD_6M, CD_5Y, BUSINESS_CHECKING.",
                    "enum": ["SAVINGS", "CHECKING", "MONEY_MARKET", "CD_1Y", "CD_6M", "CD_5Y", "BUSINESS_CHECKING"]
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking transaction initiation, used to calculate processing_time_seconds by measuring elapsed duration to final ledger commit operation and to generate transaction_reference_id incorporating timestamp encoding for global uniqueness."
                },
                "compliance_check_passed": {
                    "type": "boolean",
                    "description": "Boolean flag from Compliance_Aggregation_API indicating all mandatory control checkpoints passed validation. Must equal true as mandatory prerequisite for ledger posting execution. False values trigger HTTP 412 Precondition Failed response preventing financial posting of non-compliant transactions."
                },
                "transaction_approved": {
                    "type": "boolean",
                    "description": "Boolean flag from Compliance_Aggregation_API indicating final approval determination. Must equal true as mandatory prerequisite for ledger posting execution. False values prevent ledger posting and trigger transaction rejection notification workflows."
                },
                "cd_certificate_number": {
                    "type": "string",
                    "description": "Certificate number from Term_Deposit_Processing_API formatted as CD_CERT_YYYY_nnnnnn for term deposit transactions, used to link general ledger posting to certificate instrument record. Empty string for non-term deposit transaction types."
                },
                "debit_card_issued": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether debit card provisioning workflow should be triggered, with true values initiating card production batch processing with card number assignment through Luhn algorithm-compliant generation and linkage to account_number for POS/ATM authorization."
                },
                "online_banking_enabled": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether online banking provisioning workflow should be triggered, with true values generating user credentials, establishing security question protocols, and enabling digital channel access for account management and transaction initiation."
                },
                "mobile_app_enabled": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether mobile application provisioning workflow should be triggered, with true values generating device registration tokens and enabling mobile deposit, mobile payments, and push notification capabilities."
                },
                "checkbook_ordered": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether checkbook printing order workflow should be triggered, with true values submitting printing orders to check stock vendor with customer_name and customer_address imprinting specifications."
                }
                },
                "required": [
                "transaction_id",
                "account_number",
                "deposit_amount",
                "previous_account_balance",
                "funding_source_type",
                "funding_source_account",
                "branch_code",
                "account_type",
                "transaction_timestamp",
                "compliance_check_passed",
                "transaction_approved",
                "cd_certificate_number",
                "debit_card_issued",
                "online_banking_enabled",
                "mobile_app_enabled",
                "checkbook_ordered"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Receipt_Generation_API",
            "description": "Generates comprehensive transaction documentation for customer delivery including receipt_number formatted with branch and date identifiers. Retrieves and formats transaction details with currency symbols and comma separators, incorporates Truth in Savings Act mandatory disclosures for interest-bearing accounts including APY and early withdrawal penalties. Executes receipt delivery via specified receipt_delivery_method including printed thermal output, email PDF transmission, SMS text notification, or mobile app push notification. Updates customer profile in CRM with relationship value recalculation and next-best-action recommendations.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction, used to correlate receipt generation with transaction record and enable customer service representatives to retrieve receipt copies during subsequent inquiry resolution or dispute investigation procedures."
                },
                "customer_name": {
                    "type": "string",
                    "description": "Full legal name of the customer printed on receipt header for customer identification and record-keeping purposes, matching name from official identification documents validated during KYC verification procedures."
                },
                "account_number": {
                    "type": "string",
                    "description": "Account number receiving the deposit, rendered in receipt output as masked format displaying only final four digits (****-****-****-nnnn) to protect account confidentiality per PCI DSS data protection standards while enabling customer account identification."
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of the deposit amount formatted on receipt with United States Dollar currency symbol, comma thousand separators, and two decimal precision (e.g., $5,000.00) for clear customer communication of transaction value."
                },
                "new_account_balance": {
                    "type": "number",
                    "description": "Post-transaction ledger balance from Account_Ledger_Posting_API formatted with currency symbol and comma separators, providing customer with definitive confirmation of current available funds subject to funds availability schedule overlays for check deposits per Federal Reserve Regulation CC."
                },
                "transaction_reference_id": {
                    "type": "string",
                    "description": "Globally unique transaction reference identifier from Account_Ledger_Posting_API formatted as REF_YYYYMMDDHHmmssnnnnnn, printed prominently on receipt for customer record keeping and future inquiry reference enabling rapid transaction retrieval during research or dispute resolution."
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) from transaction initiation, converted to human-readable twelve-hour clock format with meridiem indicator and time zone designation (e.g., 'April 18, 2025 10:23:45 AM EDT') for receipt printing."
                },
                "branch_code": {
                    "type": "string",
                    "description": "Unique identifier for the branch location processing the transaction, formatted as BR_STATE_nnnn, used to construct receipt_number with branch identification component and enable receipt numbering schema supporting rapid retrieval based on branch location and date parameters."
                },
                "receipt_delivery_method": {
                    "type": "string",
                    "description": "Delivery mechanism for receipt transmission, with PRINTED triggering thermal printer output, EMAIL transmitting formatted PDF attachment with TLS encryption and SPF validation, SMS sending abbreviated confirmation to customer_phone, MOBILE_APP pushing notification through API to authenticated device. Valid values: PRINTED, EMAIL, SMS, MOBILE_APP.",
                    "enum": ["PRINTED", "EMAIL", "SMS", "MOBILE_APP"]
                },
                "customer_email": {
                    "type": "string",
                    "description": "Primary email address for customer communication, required when receipt_delivery_method equals EMAIL, validated for RFC 5322 compliance and domain verification, used for PDF receipt attachment transmission with transport layer encryption."
                },
                "customer_phone": {
                    "type": "string",
                    "description": "Primary contact telephone number in international format +n-nnn-nnn-nnnn, required when receipt_delivery_method equals SMS, used for abbreviated transaction confirmation transmission via secure short message service gateway."
                },
                "cd_certificate_number": {
                    "type": "string",
                    "description": "Certificate number from Term_Deposit_Processing_API formatted as CD_CERT_YYYY_nnnnnn for term deposit transactions, printed on receipt to document certificate issuance. Empty string for non-term deposit transaction types."
                },
                "maturity_date": {
                    "type": "string",
                    "description": "ISO 8601 formatted date for term deposit maturity from Term_Deposit_Processing_API, converted to full month name and four-digit year format (e.g., 'April 18, 2026') for receipt printing with Truth in Savings Act disclosure requirements. Empty string for non-term deposits."
                },
                "interest_rate_apy": {
                    "type": "number",
                    "description": "Annual Percentage Yield expressed as percentage value (e.g., 2.50 represents 2.50% APY), formatted on receipt as 'APY: n.nn%' per Truth in Savings Act mandatory disclosure requirements for interest-bearing accounts. Zero value for non-interest-bearing checking accounts."
                },
                "calculated_interest_at_maturity": {
                    "type": "number",
                    "description": "Total interest earnings in United States Dollars from Term_Deposit_Processing_API formatted as currency value (e.g., $1,062.50) on receipt for term deposits, providing customers with definitive projection of investment returns assuming no early withdrawal. Zero value for non-term deposits."
                },
                "transaction_notes": {
                    "type": "string",
                    "description": "Free-text field containing teller observations, customer requests, or service quality notations (e.g., 'Customer requested expedited card delivery', 'Discussed CD laddering strategy') for relationship management context and service quality documentation."
                },
                "transaction_approved": {
                    "type": "boolean",
                    "description": "Boolean flag from Compliance_Aggregation_API indicating final approval determination. Must equal true as prerequisite for receipt generation. False values trigger HTTP 400 Bad Request preventing receipt generation for rejected transactions."
                },
                "minimum_balance_requirement": {
                    "type": "number",
                    "description": "Minimum balance threshold required to avoid monthly maintenance fees or qualify for advertised APY, disclosed on receipt per Truth in Savings Act requirements formatted as currency value (e.g., $500.00)."
                },
                "monthly_maintenance_fee": {
                    "type": "number",
                    "description": "Monthly account maintenance fee charged unless minimum balance or activity requirements met, disclosed on receipt per Truth in Savings Act requirements formatted as currency value (e.g., $5.00). Zero value for fee-free accounts."
                },
                "early_withdrawal_penalty_rate": {
                    "type": "number",
                    "description": "Penalty rate for term deposit early withdrawal expressed as decimal fraction (e.g., 0.5 represents forfeiture of 50% of interest earned or 180 days interest), disclosed on receipt with Truth in Savings Act-compliant explanation of penalty calculation methodology. Zero value for non-term deposits."
                },
                "final_transaction_fee": {
                    "type": "number",
                    "description": "Any charges assessed to customer for specialized services such as currency exchange fees for foreign currency deposits, expedited availability fees for next-day check clearing, or account research fees, formatted as currency value. Typically 0.00 for standard deposit transactions."
                }
                },
                "required": [
                "transaction_id",
                "customer_name",
                "account_number",
                "deposit_amount",
                "new_account_balance",
                "transaction_reference_id",
                "transaction_timestamp",
                "branch_code",
                "receipt_delivery_method",
                "customer_email",
                "customer_phone",
                "cd_certificate_number",
                "maturity_date",
                "interest_rate_apy",
                "calculated_interest_at_maturity",
                "transaction_notes",
                "transaction_approved",
                "minimum_balance_requirement",
                "monthly_maintenance_fee",
                "early_withdrawal_penalty_rate",
                "final_transaction_fee"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "Audit_Trail_Recording_API",
            "description": "Creates comprehensive immutable audit trail documentation capturing complete transaction lifecycle from initiation through final approval and posting. Incorporates all schema fields from Existing_fields and Field_to_be_obtained specifications along with supplementary metadata including IP addresses and device fingerprints. Preserves pre-transaction state snapshots and validation check result details, implements blockchain-inspired cryptographic hashing for data integrity verification, establishes multi-dimensional indexing for efficient retrieval, and executes automated data lifecycle management transitioning records through cost-optimized storage tiers.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": "Unique alphanumeric identifier for the deposit transaction serving as primary key for audit trail record retrieval and correlation across all system components involved in transaction processing lifecycle."
                },
                "customer_id": {
                    "type": "string",
                    "description": "Customer Information File identifier used for customer_id indexing enabling relationship-level transaction history queries aggregating all transactions for specific customer across temporal dimensions for pattern analysis and regulatory examination."
                },
                "account_number": {
                    "type": "string",
                    "description": "Account number receiving deposit, stored in audit trail to enable account-level transaction reconstruction and balance reconciliation for discrepancy investigation and point-in-time balance verification during audit procedures."
                },
                "transaction_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime in Zulu time zone (YYYY-MM-DDTHH:mm:ssZ) marking transaction initiation, used for chronological indexing enabling time-series analysis and regulatory examination period-based extraction with query performance requirements mandating <2 second response for single transaction retrieval."
                },
                "deposit_type": {
                    "type": "string",
                    "description": "Type of deposit transaction processed, preserved in audit trail for product-level reporting and statistical analysis of transaction volume distributions across deposit categories. Valid values: RETAIL_DEMAND, TERM_DEPOSIT, CORPORATE_DEPOSIT, CASH_COUNTER, MOBILE_DEPOSIT.",
                    "enum": ["RETAIL_DEMAND", "TERM_DEPOSIT", "CORPORATE_DEPOSIT", "CASH_COUNTER", "MOBILE_DEPOSIT"]
                },
                "deposit_amount": {
                    "type": "number",
                    "description": "Numeric value of deposit amount preserved in audit trail for transaction amount analysis, threshold breach investigations, and aggregation queries supporting CTR structuring pattern detection and suspicious activity monitoring."
                },
                "teller_id": {
                    "type": "string",
                    "description": "Employee identifier for teller/banker processing transaction, used for teller_id indexing enabling employee performance monitoring, quality assurance sampling, segregation of duties compliance verification, and individual accountability attribution. Empty string for automated channel transactions (ATM, mobile, online)."
                },
                "branch_code": {
                    "type": "string",
                    "description": "Unique identifier for branch location processing transaction, preserved for geographic analysis, branch-level performance reporting, and investigation of geographic impossibility fraud patterns through temporal-spatial correlation analysis."
                },
                "kyc_verification_status": {
                    "type": "string",
                    "description": "Final status of Know Your Customer verification from KYC_Verification_API, preserved to demonstrate compliance with Customer Identification Program requirements during regulatory examinations. Valid values: VERIFIED, REQUIRES_REVIEW, PENDING.",
                    "enum": ["VERIFIED", "REQUIRES_REVIEW", "PENDING"]
                },
                "kyc_verification_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime marking KYC verification completion, documenting compliance control execution timing for audit trail completeness and demonstrating adherence to sequential validation workflow requirements."
                },
                "aml_screening_result": {
                    "type": "string",
                    "description": "Final result of Anti-Money Laundering sanctions screening from AML_Sanctions_Screening_API, preserved to demonstrate Bank Secrecy Act compliance during examinations. Valid values: CLEAR, WATCHLIST_HIT, MANUAL_REVIEW_REQUIRED.",
                    "enum": ["CLEAR", "WATCHLIST_HIT", "MANUAL_REVIEW_REQUIRED"]
                },
                "aml_risk_score": {
                    "type": "integer",
                    "description": "Composite AML risk score on 0-100 scale from AML_Sanctions_Screening_API, preserved for risk model performance analysis, false positive rate optimization, and documentation of risk-based decision making rationale supporting transaction approval or rejection determinations."
                },
                "sanctions_screening_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime marking sanctions screening completion, documenting compliance control execution timing and demonstrating real-time screening prior to transaction posting per regulatory requirements."
                },
                "fraud_detection_score": {
                    "type": "number",
                    "description": "Probabilistic fraud risk score on 0.0-1.0 scale from Fraud_Detection_Scoring_API, preserved for fraud model performance analysis, threshold calibration, false positive investigations, and documentation of fraud prevention control effectiveness."
                },
                "counterfeit_detection_result": {
                    "type": "string",
                    "description": "Result of physical currency validation from Currency_Validation_API, preserved to document cash handling controls and counterfeit detection procedure execution for audit verification and incident investigation. Valid values: PASS, FAIL, NOT_APPLICABLE.",
                    "enum": ["PASS", "FAIL", "NOT_APPLICABLE"]
                },
                "supervisor_approval_required": {
                    "type": "boolean",
                    "description": "Boolean flag indicating whether supervisory approval was required based on amount thresholds or risk indicators, preserved to demonstrate adherence to authorization limit controls and segregation of duties principles."
                },
                "supervisor_id": {
                    "type": "string",
                    "description": "Employee identifier for supervisor who approved transaction when supervisor_approval_required equals true, documenting approval chain hierarchy for audit verification of proper authorization controls. Empty string when supervisor approval not required."
                },
                "supervisor_approval_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 formatted datetime marking supervisor approval grant when supervisor_approval_required equals true, documenting approval timing for authorization control verification. Empty string when supervisor approval not required."
                },
                "compliance_check_passed": {
                    "type": "boolean",
                    "description": "Aggregated compliance status from Compliance_Aggregation_API indicating whether all mandatory control checkpoints passed, preserved to provide definitive evidence of compliance validation execution supporting regulatory examination defense."
                },
                "transaction_approved": {
                    "type": "boolean",
                    "description": "Final approval determination from Compliance_Aggregation_API indicating transaction disposition (approved=true, rejected=false), preserved for transaction outcome reporting, approval rate monitoring, and rejection reason trend analysis."
                },
                "approval_rejection_reason": {
                    "type": "string",
                    "description": "Structured text string from Compliance_Aggregation_API identifying specific control failure condition for rejected transactions (e.g., 'Failed AML screening - WATCHLIST_HIT without override approval'), preserved for rejection reason categorization and process improvement analysis. Empty string for approved transactions."
                },
                "new_account_balance": {
                    "type": "number",
                    "description": "Post-transaction ledger balance from Account_Ledger_Posting_API, preserved to enable point-in-time balance reconstruction, discrepancy investigation, and verification that balance calculation arithmetic was performed correctly using decimal precision."
                },
                "transaction_reference_id": {
                    "type": "string",
                    "description": "Globally unique transaction reference identifier from Account_Ledger_Posting_API, preserved to establish immutable linkage between audit trail record and general ledger posting entries for reconciliation and accounting investigation procedures."
                },
                "account_status": {
                    "type": "string",
                    "description": "Account status after transaction posting from Account_Ledger_Posting_API, typically ACTIVE for successful deposits, preserved to document account lifecycle state transitions and dormancy timer reset events. Valid values: ACTIVE, DORMANT, FROZEN, CLOSED.",
                    "enum": ["ACTIVE", "DORMANT", "FROZEN", "CLOSED"]
                },
                "processing_time_seconds": {
                    "type": "integer",
                    "description": "Elapsed duration in seconds from transaction_timestamp to final system commit operation from Account_Ledger_Posting_API, preserved for service level agreement compliance monitoring, performance trend analysis, and bottleneck identification supporting process optimization initiatives."
                },
                "receipt_number": {
                    "type": "string",
                    "description": "Unique receipt identifier from Receipt_Generation_API formatted as RCPT_YYYYMMDD_branchcode_nnn, preserved to enable receipt retrieval during customer inquiry resolution and sequential receipt number gap analysis detecting missing or voided transactions requiring investigation."
                },
                "previous_account_balance": {
                    "type": "number",
                    "description": "Pre-transaction ledger balance retrieved immediately prior to posting, preserved to enable point-in-time balance reconstruction, arithmetic verification of balance calculation (previous_balance + deposit_amount = new_balance), and discrepancy investigation through comparison with Core Banking System ledger snapshots."
                },
                "ctr_filing_required": {
                    "type": "boolean",
                    "description": "Boolean flag from CTR_Evaluation_API indicating Currency Transaction Report filing obligation per 31 USC 5313, preserved to document regulatory reporting compliance and support Bank Secrecy Act examination defense demonstrating timely CTR filing for reportable transactions."
                },
                "cash_drawer_total_after_transaction": {
                    "type": "number",
                    "description": "Teller cash drawer balance after transaction from Currency_Validation_API, preserved for teller balancing reconciliation investigation when end-of-day cash counts reveal discrepancies requiring transaction-level drill-down to identify error sources."
                }
                },
                "required": [
                "transaction_id",
                "customer_id",
                "account_number",
                "transaction_timestamp",
                "deposit_type",
                "deposit_amount",
                "teller_id",
                "branch_code",
                "kyc_verification_status",
                "kyc_verification_timestamp",
                "aml_screening_result",
                "aml_risk_score",
                "sanctions_screening_timestamp",
                "fraud_detection_score",
                "counterfeit_detection_result",
                "supervisor_approval_required",
                "supervisor_id",
                "supervisor_approval_timestamp",
                "compliance_check_passed",
                "transaction_approved",
                "approval_rejection_reason",
                "new_account_balance",
                "transaction_reference_id",
                "account_status",
                "processing_time_seconds",
                "receipt_number",
                "previous_account_balance",
                "ctr_filing_required",
                "cash_drawer_total_after_transaction"
                ]
            }
            }
        }
        }
        </SCHEMA>'''
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
    yaml_file = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_API_Function_Code.yaml"
    system_prompt = "You are an SOP-related API tool Python code implementation generator. The functions you generate, strictly based on the finalized JSON template, must comply with Python syntax rules, and input parameters must be either already present in the sample_data or obtainable through calculations from other API functions. **You must generate all the code at once, without interruption.** The code you generate must also properly maintain and record all data entries contained within <data></data> and allow reading and writing by other APIs. Additionally, design a function to retrieve the corresponding data entry information from the data you maintain based on application_id. Fields that already have values in the data are user's original data; empty fields are data that need to be obtained through API processing, and once obtained via an API, they should be written back to the corresponding entry in the data you maintain, completing the update. The code you generate must consist entirely of deterministic calculations; the use of random numbers, such as random, is strictly prohibited."

    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, api, sample_data)

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
