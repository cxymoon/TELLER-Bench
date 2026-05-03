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

        ## 1. SOP Relevance Analysis

        The 10 APIs defined in the metadata correspond to the comprehensive account opening workflow described in the SOP:

        - **Section 5.1 (CIP Verification)**: Covered by `ValidateApplicationData` and `ExecuteCIPVerification`
        - **Section 5.2 (Watchlist Screening & Risk Assessment)**: Covered by `ScreenWatchlists` and `AssessRiskProfile`
        - **Section 5.3 (Auto-Decision Engine)**: Covered by `ExecuteAutoDecision` and `ProcessManualReview`
        - **Section 5.4 (Account Establishment)**: Covered by `CreateBankAccount`, `ProcessAccountFunding`, and `ProvisionAncillaryProducts`
        - **Section 5.5 (Process Completion)**: Covered by `CalculateProcessMetrics`

        ## 2. Input Parameters Mapping

        Each API requires specific inputs aligned with the SOP's data requirements from Section 4:

        - **Application identifiers**: application_id, customer_type, business_type
        - **Personal information**: applicant_full_name, applicant_date_of_birth, applicant_ssn_tax_id, addresses, contact details
        - **Identity verification**: id_document_type, id_document_number, biometric data, OCR extracted data
        - **Business information**: business_legal_name, business_ein, beneficial_owners, authorized_signers
        - **Risk assessment outputs**: cip_verification_status, watchlist_screening_result, chexsystems_score, fraud_risk_score
        - **Decision outputs**: auto_decision_result, final_approval_status, manual_review_required
        - **Account details**: account_number_generated, routing_number, initial_deposit_amount

        ## 3. Constraints and Special Considerations

        - **Sequential dependencies**: APIs must be called in specific order (ValidateApplicationData → ExecuteCIPVerification → ScreenWatchlists → AssessRiskProfile → ExecuteAutoDecision → conditional ProcessManualReview → CreateBankAccount → ProcessAccountFunding → ProvisionAncillaryProducts → CalculateProcessMetrics)
        - **Regulatory compliance**: CIP requirements (31 CFR 1020.220), OFAC screening (31 CFR Chapter V), Beneficial Ownership (31 CFR 1010.230)
        - **Threshold-based logic**: Specific numeric thresholds for scores (ChexSystems 600+, fraud_risk_score ≤25.0, facial_match_confidence_score ≥92.0)
        - **Channel-specific processing**: Different validation rules for in_branch vs digital_remote channels
        - **Error handling**: Comprehensive error scenarios with retry logic, fallback mechanisms, and manual escalation paths

        ## 4. SOP Alignment

        The API sequence perfectly aligns with the SOP's main procedure flow:
        1. Data validation ensures quality inputs before external verification
        2. CIP verification satisfies regulatory identity confirmation requirements
        3. Watchlist screening ensures OFAC compliance
        4. Risk assessment quantifies deposit account and fraud risk
        5. Auto-decision synthesizes all metrics for approval determination
        6. Manual review provides human adjudication for ambiguous cases
        7. Account creation establishes the system of record
        8. Funding processing activates the account
        9. Ancillary products enable full functionality
        10. Metrics calculation supports operational excellence

        ## 5. Edge Cases and Limitations

        - **OCR data mismatches**: Handled through data_consistency_score calculation with manual review escalation
        - **Biometric failures**: Alternative verification pathways when facial_match_confidence_score inadequate
        - **API timeouts**: Retry logic with exponential backoff for external service failures
        - **Duplicate accounts**: Detection mechanisms for same SSN/account type combinations
        - **Beneficial ownership complexity**: Enhanced validation for corporate structures
        - **High-risk industries**: Automatic EDD escalation for money services, cannabis, cryptocurrency
        - **Abandoned applications**: Tracking of abandonment_stage for UX optimization
        - **Funding delays**: Status transitions to "dormant" after timeout periods
        - **Clock skew**: Absolute value calculations for processing time metrics

        ## 6. Workflow Integration

        The APIs form a complete end-to-end workflow:

        **Entry Point**: ValidateApplicationData (no dependencies)

        **Parallel Eligible**: ExecuteCIPVerification and initial risk checks could theoretically run concurrently but SOP prescribes sequential execution for data quality

        **Decision Points**: 
        - ExecuteAutoDecision determines auto_approved vs auto_declined vs refer_for_manual_review
        - ProcessManualReview conditionally executed only when manual_review_required=true

        **Terminal Operations**:
        - CreateBankAccount establishes the account
        - ProcessAccountFunding activates it
        - ProvisionAncillaryProducts enables services
        - CalculateProcessMetrics finalizes reporting

        **Data Flow**: Each API produces outputs consumed by downstream APIs, creating a strongly-typed pipeline with validation at each stage.

        </sop_tool_analysis>

        <SCHEMA>
        {
        "toolSpec": {
            "name": "ValidateApplicationData",
            "description": "Validates application data format, completeness, and internal consistency including OCR cross-reference for digital applications to ensure data quality before initiating external verification processes. This is the entry point API for account opening workflow with no dependencies. Performs SSN format validation, age verification, name format checks, ID document expiry validation, and calculates data_consistency_score by comparing OCR-extracted data against manually-entered fields. Returns validation status and triggers manual review for data quality issues.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique alphanumeric application identifier conforming to format 'APP_' followed by 10-digit timestamp and 5-digit random sequence (e.g., APP_20250418101234567). Used for unambiguous application tracking across distributed system components."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name of the applicant as it appears on government-issued identification without abbreviations. Must contain both given name and surname. Validated to prevent numeric or special character inclusion except legitimate apostrophes and hyphens in compound names."
                },
                "applicant_date_of_birth": {
                    "type": "string",
                    "description": "Applicant's date of birth in ISO 8601 date format (YYYY-MM-DD). Used for age verification to ensure applicant is between 18 and 120 years old for individual accounts or to identify minor status requiring custodial account setup."
                },
                "applicant_ssn_tax_id": {
                    "type": "string",
                    "description": "Nine-digit Social Security Number or Employer Identification Number in format XXX-XX-XXXX with terminal four digits visible. Validated for 9-digit structure with non-sequential and non-repetitive characteristics to prevent invalid SSN patterns."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete current physical residence including street number, directional prefixes, street name, apartment or unit designations, city, state abbreviation, and five-digit ZIP code. Used for CIP address verification requirements."
                },
                "applicant_email": {
                    "type": "string",
                    "description": "Valid electronic mail address for correspondence and online banking credential delivery. Must conform to standard email format validation."
                },
                "applicant_phone": {
                    "type": "string",
                    "description": "Primary contact telephone number in E.164 international format (e.g., +1-555-234-5678). Used for OTP verification and customer communication."
                },
                "id_document_type": {
                    "type": "string",
                    "description": "Type of government-issued identification document presented. Accepted values: 'passport', 'drivers_license', or 'state_id'. Determines document authentication procedures."
                },
                "id_document_number": {
                    "type": "string",
                    "description": "Unique identifier printed on the government-issued identification credential. Used for identity verification cross-referencing."
                },
                "id_document_expiry_date": {
                    "type": "string",
                    "description": "Expiration date of the identification document in ISO 8601 date format (YYYY-MM-DD). Must be after current date to satisfy CIP validity requirements. Expired documents trigger manual review for updated documentation."
                },
                "application_data_entry_method": {
                    "type": "string",
                    "description": "Classification specifying data capture methodology. Accepted values: 'manual_banker_entry' for in-branch applications, 'ocr_auto_fill' for digital applications with OCR extraction, or 'customer_self_service' for web/mobile self-service. Determines OCR consistency validation requirements."
                },
                "id_ocr_extracted_data": {
                    "type": "object",
                    "description": "JSON structure containing parsed fields from automated Optical Character Recognition document reading including name, date of birth, document number, and address elements. Empty object {} for manual entry applications. Used for cross-reference validation against manually-confirmed application fields to calculate data_consistency_score where perfect alignment yields 100.0 and discrepancies reduce score by 25.0 points per mismatched field."
                },
                "customer_type": {
                    "type": "string",
                    "description": "Classification of account relationship type. Accepted values: 'individual' for natural person consumer accounts, 'sole_proprietorship' for individuals with business tax identification, 'llc' for limited liability companies, 'corporation' for incorporated entities, or 'partnership' for multi-party business arrangements. Governs beneficial ownership collection requirements."
                },
                "business_type": {
                    "type": "string",
                    "description": "Taxonomic classification of application workflow. Accepted values: 'retail_individual' for in-branch personal banker-assisted applications, 'digital_remote' for technology-mediated self-service with biometric verification, or 'corporate_business' for legal entity accounts requiring beneficial ownership documentation. Routes application through type-appropriate verification protocols."
                }
                },
                "required": [
                "application_id",
                "applicant_full_name",
                "applicant_date_of_birth",
                "applicant_ssn_tax_id",
                "applicant_residential_address",
                "applicant_email",
                "applicant_phone",
                "id_document_type",
                "id_document_number",
                "id_document_expiry_date",
                "application_data_entry_method",
                "id_ocr_extracted_data",
                "customer_type",
                "business_type"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ExecuteCIPVerification",
            "description": "Orchestrates comprehensive Customer Identification Program verification including document authentication, biometric liveness checks, facial matching, and credit bureau identity validation to satisfy regulatory CIP requirements under 31 CFR 1020.220. Invokes Identity Verification Service platform to analyze identification document security features, perform facial biometric comparison for digital applications, and execute credit bureau validation through consumer reporting agencies. Returns definitive cip_verification_status classification of 'verified', 'failed', or 'pending_manual_review' based on document_authenticity_score, facial_match_confidence_score, and credit_bureau_validation_status thresholds.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier from ValidateApplicationData API. Used for tracking verification process across distributed system components."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name for identity verification service submission and credit bureau validation matching. Must match government-issued identification."
                },
                "applicant_date_of_birth": {
                    "type": "string",
                    "description": "Date of birth in ISO 8601 format for credit bureau identity validation cross-referencing and age-based verification logic."
                },
                "applicant_ssn_tax_id": {
                    "type": "string",
                    "description": "Social Security Number for credit bureau validation and identity verification service submission. Transmitted through secure TLS 1.3 encrypted API endpoints."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete physical address for credit bureau validation matching and identity verification service address consistency analysis."
                },
                "id_document_type": {
                    "type": "string",
                    "description": "Type of government-issued ID ('passport', 'drivers_license', 'state_id') for document authentication security feature analysis specific to credential type."
                },
                "id_document_number": {
                    "type": "string",
                    "description": "Government-issued identification unique identifier for identity verification service submission and potential database cross-referencing."
                },
                "id_document_issuing_country": {
                    "type": "string",
                    "description": "Country that issued the identification document. Must be 'US' for standard processing pathways. Non-US issuance requires alternative verification methodologies."
                },
                "biometric_liveness_check_completed": {
                    "type": "boolean",
                    "description": "Boolean indicating whether anti-spoofing biometric verification completed during digital application workflow. True for digital_remote applications with video selfie capture, false for in_branch applications with physical document examination."
                },
                "application_channel": {
                    "type": "string",
                    "description": "Origination pathway ('in_branch', 'mobile_app', 'web_portal') determining applicable verification methodology. Digital channels require facial_match_confidence_score ≥92.0, in-branch relies on physical document examination."
                },
                "data_consistency_score": {
                    "type": "number",
                    "description": "Data consistency score from ValidateApplicationData API ranging 0-100. Scores below 75.0 automatically populate manual_review_required due to OCR data inconsistency requiring verification."
                }
                },
                "required": [
                "application_id",
                "applicant_full_name",
                "applicant_date_of_birth",
                "applicant_ssn_tax_id",
                "applicant_residential_address",
                "id_document_type",
                "id_document_number",
                "id_document_issuing_country",
                "biometric_liveness_check_completed",
                "application_channel",
                "data_consistency_score"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ScreenWatchlists",
            "description": "Performs OFAC sanctions screening and watchlist matching against Specially Designated Nationals List, Consolidated Sanctions List, global sanctions databases, and politically exposed person databases using fuzzy matching algorithms to ensure regulatory compliance with OFAC requirements under 31 CFR Chapter V. Executes phonetic encoding and edit distance calculations to account for name variations, transliteration inconsistencies, and typographical errors. Generates individual similarity scores for potential matches with threshold-based classification: 'clear' (no matches >70%), 'potential_match' (85.0-94.9% similarity requiring manual disambiguation), or 'confirmed_match' (≥95.0% similarity or 85.0-94.9% with corroborating DOB/address requiring immediate decline and SAR filing).",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for tracking watchlist screening process and associating results with application record."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name to be screened against watchlist databases. Primary matching field for fuzzy name matching algorithms with phonetic encoding."
                },
                "applicant_date_of_birth": {
                    "type": "string",
                    "description": "Date of birth in ISO 8601 format for disambiguation purposes when potential name matches detected. DOB alignment within 30 days increases similarity score weight."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete physical address for geographic risk assessment and secondary identifier corroboration. Address geographic concordance at city level supports confirmed_match determination."
                },
                "applicant_ssn_tax_id": {
                    "type": "string",
                    "description": "Social Security Number or Tax ID as unique identifier for watchlist screening and potential match disambiguation."
                },
                "cip_verification_status": {
                    "type": "string",
                    "description": "CIP verification status from ExecuteCIPVerification API. Must equal 'verified' before watchlist screening proceeds per SOP Section 5.2.1 workflow sequencing. Values: 'verified', 'failed', 'pending_manual_review'."
                },
                "business_legal_name": {
                    "type": "string",
                    "description": "Legal business name for corporate applications to be screened against entity watchlists. Empty string for individual applications. Required for corporate_business applications."
                },
                "beneficial_owners": {
                    "type": "array",
                    "description": "List of JSON objects containing beneficial owner information for corporate applications per 31 CFR 1010.230 requirements. Each beneficial owner must be individually screened against watchlists. Empty array for individual applications. Each object contains name, ownership_percentage, ssn, and dob fields."
                }
                },
                "required": [
                "application_id",
                "applicant_full_name",
                "applicant_date_of_birth",
                "applicant_residential_address",
                "applicant_ssn_tax_id",
                "cip_verification_status",
                "business_legal_name",
                "beneficial_owners"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "AssessRiskProfile",
            "description": "Calculates comprehensive risk assessment including ChexSystems deposit account history score retrieval (300-899 scale quantifying likelihood of account mismanagement) and multi-factor fraud risk scoring (0-100 scale) based on behavioral analytics, device fingerprinting, income coherence analysis, address validation consistency, and data integrity cross-checks. Implements weighted scoring model: data consistency (15%), address alignment (20%), income coherence (15%), deposit amount reasonableness (10%), device fingerprinting (20%), behavioral analytics (20%). Determines due_diligence_level classification ('standard_cdd' or 'enhanced_edd') based on business_industry risk categorization. Triggers manual review for ChexSystems scores 400-499, fraud_risk_score 40.1-60.0, or high-risk industry associations.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for tracking risk assessment process and associating calculated risk metrics with application record."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name for ChexSystems consumer score retrieval and fraud risk identity analysis."
                },
                "applicant_ssn_tax_id": {
                    "type": "string",
                    "description": "Social Security Number for ChexSystems reporting agency transmission and credit bureau cross-referencing. Required for FCRA-compliant consumer reporting access."
                },
                "applicant_date_of_birth": {
                    "type": "string",
                    "description": "Date of birth in ISO 8601 format for ChexSystems matching and age-based risk factor analysis."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete physical address for ChexSystems transmission and address validation consistency component of fraud risk scoring (20% weight). Cross-referenced against credit bureau-reported addresses."
                },
                "applicant_employment_status": {
                    "type": "string",
                    "description": "Current occupational situation classification. Values: 'employed', 'self_employed', 'unemployed', 'student', 'retired'. Used for income coherence analysis component (15% weight) - unemployed claiming >$30K adds 18 risk points, student claiming >$60K adds 10 points."
                },
                "applicant_annual_income": {
                    "type": "number",
                    "description": "Gross yearly earnings in integer dollar amounts. Used for income coherence component validation against employment_status and deposit amount reasonableness component (10% weight) - deposits exceeding 25% of annual income add 15 risk points."
                },
                "initial_deposit_amount": {
                    "type": "number",
                    "description": "Monetary value in floating-point format representing funding commitment. Analyzed in deposit amount reasonableness component comparing to stated income for new customer relationships."
                },
                "customer_type": {
                    "type": "string",
                    "description": "Account relationship classification ('individual', 'sole_proprietorship', 'llc', 'corporation', 'partnership'). Determines ChexSystems scoring applicability - only 'individual' receives consumer score, corporate entities skip ChexSystems component."
                },
                "cip_verification_status": {
                    "type": "string",
                    "description": "CIP verification status from ExecuteCIPVerification API ensuring identity confirmed before ChexSystems access per FCRA compliance. Values: 'verified', 'failed', 'pending_manual_review'."
                },
                "credit_bureau_validation_status": {
                    "type": "string",
                    "description": "Credit bureau validation outcome from ExecuteCIPVerification API used in fraud risk calculation. Values: 'verified' scores 100 points in 10% weight component, 'not_verified' triggers manual review, 'insufficient_data' scores 60 points."
                },
                "facial_match_confidence_score": {
                    "type": "number",
                    "description": "Biometric comparison confidence percentage (0-100) from ExecuteCIPVerification API. Contributes to fraud risk assessment - scores <92.0 indicate potential identity fraud risk requiring enhanced scrutiny."
                },
                "id_ocr_extracted_data": {
                    "type": "object",
                    "description": "JSON structure from ValidateApplicationData containing OCR-parsed identification fields. Used for data consistency component (15% weight) - each inconsistency between OCR and manual data adds 10 risk points to fraud_risk_score."
                },
                "data_consistency_score": {
                    "type": "number",
                    "description": "Data consistency score (0-100) from ValidateApplicationData API. Feeds directly into fraud risk scoring data consistency component calculation with 15% weight."
                },
                "application_channel": {
                    "type": "string",
                    "description": "Origination pathway ('in_branch', 'mobile_app', 'web_portal') determining device fingerprinting and behavioral analytics applicability. Digital channels enable device fingerprinting analysis (20% weight) - VPN/proxy detection adds 20 points, automation tooling adds 40 points."
                },
                "business_industry": {
                    "type": "string",
                    "description": "Business industry classification string for corporate applications. Cross-referenced against high-risk sector taxonomy including money services businesses, cannabis-related enterprises, virtual currency exchangers, international trade facilitators. Empty string for individual applications. High-risk alignment sets due_diligence_level='enhanced_edd' and manual_review_required=true."
                },
                "business_monthly_transaction_volume_estimate": {
                    "type": ["number", "null"],
                    "description": "Integer providing activity forecasting for transaction monitoring calibration. Values exceeding $10,000,000 trigger due_diligence_level='enhanced_edd' requiring enhanced documentation. Null for individual applications."
                }
                },
                "required": [
                "application_id",
                "applicant_full_name",
                "applicant_ssn_tax_id",
                "applicant_date_of_birth",
                "applicant_residential_address",
                "applicant_employment_status",
                "applicant_annual_income",
                "initial_deposit_amount",
                "customer_type",
                "cip_verification_status",
                "credit_bureau_validation_status",
                "facial_match_confidence_score",
                "id_ocr_extracted_data",
                "data_consistency_score",
                "application_channel",
                "business_industry",
                "business_monthly_transaction_volume_estimate"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ExecuteAutoDecision",
            "description": "Synthesizes all risk assessment metrics through multi-factor decision matrix to render automated approval, decline, or manual review referral decision based on institutional risk appetite thresholds. Implements hierarchical evaluation logic: (1) First-level screening eliminates applications with cip_verification_status='failed' or watchlist_screening_result='confirmed_match' for auto_decision_result='auto_declined', (2) Second-level screening identifies marginal risk metrics triggering auto_decision_result='refer_for_manual_review' (ChexSystems 400-499, fraud_risk_score 40.1-60.0, facial_match_confidence_score 88.0-91.9), (3) Third-level applies optimistic auto-approval for all metrics within acceptable thresholds (ChexSystems ≥600, fraud_risk_score ≤25.0, facial_match_confidence_score ≥92.0), (4) Fourth-level calculates weighted composite_risk_score with auto-approval at ≥70.0. Enables straight-through processing for low-risk applications without human intervention.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for tracking auto-decision process and recording decision outcome in application database."
                },
                "cip_verification_status": {
                    "type": "string",
                    "description": "CIP verification status from ExecuteCIPVerification API. First-level screening requirement - must equal 'verified' for approval consideration, 'failed' results in auto_decision_result='auto_declined'. Values: 'verified', 'failed', 'pending_manual_review'."
                },
                "watchlist_screening_result": {
                    "type": "string",
                    "description": "Watchlist screening outcome from ScreenWatchlists API. First-level screening disqualifier - 'confirmed_match' results in auto_decision_result='auto_declined' and decline_reason='Applicant identified on OFAC Specially Designated Nationals List'. Values: 'clear', 'potential_match', 'confirmed_match'."
                },
                "chexsystems_score": {
                    "type": ["number", "null"],
                    "description": "ChexSystems consumer deposit account risk score (100-899) from AssessRiskProfile API. Decision thresholds: ≥600 supports auto-approval, 500-599 conditional approval with enhanced scrutiny, 400-499 triggers manual review, <400 presumptive decline. Null for corporate accounts not subject to ChexSystems scoring."
                },
                "fraud_risk_score": {
                    "type": "number",
                    "description": "Multi-factor composite fraud probability score (0-100) from AssessRiskProfile API. Decision thresholds: ≤25.0 acceptable for auto-approval, 25.1-40.0 moderate concern requiring additional verification, 40.1-60.0 elevated suspicion mandating manual review, >60.0 presumptive decline."
                },
                "credit_bureau_validation_status": {
                    "type": "string",
                    "description": "Credit bureau identity validation outcome from ExecuteCIPVerification API. Contributes 10% weight to composite_risk_score calculation: 'verified' scores 100 points, 'insufficient_data' scores 60 points, 'not_verified' reduces composite score potentially triggering manual review. Values: 'verified', 'not_verified', 'insufficient_data'."
                },
                "facial_match_confidence_score": {
                    "type": "number",
                    "description": "Biometric comparison confidence percentage (0-100) from ExecuteCIPVerification API for digital applications. Decision thresholds: ≥92.0 satisfies auto-approval requirement, 88.0-91.9 triggers manual review for secondary verification, <88.0 requires alternative documentary verification. Contributes 20% weight to composite_risk_score when available."
                },
                "due_diligence_level": {
                    "type": "string",
                    "description": "Due diligence classification from AssessRiskProfile API. 'enhanced_edd' automatically triggers auto_decision_result='refer_for_manual_review' requiring compliance officer assessment for high-risk business profiles. 'standard_cdd' permits automated decisioning. Values: 'standard_cdd', 'enhanced_edd'."
                },
                "customer_type": {
                    "type": "string",
                    "description": "Account relationship classification determining applicable decision logic. Corporate types may bypass ChexSystems scoring requirement. Values: 'individual', 'sole_proprietorship', 'llc', 'corporation', 'partnership'."
                },
                "application_channel": {
                    "type": "string",
                    "description": "Origination pathway determining facial_match_confidence_score requirement applicability. Digital channels ('mobile_app', 'web_portal') require ≥92.0 confidence for auto-approval, 'in_branch' relies on physical document examination. Values: 'in_branch', 'mobile_app', 'web_portal'."
                }
                },
                "required": [
                "application_id",
                "cip_verification_status",
                "watchlist_screening_result",
                "chexsystems_score",
                "fraud_risk_score",
                "credit_bureau_validation_status",
                "facial_match_confidence_score",
                "due_diligence_level",
                "customer_type",
                "application_channel"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ProcessManualReview",
            "description": "Facilitates compliance officer adjudication for applications referred for manual review where auto_decision_result='refer_for_manual_review'. Enables human expert review of watchlist potential matches requiring disambiguation (similarity 85.0-94.9%), investigation of fraud risk factors contributing to elevated scores, assessment of ChexSystems negative events with mitigating circumstances evaluation, verification of beneficial ownership documentation completeness per 31 CFR 1010.230, and enhanced due diligence documentation review for high-risk business classifications. Compliance officer conducts comprehensive file review, validates internal consistency, assesses credibility of unusual circumstances, and documents analytical conclusions in compliance_review_notes. Renders final_approval_status of 'approved', 'declined', or 'conditional_approval' with operational restrictions. Assigns cases based on workload balancing and specialized expertise matching.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for tracking manual review case and associating compliance officer decision with application record."
                },
                "auto_decision_result": {
                    "type": "string",
                    "description": "Auto-decision outcome from ExecuteAutoDecision API. Must equal 'refer_for_manual_review' for this API invocation. Conditional execution trigger. Values: 'auto_approved', 'auto_declined', 'refer_for_manual_review'."
                },
                "manual_review_required": {
                    "type": "boolean",
                    "description": "Boolean flag from ExecuteAutoDecision API confirming manual review necessity. Must be true for compliance officer case assignment and adjudication workflow initiation."
                },
                "manual_review_reason": {
                    "type": "string",
                    "description": "Detailed explanation of specific risk factors or verification failures triggering manual review escalation. Provides compliance officer with investigation focus areas. Examples: 'Facial biometric confidence below threshold requiring secondary verification', 'Potential watchlist match requiring disambiguation', 'ChexSystems score indicates elevated risk requiring assessment'."
                },
                "watchlist_screening_result": {
                    "type": "string",
                    "description": "Watchlist screening outcome from ScreenWatchlists API. 'potential_match' requires compliance officer to conduct enhanced investigation through commercial database services, social media examination, and potentially direct applicant contact for disambiguating documentation. Officer can override to 'clear' or 'confirmed_match' based on investigation findings. Values: 'clear', 'potential_match', 'confirmed_match'."
                },
                "fraud_risk_score": {
                    "type": "number",
                    "description": "Multi-factor fraud probability score (0-100) from AssessRiskProfile API providing compliance officer context for risk assessment. Scores 40.1-60.0 require investigation of contributing factors, >60.0 require extensive documentation review to justify approval."
                },
                "chexsystems_score": {
                    "type": ["number", "null"],
                    "description": "ChexSystems deposit account history score (100-899) from AssessRiskProfile API. Scores 400-499 require assessment of specific negative events and mitigating circumstances evaluation (identity theft victimization, medical financial hardship). Scores <400 require documented approval justification with enhanced monitoring. Null for corporate accounts."
                },
                "due_diligence_level": {
                    "type": "string",
                    "description": "Due diligence classification from AssessRiskProfile API. 'enhanced_edd' requires compliance officer verification of business plan submission, source of funds attestation, detailed transaction forecasting, and potentially site visit completion. Values: 'standard_cdd', 'enhanced_edd'."
                },
                "documents_submitted": {
                    "type": "array",
                    "description": "List of JSON objects enumerating all uploaded documentation files with doc_type classifications ('drivers_license', 'passport', 'utility_bill', 'articles_of_incorporation', 'operating_agreement', 'business_license', 'ein_confirmation_letter', 'partnership_agreement'), file_path locations, and upload_timestamp ISO 8601 datetimes. Compliance officer reviews for completeness and authenticity."
                },
                "beneficial_owners": {
                    "type": "array",
                    "description": "List of JSON objects containing beneficial owner identification per 31 CFR 1010.230 for corporate applications. Each object includes name, ownership_percentage, ssn, and dob. Compliance officer validates all individuals meeting 25% threshold identified, personal ID documentation collected, and ownership structure complete/accurate rather than nominee arrangements. Empty array for individual applications."
                },
                "compliance_officer_id": {
                    "type": "string",
                    "description": "Identifier of compliance officer assigned to manual review case based on workload balancing algorithms and specialized expertise matching. Officer with lowest pending case count assigned, or routing based on application complexity (corporate to commercial banking experts, high-net-worth to private banking specialists). Format: COMP_OFF_XXX."
                }
                },
                "required": [
                "application_id",
                "auto_decision_result",
                "manual_review_required",
                "manual_review_reason",
                "watchlist_screening_result",
                "fraud_risk_score",
                "chexsystems_score",
                "due_diligence_level",
                "documents_submitted",
                "beneficial_owners",
                "compliance_officer_id"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CreateBankAccount",
            "description": "Generates new account in Core Banking System of record including account_number_generated assignment conforming to institutional numbering schema (may incorporate branch_code prefix, product type identifier, check digit validation), routing_number association representing financial institution's ABA routing transit number (011401533), product configuration for account_type_requested fee schedules and interest accrual rates, and initial account balance establishment at zero pending funding completion. Formats validated customer data per Core Banking System ingestion specifications (field length truncations, character set conversions, standardized state/street abbreviations). Transmits through secure API integration or batch file processing. Establishes formal contractual banking relationship system of record. Records account_creation_timestamp for process efficiency metrics. Only executes when final_approval_status='approved' from ExecuteAutoDecision or ProcessManualReview.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier linking Core Banking System account record to originating application for audit trail and cross-system tracking."
                },
                "final_approval_status": {
                    "type": "string",
                    "description": "Terminal adjudication outcome from ExecuteAutoDecision or ProcessManualReview API. Must equal 'approved' for account creation to proceed. 'declined' or 'conditional_approval' prevent this API execution. Prerequisite validation requirement. Values: 'approved', 'declined', 'conditional_approval'."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name for Core Banking System customer record creation and account ownership establishment. May require field length truncation per core system specifications."
                },
                "applicant_date_of_birth": {
                    "type": "string",
                    "description": "Date of birth in ISO 8601 format for Core Banking System customer profile and regulatory reporting requirements."
                },
                "applicant_ssn_tax_id": {
                    "type": "string",
                    "description": "Social Security Number or EIN for Core Banking System customer identification, tax reporting (1099-INT forms), and regulatory compliance. Encrypted storage in core system."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete physical address for Core Banking System customer record and statement mailing. May require standardization for state abbreviations (CA not California) and street type designations (St not Street) per core system formatting."
                },
                "applicant_email": {
                    "type": "string",
                    "description": "Email address for Core Banking System electronic communication preferences and online banking association."
                },
                "applicant_phone": {
                    "type": "string",
                    "description": "Primary contact telephone for Core Banking System customer profile and notification delivery."
                },
                "account_type_requested": {
                    "type": "string",
                    "description": "Deposit product variant from institutional catalog determining Core Banking System product configuration, fee schedule parameters, interest accrual rate initialization, and minimum balance requirements. Values: 'checking', 'savings', 'student_checking', 'high_yield_savings', 'money_market', 'business_checking'."
                },
                "customer_type": {
                    "type": "string",
                    "description": "Account relationship classification determining Core Banking System account ownership structure and tax reporting form selection (1040 Schedule B for individuals, 1099-INT for businesses). Validates account_type_requested compatibility (business_checking requires non-individual customer_type). Values: 'individual', 'sole_proprietorship', 'llc', 'corporation', 'partnership'."
                },
                "branch_code": {
                    "type": "string",
                    "description": "Physical branch location identifier for account origination attribution and potential account_number_generated prefix incorporation per institutional numbering schema. Format: BR_XXX_NNN. Used for performance tracking and relationship manager assignment."
                },
                "business_legal_name": {
                    "type": "string",
                    "description": "Legal business name as registered with state authorities for corporate account records in Core Banking System. Required for non-individual customer_type values. Empty string for individual applications."
                },
                "business_ein": {
                    "type": "string",
                    "description": "Employer Identification Number for corporate tax reporting and Core Banking System business entity identification. Required for non-individual customer_type values. Empty string for individual applications. Format: XX-XXXXXXX."
                },
                "authorized_signers": {
                    "type": "array",
                    "description": "List of JSON objects identifying individuals empowered to conduct account transactions for business accounts. Each object contains name, title, and signature_captured boolean. Core Banking System associates transaction authority with these individuals. Minimum one required for business accounts. Empty array for individual applications."
                }
                },
                "required": [
                "application_id",
                "final_approval_status",
                "applicant_full_name",
                "applicant_date_of_birth",
                "applicant_ssn_tax_id",
                "applicant_residential_address",
                "applicant_email",
                "applicant_phone",
                "account_type_requested",
                "customer_type",
                "branch_code",
                "business_legal_name",
                "business_ein",
                "authorized_signers"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ProcessAccountFunding",
            "description": "Executes initial deposit processing based on initial_deposit_method with distinct pathways: (1) 'cash'/'check' interfaces with branch teller platforms creating pending deposit transaction awaiting physical instrument presentment, account_funding_completed=false until teller posting confirmation, (2) 'external_transfer' with external_bank_account_linked=true invokes Plaid API to initiate ACH pull transaction from verified external account, monitoring over 1-3 business day ACH settlement timeline, (3) 'debit_card' executes immediate card-not-present authorization/settlement enabling near-instantaneous funds availability. Updates account_activation_status: 'pending_funding' when account_number_generated exists but funding incomplete, 'active' when funding successfully completed enabling full functionality, 'dormant' when funding not completed within institutional timelines (7 business days in-branch, 3 days digital) triggering automated reminders. Records funding_transaction_id for audit trail.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier linking funding transaction to originating application for tracking and reconciliation."
                },
                "account_number_generated": {
                    "type": "string",
                    "description": "Core Banking System assigned account number from CreateBankAccount API. Prerequisite for funding processing - identifies target account for deposit posting. Format per institutional numbering schema (typically 10 digits)."
                },
                "initial_deposit_amount": {
                    "type": "number",
                    "description": "Monetary value in floating-point format representing funding commitment from application submission. Must meet minimum thresholds: $25.00 standard checking, $100.00 savings, $500.00 business checking per institutional policy. Used for transaction authorization amount or ACH transfer value."
                },
                "initial_deposit_method": {
                    "type": "string",
                    "description": "Funding mechanism classification determining processing pathway. 'cash' or 'check' create pending teller transaction requiring physical presentment within 7 days in-branch or 3 days digital before dormant status. 'external_transfer' initiates Plaid API ACH pull with 1-3 day settlement. 'debit_card' executes immediate card-not-present authorization. Values: 'cash', 'check', 'external_transfer', 'debit_card'."
                },
                "external_bank_account_linked": {
                    "type": "boolean",
                    "description": "Boolean indicating successful Plaid API or account aggregation service connection to external financial institution for funding. Must be true when initial_deposit_method='external_transfer' to proceed with ACH pull transaction initiation. False for other funding methods."
                },
                "account_activation_status": {
                    "type": "string",
                    "description": "Current operational state from CreateBankAccount API. Must equal 'pending_funding' at input. Updated to 'active' upon successful funding completion or 'dormant' if funding timeout occurs. Prerequisite validation. Values: 'pending_funding', 'active', 'dormant', 'closed'."
                }
                },
                "required": [
                "application_id",
                "account_number_generated",
                "initial_deposit_amount",
                "initial_deposit_method",
                "external_bank_account_linked",
                "account_activation_status"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "ProvisionAncillaryProducts",
            "description": "Coordinates issuance of ancillary banking products and services based on cross_sell_products_accepted selections: (1) Debit card issuance submits manufacturing orders to card processor including account_number_generated for embossing/magnetic stripe encoding, customer name/address for postal delivery, card design template based on account type (standard/premium metallic for high-balance), typical 7-10 business day delivery timeline, returns masked debit_card_number_issued (format: ****-****-****-NNNN), (2) Online banking creates unique username/temporary password transmitted securely to applicant_email, requires password reset on first use with MFA enrollment, sets online_banking_credentials_created=true, (3) Check stock orders determine quantity (100 personal, 200 business), incorporates account_number_generated/routing_number in MICR encoding, prints customer name/address, processes through third-party vendors with 10-15 day delivery, issues temporary_checks_issued counter/starter checks for immediate needs, (4) Welcome kit assembles all onboarding materials for delivery confirmation.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for tracking ancillary product provisioning process and associating issued products with account opening record."
                },
                "account_number_generated": {
                    "type": "string",
                    "description": "Core Banking System assigned account number from CreateBankAccount API. Prerequisite for product provisioning - used for debit card magnetic stripe encoding, check stock MICR line printing, and online banking account association."
                },
                "routing_number": {
                    "type": "string",
                    "description": "Financial institution's ABA routing transit number from CreateBankAccount API. Required for check stock MICR line encoding to enable check clearing and ACH routing. Format: 9-digit numeric (e.g., 011401533)."
                },
                "applicant_full_name": {
                    "type": "string",
                    "description": "Complete legal name for debit card embossing (may truncate for space constraints), check stock printing in upper left corner, and online banking account display name. Subject to character length restrictions per card processor specifications."
                },
                "applicant_residential_address": {
                    "type": "string",
                    "description": "Complete physical address for debit card postal delivery (7-10 business days), check stock mailing address printing, and welcome kit physical shipment destination."
                },
                "applicant_email": {
                    "type": "string",
                    "description": "Email address for online banking credential secure transmission including generated username and temporary password requiring first-use reset. Electronic welcome kit delivery for digital applications."
                },
                "account_type_requested": {
                    "type": "string",
                    "description": "Deposit product variant determining ancillary product specifications: debit card design template (standard for basic accounts, premium metallic for high-balance), check quantity (100 personal, 200 business), and welcome kit contents. Values: 'checking', 'savings', 'student_checking', 'high_yield_savings', 'money_market', 'business_checking'."
                },
                "cross_sell_products_accepted": {
                    "type": "array",
                    "description": "List of ancillary services customer accepted during application. Determines provisioning actions: 'debit_card' triggers card manufacturing order, 'online_banking' creates credentials, 'mobile_app' enables mobile banking activation, 'merchant_services' for business accounts, 'wire_transfer_services' for corporate accounts. Only accepted products provisioned. Values: ['debit_card', 'online_banking', 'mobile_app', 'merchant_services', 'wire_transfer_services']."
                },
                "customer_type": {
                    "type": "string",
                    "description": "Account relationship classification determining check stock quantity (100 for individual, 200 for business), card design selection, and welcome kit package type (personal vs business materials). Values: 'individual', 'sole_proprietorship', 'llc', 'corporation', 'partnership'."
                }
                },
                "required": [
                "application_id",
                "account_number_generated",
                "routing_number",
                "applicant_full_name",
                "applicant_residential_address",
                "applicant_email",
                "account_type_requested",
                "cross_sell_products_accepted",
                "customer_type"
                ]
            }
            }
        },
        "toolSpec": {
            "name": "CalculateProcessMetrics",
            "description": "Computes process performance metrics for operational monitoring and continuous improvement: (1) total_processing_time_minutes calculates integer difference between application_start_timestamp and account_creation_timestamp for service level agreement compliance monitoring (targets: 45 min in-branch, 15 min digital auto-approval, 4 hours manual review), (2) first_time_yield boolean evaluates whether rework_count=0 indicating quality processing without corrections/supplemental documentation requests, aggregates to institutional FTY% = (count first_time_yield=true / total applications) * 100 targeting ≥95%, (3) rework_count integer quantifies process inefficiency costs from duplicative effort/extended cycles, (4) welcome_kit_issued validates customer_consent_forms_signed completeness including all regulatory disclosures delivered, (5) application_abandoned tracks abandonment_stage for UX friction point identification (ID capture, liveness detection, external account linking, funding stages). Finalizes all completion status fields for operational reporting and regulatory examination preparation.",
            "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                "application_id": {
                    "type": "string",
                    "description": "Unique application identifier for recording calculated process metrics and associating performance data with application record for aggregated analytics."
                },
                "application_start_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 datetime with UTC timezone from application initiation recording precise commencement moment. Used as subtraction minuend for total_processing_time_minutes calculation. Format: YYYY-MM-DDTHH:MM:SSZ."
                },
                "account_creation_timestamp": {
                    "type": "string",
                    "description": "ISO 8601 datetime with UTC timezone from CreateBankAccount API recording Core Banking System account establishment moment. Used as subtraction subtrahend for total_processing_time_minutes calculation. Missing value uses current timestamp as fallback with data quality flag. Format: YYYY-MM-DDTHH:MM:SSZ."
                },
                "account_funding_completed": {
                    "type": "boolean",
                    "description": "Boolean from ProcessAccountFunding API indicating successful initial deposit completion. True enables welcome_kit_issued=true when combined with complete customer_consent_forms_signed. False maintains account_activation_status='pending_funding' and welcome_kit_issued=false."
                },
                "final_approval_status": {
                    "type": "string",
                    "description": "Terminal adjudication outcome from ExecuteAutoDecision or ProcessManualReview determining whether account creation occurred for metrics calculation. 'approved' with account_creation_timestamp enables full metrics computation. 'declined' results in partial metrics for abandonment analysis. Values: 'approved', 'declined', 'conditional_approval'."
                },
                "auto_decision_result": {
                    "type": "string",
                    "description": "Auto-decision outcome from ExecuteAutoDecision API indicating whether straight-through processing achieved or manual intervention required. 'auto_approved'/'auto_declined' without manual_review_required=true indicates optimal processing efficiency. Used for auto-approval rate calculation. Values: 'auto_approved', 'auto_declined', 'refer_for_manual_review'."
                },
                "manual_review_required": {
                    "type": "boolean",
                    "description": "Boolean indicating whether compliance officer adjudication occurred. True adds manual review time component to total_processing_time_minutes and typically indicates rework_count≥1 affecting first_time_yield=false unless single-pass approval achieved during review."
                },
                "customer_consent_forms_signed": {
                    "type": "array",
                    "description": "List of regulatory disclosure acknowledgments from application process. Complete list required for welcome_kit_issued=true: signature_card, terms_and_conditions, truth_in_savings_disclosure, privacy_notice, business_account_agreement (business accounts), electronic_communications_consent (digital channels). Incomplete triggers welcome_kit_issued=false and remediation flag."
                },
                "application_abandoned": {
                    "type": "boolean",
                    "description": "Boolean indicating whether customer discontinued self-service workflow before completion. True when customer initiated but did not complete within institutional timeout (7 days from application_start_timestamp). Used for abandonment rate calculation = (count application_abandoned=true / total initiated) * 100."
                },
                "abandonment_stage": {
                    "type": "string",
                    "description": "Specific workflow step where customer discontinued digital application for UX friction point analysis. Common values: 'id_document_capture' (lighting/focus issues), 'liveness_detection' (uncomfortable with facial movements/technical difficulties), 'external_bank_account_linking' (credential reluctance/connection failures), 'funding' (deposit amount reconsideration). Empty string for completed/in-branch applications. Aggregated for product management prioritization."
                }
                },
                "required": [
                "application_id",
                "application_start_timestamp",
                "account_creation_timestamp",
                "account_funding_completed",
                "final_approval_status",
                "auto_decision_result",
                "manual_review_required",
                "customer_consent_forms_signed",
                "application_abandoned",
                "abandonment_stage"
                ]
            }
            }
        }
        }
        </SCHEMA>'''
    sample_data = '''
        <data>
        ['application_id:APP_20250418101234567', 'application_channel:in_branch', 'business_type:retail_individual', 'application_start_timestamp:2025-04-18T10:12:34Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Sarah Michelle Johnson', 'applicant_date_of_birth:1990-03-15', 'applicant_ssn_tax_id:XXX-XX-4521', 'applicant_email:sarah.johnson@email.com', 'applicant_phone:+1-555-234-5678', 'applicant_residential_address:456 Oak Avenue, Apt 2C, Boston, MA 02115', 'applicant_employment_status:employed', 'applicant_annual_income:75000', 'id_document_type:drivers_license', 'id_document_number:DL987654321', 'id_document_expiry_date:2028-03-15', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:500.00', 'initial_deposit_method:cash', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418101234567_dl.pdf","upload_timestamp":"2025-04-18T10:15:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418101234567_utility.pdf","upload_timestamp":"2025-04-18T10:16:00Z"}]', 'banker_id:EMP_5421', 'branch_code:BR_BOS_003', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","truth_in_savings_disclosure","privacy_notice"]', 'esignature_timestamp:2025-04-18T10:25:30Z', 'esignature_ip_address:192.168.10.45', 'marketing_opt_in:True', 'referral_source:branch_walk_in', 'product_bundle_offered:standard_checking', 'cross_sell_products_accepted:["debit_card","online_banking","mobile_app"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418143045123', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T14:30:45Z', 'customer_type:individual', 'account_type_requested:high_yield_savings', 'applicant_full_name:Michael Robert Chen', 'applicant_date_of_birth:1988-07-22', 'applicant_ssn_tax_id:XXX-XX-7890', 'applicant_email:michael.chen@gmail.com', 'applicant_phone:+1-555-876-5432', 'applicant_residential_address:789 Pine Street, Unit 15B, San Francisco, CA 94102', 'applicant_employment_status:employed', 'applicant_annual_income:125000', 'id_document_type:passport', 'id_document_number:P123456789', 'id_document_expiry_date:2030-07-22', 'id_document_issuing_country:US', 'proof_of_address_type:bank_statement', 'initial_deposit_amount:5000.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Michael Robert Chen","dob":"1988-07-22","id_number":"P123456789","address":"789 Pine Street, San Francisco, CA"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:96.5', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"passport","file_path":"/docs/APP_20250418143045123_passport.pdf","upload_timestamp":"2025-04-18T14:32:00Z"},{"doc_type":"bank_statement","file_path":"/docs/APP_20250418143045123_stmt.pdf","upload_timestamp":"2025-04-18T14:33:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T14:38:15Z', 'esignature_ip_address:24.156.78.92', 'marketing_opt_in:False', 'referral_source:online_search', 'product_bundle_offered:premium_savings_with_rewards', 'cross_sell_products_accepted:["online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418091523789', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T09:15:23Z', 'customer_type:individual', 'account_type_requested:student_checking', 'applicant_full_name:Emily Rose Martinez', 'applicant_date_of_birth:2002-11-08', 'applicant_ssn_tax_id:XXX-XX-3210', 'applicant_email:emily.martinez@university.edu', 'applicant_phone:+1-555-345-6789', 'applicant_residential_address:123 College Drive, Dorm Hall A, Austin, TX 78705', 'applicant_employment_status:student', 'applicant_annual_income:15000', 'id_document_type:state_id', 'id_document_number:TX12345678', 'id_document_expiry_date:2027-11-08', 'id_document_issuing_country:US', 'proof_of_address_type:lease_agreement', 'initial_deposit_amount:100.00', 'initial_deposit_method:debit_card', 'external_bank_account_linked:False', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Emily Rose Martinez","dob":"2002-11-08","id_number":"TX12345678","address":"123 College Drive, Austin, TX"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:94.2', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"state_id","file_path":"/docs/APP_20250418091523789_id.pdf","upload_timestamp":"2025-04-18T09:17:00Z"},{"doc_type":"lease_agreement","file_path":"/docs/APP_20250418091523789_lease.pdf","upload_timestamp":"2025-04-18T09:18:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T09:22:45Z', 'esignature_ip_address:136.244.11.56', 'marketing_opt_in:True', 'referral_source:friend_referral', 'product_bundle_offered:student_banking_package', 'cross_sell_products_accepted:["debit_card","online_banking","mobile_app"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418155612456', 'application_channel:in_branch', 'business_type:corporate_business', 'application_start_timestamp:2025-04-18T15:56:12Z', 'customer_type:llc', 'account_type_requested:business_checking', 'applicant_full_name:David Alexander Thompson', 'applicant_date_of_birth:1975-04-30', 'applicant_ssn_tax_id:XXX-XX-6543', 'applicant_email:david.thompson@techsolutions.com', 'applicant_phone:+1-555-987-6543', 'applicant_residential_address:567 Business Plaza, Suite 200, Seattle, WA 98101', 'applicant_employment_status:self_employed', 'applicant_annual_income:250000', 'id_document_type:drivers_license', 'id_document_number:WA987654321', 'id_document_expiry_date:2029-04-30', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:5000.00', 'initial_deposit_method:check', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:Tech Solutions LLC', 'business_ein:12-3456789', 'business_structure:llc', 'business_industry:technology_services', 'business_registration_date:2020-03-15', 'business_monthly_transaction_volume_estimate:150000', 'beneficial_owners:[{"name":"David Alexander Thompson","ownership_percentage":60,"ssn":"XXX-XX-6543","dob":"1975-04-30"},{"name":"Jennifer Marie Thompson","ownership_percentage":40,"ssn":"XXX-XX-7654","dob":"1978-08-12"}]', 'authorized_signers:[{"name":"David Alexander Thompson","title":"CEO","signature_captured":true},{"name":"Jennifer Marie Thompson","title":"CFO","signature_captured":true}]', 'documents_submitted:[{"doc_type":"articles_of_incorporation","file_path":"/docs/APP_20250418155612456_articles.pdf","upload_timestamp":"2025-04-18T16:00:00Z"},{"doc_type":"operating_agreement","file_path":"/docs/APP_20250418155612456_operating.pdf","upload_timestamp":"2025-04-18T16:01:00Z"},{"doc_type":"ein_confirmation_letter","file_path":"/docs/APP_20250418155612456_ein.pdf","upload_timestamp":"2025-04-18T16:02:00Z"},{"doc_type":"business_license","file_path":"/docs/APP_20250418155612456_license.pdf","upload_timestamp":"2025-04-18T16:03:00Z"}]', 'banker_id:EMP_7823', 'branch_code:BR_SEA_007', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","business_account_agreement","privacy_notice"]', 'esignature_timestamp:2025-04-18T16:15:30Z', 'esignature_ip_address:192.168.25.78', 'marketing_opt_in:False', 'referral_source:advertisement', 'product_bundle_offered:business_advantage_package', 'cross_sell_products_accepted:["business_debit_card","online_banking","merchant_services"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418112345678', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T11:23:45Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Jennifer Lynn Anderson', 'applicant_date_of_birth:1992-09-18', 'applicant_ssn_tax_id:XXX-XX-2109', 'applicant_email:jennifer.anderson@yahoo.com', 'applicant_phone:+1-555-432-1098', 'applicant_residential_address:890 Maple Lane, Chicago, IL 60614', 'applicant_employment_status:employed', 'applicant_annual_income:55000', 'id_document_type:drivers_license', 'id_document_number:IL234567890', 'id_document_expiry_date:2026-09-18', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:250.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Jennifer Lynn Anderson","dob":"1992-09-18","id_number":"IL234567890","address":"890 Maple Lane, Chicago, IL"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:89.5', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418112345678_dl.pdf","upload_timestamp":"2025-04-18T11:25:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418112345678_utility.pdf","upload_timestamp":"2025-04-18T11:26:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T11:30:20Z', 'esignature_ip_address:72.134.89.45', 'marketing_opt_in:True', 'referral_source:online_search', 'product_bundle_offered:standard_checking', 'cross_sell_products_accepted:["debit_card","online_banking"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418134512890', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T13:45:12Z', 'customer_type:individual', 'account_type_requested:money_market', 'applicant_full_name:Robert James Wilson', 'applicant_date_of_birth:1965-12-05', 'applicant_ssn_tax_id:XXX-XX-8901', 'applicant_email:robert.wilson@hotmail.com', 'applicant_phone:+1-555-765-4321', 'applicant_residential_address:234 Elm Street, Phoenix, AZ 85001', 'applicant_employment_status:retired', 'applicant_annual_income:45000', 'id_document_type:drivers_license', 'id_document_number:AZ345678901', 'id_document_expiry_date:2025-12-05', 'id_document_issuing_country:US', 'proof_of_address_type:bank_statement', 'initial_deposit_amount:10000.00', 'initial_deposit_method:check', 'external_bank_account_linked:False', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Robert James Wilson","dob":"1965-12-05","id_number":"AZ345678901","address":"234 Elm Street, Phoenix, AZ"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:93.8', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418134512890_dl.pdf","upload_timestamp":"2025-04-18T13:47:00Z"},{"doc_type":"bank_statement","file_path":"/docs/APP_20250418134512890_stmt.pdf","upload_timestamp":"2025-04-18T13:48:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice"]', 'esignature_timestamp:2025-04-18T13:52:35Z', 'esignature_ip_address:198.45.67.123', 'marketing_opt_in:False', 'referral_source:friend_referral', 'product_bundle_offered:premium_money_market', 'cross_sell_products_accepted:["online_banking"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418162312345', 'application_channel:in_branch', 'business_type:retail_individual', 'application_start_timestamp:2025-04-18T16:23:12Z', 'customer_type:individual', 'account_type_requested:savings', 'applicant_full_name:Maria Elena Rodriguez', 'applicant_date_of_birth:1995-02-28', 'applicant_ssn_tax_id:XXX-XX-4567', 'applicant_email:maria.rodriguez@outlook.com', 'applicant_phone:+1-555-234-8765', 'applicant_residential_address:678 Cedar Avenue, Miami, FL 33101', 'applicant_employment_status:employed', 'applicant_annual_income:62000', 'id_document_type:passport', 'id_document_number:P987654321', 'id_document_expiry_date:2031-02-28', 'id_document_issuing_country:US', 'proof_of_address_type:lease_agreement', 'initial_deposit_amount:1000.00', 'initial_deposit_method:cash', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"passport","file_path":"/docs/APP_20250418162312345_passport.pdf","upload_timestamp":"2025-04-18T16:25:00Z"},{"doc_type":"lease_agreement","file_path":"/docs/APP_20250418162312345_lease.pdf","upload_timestamp":"2025-04-18T16:26:00Z"}]', 'banker_id:EMP_3456', 'branch_code:BR_MIA_012', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","truth_in_savings_disclosure","privacy_notice"]', 'esignature_timestamp:2025-04-18T16:35:45Z', 'esignature_ip_address:192.168.15.89', 'marketing_opt_in:True', 'referral_source:branch_walk_in', 'product_bundle_offered:standard_savings', 'cross_sell_products_accepted:["debit_card","online_banking"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418084523678', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T08:45:23Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Christopher Paul Davis', 'applicant_date_of_birth:1987-06-10', 'applicant_ssn_tax_id:XXX-XX-5678', 'applicant_email:chris.davis@icloud.com', 'applicant_phone:+1-555-876-2345', 'applicant_residential_address:345 Birch Road, Denver, CO 80202', 'applicant_employment_status:self_employed', 'applicant_annual_income:95000', 'id_document_type:drivers_license', 'id_document_number:CO456789012', 'id_document_expiry_date:2027-06-10', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:750.00', 'initial_deposit_method:debit_card', 'external_bank_account_linked:False', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Christopher Paul Davis","dob":"1987-06-10","id_number":"CO456789012","address":"345 Birch Road, Denver, CO"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:97.2', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418084523678_dl.pdf","upload_timestamp":"2025-04-18T08:47:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418084523678_utility.pdf","upload_timestamp":"2025-04-18T08:48:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T08:52:10Z', 'esignature_ip_address:45.123.78.234', 'marketing_opt_in:False', 'referral_source:advertisement', 'product_bundle_offered:premium_checking_with_rewards', 'cross_sell_products_accepted:["debit_card","online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418172345901', 'application_channel:in_branch', 'business_type:corporate_business', 'application_start_timestamp:2025-04-18T17:23:45Z', 'customer_type:corporation', 'account_type_requested:business_checking', 'applicant_full_name:Patricia Anne Smith', 'applicant_date_of_birth:1980-10-15', 'applicant_ssn_tax_id:XXX-XX-9012', 'applicant_email:patricia.smith@globaltrading.com', 'applicant_phone:+1-555-345-9876', 'applicant_residential_address:901 Commerce Boulevard, Suite 500, New York, NY 10001', 'applicant_employment_status:employed', 'applicant_annual_income:180000', 'id_document_type:passport', 'id_document_number:P246813579', 'id_document_expiry_date:2029-10-15', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:25000.00', 'initial_deposit_method:check', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:Global Trading Corporation', 'business_ein:98-7654321', 'business_structure:corporation', 'business_industry:international_trade', 'business_registration_date:2015-06-20', 'business_monthly_transaction_volume_estimate:2500000', 'beneficial_owners:[{"name":"Patricia Anne Smith","ownership_percentage":35,"ssn":"XXX-XX-9012","dob":"1980-10-15"},{"name":"Marcus William Johnson","ownership_percentage":30,"ssn":"XXX-XX-3456","dob":"1972-03-25"},{"name":"Sophia Grace Lee","ownership_percentage":35,"ssn":"XXX-XX-7890","dob":"1985-07-18"}]', 'authorized_signers:[{"name":"Patricia Anne Smith","title":"President","signature_captured":true},{"name":"Marcus William Johnson","title":"Vice President","signature_captured":true}]', 'documents_submitted:[{"doc_type":"articles_of_incorporation","file_path":"/docs/APP_20250418172345901_articles.pdf","upload_timestamp":"2025-04-18T17:30:00Z"},{"doc_type":"business_license","file_path":"/docs/APP_20250418172345901_license.pdf","upload_timestamp":"2025-04-18T17:31:00Z"},{"doc_type":"ein_confirmation_letter","file_path":"/docs/APP_20250418172345901_ein.pdf","upload_timestamp":"2025-04-18T17:32:00Z"}]', 'banker_id:EMP_9012', 'branch_code:BR_NYC_001', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","business_account_agreement","privacy_notice"]', 'esignature_timestamp:2025-04-18T17:50:20Z', 'esignature_ip_address:192.168.40.67', 'marketing_opt_in:False', 'referral_source:branch_walk_in', 'product_bundle_offered:corporate_banking_suite', 'cross_sell_products_accepted:["business_debit_card","online_banking","wire_transfer_services","merchant_services"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418105634812', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T10:56:34Z', 'customer_type:individual', 'account_type_requested:student_checking', 'applicant_full_name:Daniel Kevin Brown', 'applicant_date_of_birth:2003-01-20', 'applicant_ssn_tax_id:XXX-XX-1234', 'applicant_email:daniel.brown@college.edu', 'applicant_phone:+1-555-567-8901', 'applicant_residential_address:456 University Way, Student Housing B, Ann Arbor, MI 48104', 'applicant_employment_status:student', 'applicant_annual_income:8000', 'id_document_type:state_id', 'id_document_number:MI567890123', 'id_document_expiry_date:2028-01-20', 'id_document_issuing_country:US', 'proof_of_address_type:lease_agreement', 'initial_deposit_amount:50.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Daniel Kevin Brown","dob":"2003-01-20","id_number":"MI567890123","address":"456 University Way, Ann Arbor, MI"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:95.7', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"state_id","file_path":"/docs/APP_20250418105634812_id.pdf","upload_timestamp":"2025-04-18T10:58:00Z"},{"doc_type":"lease_agreement","file_path":"/docs/APP_20250418105634812_lease.pdf","upload_timestamp":"2025-04-18T10:59:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T11:03:25Z', 'esignature_ip_address:156.78.234.90', 'marketing_opt_in:True', 'referral_source:online_search', 'product_bundle_offered:student_banking_package', 'cross_sell_products_accepted:["debit_card","mobile_app"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418140123567', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T14:01:23Z', 'customer_type:individual', 'account_type_requested:high_yield_savings', 'applicant_full_name:Amanda Grace Taylor', 'applicant_date_of_birth:1993-05-12', 'applicant_ssn_tax_id:XXX-XX-6789', 'applicant_email:amanda.taylor@protonmail.com', 'applicant_phone:+1-555-678-9012', 'applicant_residential_address:789 Willow Street, Portland, OR 97201', 'applicant_employment_status:employed', 'applicant_annual_income:82000', 'id_document_type:drivers_license', 'id_document_number:OR678901234', 'id_document_expiry_date:2029-05-12', 'id_document_issuing_country:US', 'proof_of_address_type:bank_statement', 'initial_deposit_amount:3500.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Amanda Grace Taylor","dob":"1993-05-11","id_number":"OR678901234","address":"789 Willow Street, Portland, OR"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:91.2', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418140123567_dl.pdf","upload_timestamp":"2025-04-18T14:03:00Z"},{"doc_type":"bank_statement","file_path":"/docs/APP_20250418140123567_stmt.pdf","upload_timestamp":"2025-04-18T14:04:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T14:08:40Z', 'esignature_ip_address:67.234.123.89', 'marketing_opt_in:False', 'referral_source:friend_referral', 'product_bundle_offered:premium_savings_with_rewards', 'cross_sell_products_accepted:["online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418093412789', 'application_channel:in_branch', 'business_type:corporate_business', 'application_start_timestamp:2025-04-18T09:34:12Z', 'customer_type:sole_proprietorship', 'account_type_requested:business_checking', 'applicant_full_name:Thomas Richard Moore', 'applicant_date_of_birth:1970-08-25', 'applicant_ssn_tax_id:XXX-XX-2345', 'applicant_email:thomas.moore@mooreconsulting.com', 'applicant_phone:+1-555-789-0123', 'applicant_residential_address:123 Professional Drive, Atlanta, GA 30303', 'applicant_employment_status:self_employed', 'applicant_annual_income:110000', 'id_document_type:drivers_license', 'id_document_number:GA789012345', 'id_document_expiry_date:2027-08-25', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:2000.00', 'initial_deposit_method:check', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:Moore Consulting Services', 'business_ein:45-6789012', 'business_structure:sole_proprietorship', 'business_industry:professional_services', 'business_registration_date:2018-04-10', 'business_monthly_transaction_volume_estimate:35000', 'beneficial_owners:[]', 'authorized_signers:[{"name":"Thomas Richard Moore","title":"Owner","signature_captured":true}]', 'documents_submitted:[{"doc_type":"business_license","file_path":"/docs/APP_20250418093412789_license.pdf","upload_timestamp":"2025-04-18T09:38:00Z"},{"doc_type":"ein_confirmation_letter","file_path":"/docs/APP_20250418093412789_ein.pdf","upload_timestamp":"2025-04-18T09:39:00Z"}]', 'banker_id:EMP_6789', 'branch_code:BR_ATL_005', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","business_account_agreement","privacy_notice"]', 'esignature_timestamp:2025-04-18T09:48:15Z', 'esignature_ip_address:192.168.30.56', 'marketing_opt_in:True', 'referral_source:advertisement', 'product_bundle_offered:small_business_package', 'cross_sell_products_accepted:["business_debit_card","online_banking"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418151234890', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T15:12:34Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Jessica Marie White', 'applicant_date_of_birth:1991-11-30', 'applicant_ssn_tax_id:XXX-XX-3456', 'applicant_email:jessica.white@gmail.com', 'applicant_phone:+1-555-890-1234', 'applicant_residential_address:567 Oak Street, Dallas, TX 75201', 'applicant_employment_status:employed', 'applicant_annual_income:68000', 'id_document_type:drivers_license', 'id_document_number:TX890123456', 'id_document_expiry_date:2026-11-30', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:300.00', 'initial_deposit_method:debit_card', 'external_bank_account_linked:False', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Jessica Marie White","dob":"1991-11-30","id_number":"TX890123456","address":"567 Oak Street, Dallas, TX"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:98.1', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418151234890_dl.pdf","upload_timestamp":"2025-04-18T15:14:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418151234890_utility.pdf","upload_timestamp":"2025-04-18T15:15:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T15:18:50Z', 'esignature_ip_address:89.123.67.234', 'marketing_opt_in:True', 'referral_source:online_search', 'product_bundle_offered:standard_checking', 'cross_sell_products_accepted:["debit_card","online_banking","mobile_app"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418124523901', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T12:45:23Z', 'customer_type:individual', 'account_type_requested:money_market', 'applicant_full_name:Andrew Scott Harris', 'applicant_date_of_birth:1982-04-08', 'applicant_ssn_tax_id:XXX-XX-4567', 'applicant_email:andrew.harris@corporate.net', 'applicant_phone:+1-555-901-2345', 'applicant_residential_address:890 Financial Plaza, Philadelphia, PA 19102', 'applicant_employment_status:employed', 'applicant_annual_income:145000', 'id_document_type:passport', 'id_document_number:P135792468', 'id_document_expiry_date:2032-04-08', 'id_document_issuing_country:US', 'proof_of_address_type:bank_statement', 'initial_deposit_amount:15000.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Andrew Scott Harris","dob":"1982-04-08","id_number":"P135792468","address":"890 Financial Plaza, Philadelphia, PA"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:94.8', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"passport","file_path":"/docs/APP_20250418124523901_passport.pdf","upload_timestamp":"2025-04-18T12:47:00Z"},{"doc_type":"bank_statement","file_path":"/docs/APP_20250418124523901_stmt.pdf","upload_timestamp":"2025-04-18T12:48:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T12:53:20Z', 'esignature_ip_address:123.234.89.45', 'marketing_opt_in:False', 'referral_source:friend_referral', 'product_bundle_offered:premium_money_market', 'cross_sell_products_accepted:["online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418183412567', 'application_channel:in_branch', 'business_type:corporate_business', 'application_start_timestamp:2025-04-18T18:34:12Z', 'customer_type:partnership', 'account_type_requested:business_checking', 'applicant_full_name:William George Clark', 'applicant_date_of_birth:1976-01-18', 'applicant_ssn_tax_id:XXX-XX-5678', 'applicant_email:william.clark@lawfirm.com', 'applicant_phone:+1-555-012-3456', 'applicant_residential_address:234 Legal Avenue, Suite 300, Houston, TX 77002', 'applicant_employment_status:self_employed', 'applicant_annual_income:320000', 'id_document_type:drivers_license', 'id_document_number:TX012345678', 'id_document_expiry_date:2028-01-18', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:10000.00', 'initial_deposit_method:check', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:Clark & Associates Law Partnership', 'business_ein:56-7890123', 'business_structure:partnership', 'business_industry:legal_services', 'business_registration_date:2010-09-15', 'business_monthly_transaction_volume_estimate:450000', 'beneficial_owners:[{"name":"William George Clark","ownership_percentage":50,"ssn":"XXX-XX-5678","dob":"1976-01-18"},{"name":"Elizabeth Anne Martin","ownership_percentage":50,"ssn":"XXX-XX-8901","dob":"1974-05-22"}]', 'authorized_signers:[{"name":"William George Clark","title":"Managing Partner","signature_captured":true},{"name":"Elizabeth Anne Martin","title":"Partner","signature_captured":true}]', 'documents_submitted:[{"doc_type":"partnership_agreement","file_path":"/docs/APP_20250418183412567_partnership.pdf","upload_timestamp":"2025-04-18T18:40:00Z"},{"doc_type":"business_license","file_path":"/docs/APP_20250418183412567_license.pdf","upload_timestamp":"2025-04-18T18:41:00Z"},{"doc_type":"ein_confirmation_letter","file_path":"/docs/APP_20250418183412567_ein.pdf","upload_timestamp":"2025-04-18T18:42:00Z"}]', 'banker_id:EMP_2345', 'branch_code:BR_HOU_008', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","business_account_agreement","privacy_notice"]', 'esignature_timestamp:2025-04-18T19:00:35Z', 'esignature_ip_address:192.168.50.123', 'marketing_opt_in:False', 'referral_source:branch_walk_in', 'product_bundle_offered:professional_services_package', 'cross_sell_products_accepted:["business_debit_card","online_banking","wire_transfer_services"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418113445678', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T11:34:45Z', 'customer_type:individual', 'account_type_requested:savings', 'applicant_full_name:Michelle Lynn Turner', 'applicant_date_of_birth:1994-07-14', 'applicant_ssn_tax_id:XXX-XX-6789', 'applicant_email:michelle.turner@outlook.com', 'applicant_phone:+1-555-123-4567', 'applicant_residential_address:678 Sunset Boulevard, Las Vegas, NV 89101', 'applicant_employment_status:employed', 'applicant_annual_income:58000', 'id_document_type:drivers_license', 'id_document_number:NV123456789', 'id_document_expiry_date:2027-07-14', 'id_document_issuing_country:US', 'proof_of_address_type:lease_agreement', 'initial_deposit_amount:800.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Michelle Lynn Turner","dob":"1994-07-14","id_number":"NV123456789","address":"678 Sunset Boulevard, Las Vegas, NV"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:92.3', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418113445678_dl.pdf","upload_timestamp":"2025-04-18T11:36:00Z"},{"doc_type":"lease_agreement","file_path":"/docs/APP_20250418113445678_lease.pdf","upload_timestamp":"2025-04-18T11:37:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T11:41:55Z', 'esignature_ip_address:45.234.156.78', 'marketing_opt_in:True', 'referral_source:advertisement', 'product_bundle_offered:standard_savings', 'cross_sell_products_accepted:["online_banking","mobile_app"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418074512234', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T07:45:12Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Brian Joseph Lee', 'applicant_date_of_birth:1989-12-03', 'applicant_ssn_tax_id:XXX-XX-7890', 'applicant_email:brian.lee@techmail.com', 'applicant_phone:+1-555-234-5678', 'applicant_residential_address:345 Innovation Drive, San Jose, CA 95110', 'applicant_employment_status:employed', 'applicant_annual_income:135000', 'id_document_type:passport', 'id_document_number:P987321654', 'id_document_expiry_date:2030-12-03', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:2000.00', 'initial_deposit_method:debit_card', 'external_bank_account_linked:False', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Brian Joseph Lee","dob":"1989-12-03","id_number":"P987321654","address":"345 Innovation Drive, San Jose, CA"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:97.8', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"passport","file_path":"/docs/APP_20250418074512234_passport.pdf","upload_timestamp":"2025-04-18T07:47:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418074512234_utility.pdf","upload_timestamp":"2025-04-18T07:48:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T07:51:30Z', 'esignature_ip_address:78.234.123.45', 'marketing_opt_in:False', 'referral_source:online_search', 'product_bundle_offered:premium_checking_with_rewards', 'cross_sell_products_accepted:["debit_card","online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418165634345', 'application_channel:in_branch', 'business_type:retail_individual', 'application_start_timestamp:2025-04-18T16:56:34Z', 'customer_type:individual', 'account_type_requested:checking', 'applicant_full_name:Nicole Marie Jackson', 'applicant_date_of_birth:1986-03-27', 'applicant_ssn_tax_id:XXX-XX-8901', 'applicant_email:nicole.jackson@email.com', 'applicant_phone:+1-555-345-6789', 'applicant_residential_address:901 Riverside Drive, Nashville, TN 37201', 'applicant_employment_status:employed', 'applicant_annual_income:72000', 'id_document_type:drivers_license', 'id_document_number:TN345678901', 'id_document_expiry_date:2028-03-27', 'id_document_issuing_country:US', 'proof_of_address_type:bank_statement', 'initial_deposit_amount:400.00', 'initial_deposit_method:cash', 'external_bank_account_linked:False', 'otp_verification_completed:False', 'id_ocr_extracted_data:{}', 'biometric_liveness_check_completed:False', 'facial_match_confidence_score:0.0', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418165634345_dl.pdf","upload_timestamp":"2025-04-18T16:58:00Z"},{"doc_type":"bank_statement","file_path":"/docs/APP_20250418165634345_stmt.pdf","upload_timestamp":"2025-04-18T16:59:00Z"}]', 'banker_id:EMP_4567', 'branch_code:BR_NSH_010', 'customer_consent_forms_signed:["signature_card","terms_and_conditions","truth_in_savings_disclosure","privacy_notice"]', 'esignature_timestamp:2025-04-18T17:08:45Z', 'esignature_ip_address:192.168.35.90', 'marketing_opt_in:True', 'referral_source:branch_walk_in', 'product_bundle_offered:standard_checking', 'cross_sell_products_accepted:["debit_card","online_banking"]', 'application_data_entry_method:manual_banker_entry', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418102345789', 'application_channel:web_portal', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T10:23:45Z', 'customer_type:individual', 'account_type_requested:student_checking', 'applicant_full_name:Ryan Christopher Adams', 'applicant_date_of_birth:2001-09-22', 'applicant_ssn_tax_id:XXX-XX-9012', 'applicant_email:ryan.adams@student.edu', 'applicant_phone:+1-555-456-7890', 'applicant_residential_address:234 Campus Street, Residence Hall C, Madison, WI 53706', 'applicant_employment_status:student', 'applicant_annual_income:12000', 'id_document_type:state_id', 'id_document_number:WI456789012', 'id_document_expiry_date:2027-09-22', 'id_document_issuing_country:US', 'proof_of_address_type:lease_agreement', 'initial_deposit_amount:75.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Ryan Christopher Adams","dob":"2001-09-22","id_number":"WI456789012","address":"234 Campus Street, Madison, WI"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:93.5', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"state_id","file_path":"/docs/APP_20250418102345789_id.pdf","upload_timestamp":"2025-04-18T10:25:00Z"},{"doc_type":"lease_agreement","file_path":"/docs/APP_20250418102345789_lease.pdf","upload_timestamp":"2025-04-18T10:26:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T10:30:15Z', 'esignature_ip_address:234.156.78.90', 'marketing_opt_in:True', 'referral_source:friend_referral', 'product_bundle_offered:student_banking_package', 'cross_sell_products_accepted:["debit_card","mobile_app"]', 'application_data_entry_method:customer_self_service', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        ['application_id:APP_20250418132123456', 'application_channel:mobile_app', 'business_type:digital_remote', 'application_start_timestamp:2025-04-18T13:21:23Z', 'customer_type:individual', 'account_type_requested:high_yield_savings', 'applicant_full_name:Stephanie Louise Roberts', 'applicant_date_of_birth:1990-08-16', 'applicant_ssn_tax_id:XXX-XX-0123', 'applicant_email:stephanie.roberts@business.com', 'applicant_phone:+1-555-567-8901', 'applicant_residential_address:567 Mountain View, Salt Lake City, UT 84101', 'applicant_employment_status:employed', 'applicant_annual_income:98000', 'id_document_type:drivers_license', 'id_document_number:UT567890123', 'id_document_expiry_date:2029-08-16', 'id_document_issuing_country:US', 'proof_of_address_type:utility_bill', 'initial_deposit_amount:6000.00', 'initial_deposit_method:external_transfer', 'external_bank_account_linked:True', 'otp_verification_completed:True', 'id_ocr_extracted_data:{"name":"Stephanie Louise Roberts","dob":"1990-08-16","id_number":"UT567890123","address":"567 Mountain View, Salt Lake City, UT"}', 'biometric_liveness_check_completed:True', 'facial_match_confidence_score:95.3', 'business_legal_name:', 'business_ein:', 'business_structure:', 'business_industry:', 'business_registration_date:', 'business_monthly_transaction_volume_estimate:', 'beneficial_owners:[]', 'authorized_signers:[]', 'documents_submitted:[{"doc_type":"drivers_license","file_path":"/docs/APP_20250418132123456_dl.pdf","upload_timestamp":"2025-04-18T13:23:00Z"},{"doc_type":"utility_bill","file_path":"/docs/APP_20250418132123456_utility.pdf","upload_timestamp":"2025-04-18T13:24:00Z"}]', 'banker_id:', 'branch_code:', 'customer_consent_forms_signed:["terms_and_conditions","truth_in_savings_disclosure","privacy_notice","electronic_communications_consent"]', 'esignature_timestamp:2025-04-18T13:28:40Z', 'esignature_ip_address:156.234.90.12', 'marketing_opt_in:False', 'referral_source:online_search', 'product_bundle_offered:premium_savings_with_rewards', 'cross_sell_products_accepted:["online_banking","mobile_app"]', 'application_data_entry_method:ocr_auto_fill', 'cip_verification_status:', 'watchlist_screening_result:', 'chexsystems_score:', 'credit_bureau_validation_status:', 'fraud_risk_score:', 'auto_decision_result:', 'manual_review_required:', 'manual_review_reason:', 'compliance_officer_id:', 'compliance_review_timestamp:', 'compliance_review_notes:', 'due_diligence_level:', 'final_approval_status:', 'decline_reason:', 'account_number_generated:', 'routing_number:', 'account_creation_timestamp:', 'total_processing_time_minutes:', 'first_time_yield:', 'rework_count:', 'welcome_kit_issued:', 'debit_card_number_issued:', 'online_banking_credentials_created:', 'temporary_checks_issued:', 'account_funding_completed:', 'account_activation_status:', 'application_abandoned:', 'abandonment_stage:'],
        </data>'''
    yaml_file = "/Account/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/SOP_API_Function_Code.yaml"
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
