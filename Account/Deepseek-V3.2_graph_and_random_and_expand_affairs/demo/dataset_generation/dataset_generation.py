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
    sop_title = "Banking Account Opening Business Handling Process"
    sop_file_contents = '''
        <SOP>

        1. Purpose

        This Standard Operating Procedure establishes the comprehensive, risk-based framework governing the end-to-end account opening business process for retail individual, digital remote, and corporate business customer segments within the financial institution's customer onboarding infrastructure, ensuring strict adherence to the Customer Identification Program requirements mandated under 31 CFR 1020.220, the Beneficial Ownership Identification provisions codified in 31 CFR 1010.230, and the watchlist screening obligations imposed by the Office of Foreign Assets Control pursuant to 31 CFR Chapter V, while simultaneously optimizing operational efficiency through algorithmically-driven auto-decisioning engines that stratify applications based on quantifiable risk metrics and route high-complexity cases through Enhanced Due Diligence protocols requiring compliance officer adjudication. The primary objective encompasses the systematic transformation of raw applicant data captured through multiple origination channels into fully validated, regulatory-compliant account relationships recorded within the Core Banking System of Record, thereby establishing the contractual foundation for deposit acceptance, payment processing, and ancillary financial product cross-selling, while generating quantifiable process performance metrics including total processing time in minutes, first-time yield percentages, and channel-specific abandonment rates that inform continuous process improvement initiatives.

        2. Scope

        This procedure applies universally to all account opening applications regardless of origination channel, encompassing in-branch manual banker-assisted applications captured through the Digital Onboarding Platform with manual data entry protocols, mobile application and web portal self-service customer journeys employing Optical Character Recognition auto-fill technology and biometric liveness verification, and complex corporate entity account establishments requiring multi-party beneficial ownership verification and authorized signatory specimen collection, with applicability extending to all customer-facing personnel including Retail Personal Bankers operating within physical branch networks, Business Relationship Managers servicing commercial client portfolios, Compliance Officers and Anti-Money Laundering specialists conducting manual review adjudications, Back-Office Operations personnel performing quality assurance validation, and System Administrators maintaining the decisioning rule matrices embedded within the automated risk evaluation engines. The scope explicitly excludes investment account openings governed by separate securities regulations, trust account establishments requiring fiduciary documentation, and international correspondent banking relationships subject to enhanced cross-border due diligence requirements, but does include all permutations of individual checking accounts, savings accounts, money market deposit accounts, student-classified accounts, and business operating accounts for sole proprietorships, limited liability companies, corporations, and partnerships operating within domestic United States jurisdictions.

        3. Definitions

        **Customer Identification Program (CIP):** The regulatory-mandated verification framework established pursuant to Section 326 of the USA PATRIOT Act requiring financial institutions to implement risk-based procedures for verifying the identity of account applicants through document examination, non-documentary methods, or a combination thereof, culminating in a definitive cip_verification_status determination of "verified," "failed," or "pending_manual_review" based on the institution's ability to form a reasonable belief regarding the true identity of the customer.

        **Beneficial Ownership Identification (BOI):** The process mandated by FinCEN's Customer Due Diligence Final Rule requiring the identification and verification of natural persons who directly or indirectly own twenty-five percent or more of the equity interests in a legal entity customer or who exercise substantial control over such entity, necessitating the collection of name, date of birth, residential address, and social security number for each identified beneficial owner and storage within the beneficial_owners structured data field as a list of JSON objects containing ownership_percentage quantification.

        **ChexSystems Consumer Score:** A tri-bureau negative database risk assessment metric ranging from 300 to 899 generated by analyzing the applicant's deposit account history including occurrences of involuntary account closures, returned checks due to insufficient funds, suspected fraud incidents, and account abuse patterns, with scores below 500 indicating elevated risk requiring enhanced scrutiny, scores between 500-599 triggering conditional approval protocols, and scores at or above 600 supporting streamlined auto-approval pathways when combined with favorable fraud risk score calculations.

        **Fraud Risk Score:** A proprietary algorithmic composite metric scaled from 0 to 100 derived through weighted analysis of multiple risk indicators including inconsistencies between OCR-extracted identification data and manually-entered application information, mismatches between declared residential addresses and credit bureau-reported addresses, anomalous relationships between stated employment status and declared annual income levels, suspicious device fingerprinting patterns indicating potential synthetic identity construction, and behavioral analytics detecting non-human interaction patterns during digital application completion, with scores exceeding 40.0 mandating manual review escalation and scores surpassing 60.0 triggering automatic application decline absent compelling mitigating documentation.

        **Facial Match Confidence Score:** A biometric comparison metric expressed as a percentage from 0 to 100 quantifying the algorithmic certainty that the live facial image captured during liveness detection procedures corresponds to the photograph appearing on the government-issued identification document submitted for verification purposes, calculated through neural network analysis of facial geometry, skin texture mapping, and landmark positioning, with institutional policy requiring minimum threshold achievement of 92.0 percent confidence to satisfy non-documentary CIP verification requirements for digital remote account opening channels.

        **Enhanced Due Diligence (EDD):** An intensive investigation and verification protocol applied to account opening applications that present elevated money laundering risk, terrorist financing exposure, or reputational hazard based on factors including industry classification in high-risk sectors such as money services businesses, cannabis-related enterprises in jurisdictions lacking federal regulatory clarity, politically exposed person status, geographic associations with high-risk jurisdictions identified on FATF lists, complex beneficial ownership structures obscuring ultimate control, or anticipated transaction volumes exceeding ten million dollars monthly, requiring compliance_officer_id assignment and documented compliance_review_notes justifying approval decisions.

        **Auto-Decision Result:** The systematized adjudication output generated by the multi-factor risk evaluation engine that processes inputs including cip_verification_status, watchlist_screening_result, chexsystems_score, fraud_risk_score, credit_bureau_validation_status, and facial_match_confidence_score through predetermined decisioning matrices, producing one of three categorical outcomes: "auto_approved" when all risk metrics fall within acceptable thresholds permitting immediate account establishment without human intervention, "auto_declined" when disqualifying risk factors are detected precluding account relationship establishment, or "refer_for_manual_review" when ambiguous risk indicators require expert compliance officer judgment to render final_approval_status determination.

        **First Time Yield (FTY):** A quality assurance metric calculated as the quotient of applications processed to completion without requiring rework, corrections, or supplemental documentation requests divided by total applications submitted, expressed as a percentage, with the binary first_time_yield field populated as true only when rework_count equals zero, serving as a key performance indicator for process design effectiveness and staff training adequacy.

        **Liveness Detection:** An anti-spoofing biometric verification technique employed during digital remote account opening workflows that requires applicants to perform randomized physical movements such as blinking, turning head laterally, or tilting device at specified angles while video capture occurs, with sophisticated algorithms analyzing micro-movements, shadow patterns, depth perception cues, and physiological indicators to confirm the presence of a living human subject rather than a photograph, video recording, or three-dimensional mask, thereby preventing presentation attacks that could facilitate identity theft or account opening fraud schemes.

        **OFAC Watchlist Screening:** The mandatory sanctions compliance procedure requiring comparison of applicant names, addresses, and other identifying information against the Specially Designated Nationals and Blocked Persons List maintained by the Office of Foreign Assets Control, the Consolidated Sanctions List encompassing multiple U.S. government restricted party databases, and supplementary global watchlists identifying terrorists, weapons proliferators, narcotics traffickers, and human rights violators, with algorithmic fuzzy matching logic generating watchlist_screening_result classifications of "clear" when no potential matches are detected, "potential_match" when name similarity scores fall between 85.0 and 94.9 percent requiring manual disambiguation, or "confirmed_match" when similarity exceeds 95.0 percent combined with corroborating address or date of birth alignment necessitating immediate application decline and suspicious activity reporting.

        **Omnichannel Session Persistence:** The technological capability enabling customers who initiate account applications through one channel (e.g., mobile application) to suspend progress, preserve entered data, and resume completion through an alternative channel (e.g., web portal or in-branch assisted service) without requiring redundant data re-entry, achieved through cloud-based application state synchronization keyed to applicant_email and applicant_phone identifiers validated through otp_verification_completed protocols, designed to reduce abandonment_stage occurrences during lengthy multi-step application workflows.

        4. Input

        The initiation of account opening procedures requires the following prerequisite data elements, documentation artifacts, and system-generated identifiers to be present within the application database record prior to commencement of verification and decisioning workflows. The application_id field must contain a unique alphanumeric string conforming to the institution's identifier taxonomy (format: "APP_" followed by ten-digit timestamp concatenated with five-digit random sequence) to enable unambiguous application tracking across distributed system components. The application_channel field must specify the origination pathway through which the customer accessed account opening services, accepting enumerated values of "in_branch" for physical branch-assisted applications, "mobile_app" for smartphone native application submissions, or "web_portal" for browser-based self-service completions, as this classification determines applicable user experience workflows, data capture methodologies, and staffing resource allocation. The business_type field must categorize the application into one of three taxonomic classifications: "retail_individual" for natural person consumer accounts, "digital_remote" for technology-mediated self-service individual accounts employing biometric verification, or "corporate_business" for legal entity accounts requiring beneficial ownership documentation, thereby routing the application through type-appropriate verification protocols and due diligence depth determinations.

        The application_start_timestamp must record the precise moment of application commencement in ISO 8601 datetime format with UTC timezone specification to enable elapsed time calculations supporting total_processing_time_minutes metric derivation. The customer_type field must identify whether the account relationship is being established for an "individual" natural person, "sole_proprietorship" operating as an individual with business tax identification, "llc" limited liability company, "corporation" incorporated entity, or "partnership" multi-party business arrangement, as this classification governs beneficial ownership collection requirements and authorized signatory documentation expectations. The account_type_requested field must specify the precise deposit product variant sought from the institution's product catalog, accepting values including "checking," "savings," "student_checking," "high_yield_savings," "money_market," or "business_checking," with this selection determining applicable fee schedules, interest rate structures, minimum balance requirements, and regulatory disclosure packages requiring customer acknowledgment.

        For all applicant types, the following personally identifiable information fields must be populated either through banker manual entry, customer self-service data input, or Optical Character Recognition extraction from identification documents: applicant_full_name containing the complete legal name as it appears on government-issued identification without abbreviations, applicant_date_of_birth in ISO date format enabling age verification and minor account detection, applicant_ssn_tax_id storing the nine-digit Social Security Number or Employer Identification Number in encrypted format with only terminal four digits visible in user interfaces, applicant_email providing a valid electronic mail address for correspondence and online banking credential delivery, applicant_phone capturing the primary contact telephone number in E.164 international format, applicant_residential_address specifying the complete current physical residence including street number, directional prefixes, street name, apartment or unit designations, city, state abbreviation, and five-digit ZIP code, applicant_employment_status classifying current occupational situation as "employed," "self_employed," "unemployed," "student," or "retired," and applicant_annual_income quantifying gross yearly earnings in integer dollar amounts.

        Identity verification procedures require capture of id_document_type specifying whether "passport," "drivers_license," or "state_id" is being presented, id_document_number recording the unique identifier printed on the credential, id_document_expiry_date ensuring document validity extends beyond application date, and id_document_issuing_country confirming domestic United States issuance for standard processing pathways. The proof_of_address_type field must indicate whether "utility_bill," "bank_statement," "lease_agreement," or alternative acceptable documentation is being submitted to satisfy CIP regulatory requirements for address verification independent of identification document. The initial_deposit_amount field must specify the monetary value in floating-point format representing the funding commitment, with institutional policy requiring minimum thresholds of $25.00 for standard checking accounts, $100.00 for savings products, and $500.00 for business operating accounts, while the initial_deposit_method field indicates the funding mechanism as "cash," "check," "external_transfer," or "debit_card" which influences account_funding_completed determination timing.

        For digital remote applications specifically, additional inputs include otp_verification_completed boolean confirming successful multi-factor authentication through one-time password validation sent to applicant_phone and applicant_email, id_ocr_extracted_data JSON structure containing parsed fields from automated document reading including name, date of birth, document number, and address elements requiring validation against manually-confirmed data, biometric_liveness_check_completed boolean indicating successful anti-spoofing verification completion, and facial_match_confidence_score floating-point value quantifying biometric comparison certainty. The external_bank_account_linked boolean signals whether application includes Plaid API or similar account aggregation service connection to external financial institution for funding purposes.

        For corporate business applications, mandatory inputs expand to include business_legal_name as registered with state authorities, business_ein nine-digit Employer Identification Number, business_structure enumerated classification, business_industry string descriptor enabling risk categorization, business_registration_date for entity age assessment, business_monthly_transaction_volume_estimate integer providing activity forecasting for transaction monitoring calibration, beneficial_owners list containing JSON objects with name, ownership_percentage, ssn, and dob for all individuals meeting twenty-five percent threshold or exercising control, authorized_signers list identifying individuals empowered to conduct account transactions with associated titles and signature capture confirmation, and documents_submitted list enumerating all uploaded files with document types including "articles_of_incorporation," "operating_agreement," "business_license," and "ein_confirmation_letter."

        Administrative metadata inputs include banker_id identifying the employee facilitating the application when applicable, branch_code specifying the physical location for in-branch originations, customer_consent_forms_signed list tracking all regulatory disclosure acknowledgments, esignature_timestamp and esignature_ip_address for electronic signature audit trail establishment, marketing_opt_in boolean for communication preference capture, referral_source enumerated classification for marketing attribution, product_bundle_offered string for cross-sell tracking, cross_sell_products_accepted list identifying ancillary services activated, and application_data_entry_method classification specifying whether "manual_banker_entry," "ocr_auto_fill," or "customer_self_service" data capture occurred.

        5. Main Procedure

        5.1 Customer Identification Program Verification and Identity Validation Protocol Execution

        5.1.1 Upon confirmation that all mandatory input fields specified in Section 4 have been populated within the application record and the application_start_timestamp has been registered in the system, the CIP verification orchestration engine must initiate the multi-layered identity validation sequence designed to establish reasonable belief in the applicant's claimed identity pursuant to regulatory requirements. For applications where application_data_entry_method equals "manual_banker_entry" or "customer_self_service," the system must perform primary data integrity checks comparing applicant_full_name, applicant_date_of_birth, and applicant_ssn_tax_id against formatting validation rules including Social Security Number pattern matching confirming nine-digit structure with non-sequential and non-repetitive characteristics, date of birth logical validation ensuring applicant age falls between eighteen and one hundred twenty years for individual accounts or identification of minor status triggering custodial account requirements, and name field validation confirming presence of both given name and surname components with character restrictions preventing numeric or special character inclusion except for legitimate apostrophes and hyphens in compound names. When application_channel equals "mobile_app" or "web_portal" and id_ocr_extracted_data JSON structure contains populated values, the system must execute cross-reference validation comparing OCR-extracted name, date of birth, and document number against corresponding manually-entered or user-confirmed application fields, calculating a data consistency score where perfect alignment yields 100.0 and any discrepancy reduces the score by 25.0 points per mismatched field, with consistency scores below 75.0 automatically populating manual_review_required as true and manual_review_reason as "OCR data inconsistency detected requiring verification."

        5.1.2 Following successful completion of data integrity validation, the system must invoke integration with the designated Identity Verification Service platform operating under institutional service level agreements, transmitting applicant_full_name, applicant_date_of_birth, applicant_ssn_tax_id, applicant_residential_address, id_document_type, id_document_number, and id_document_issuing_country through secure API endpoints employing TLS 1.3 encryption protocols. The Identity Verification Service must execute document authentication procedures analyzing submitted identification images for presence of security features including holographic overlays, ultraviolet reactive elements, microprinting characteristics, and substrate composition patterns consistent with genuine government issuance, returning a document_authenticity_score ranging from 0 to 100 where scores at or above 85.0 satisfy documentary verification requirements and scores below 70.0 trigger cip_verification_status population as "failed" with decline_reason notation "Identification document authenticity could not be confirmed." For applications where biometric_liveness_check_completed equals true, the Identity Verification Service must additionally execute facial biometric comparison algorithms analyzing geometric feature relationships between the live-captured facial image and the photograph extracted from the identification document, calculating and returning the facial_match_confidence_score which must meet or exceed the institutional threshold of 92.0 to contribute positively toward cip_verification_status determination, with scores falling between 88.0 and 91.9 requiring secondary verification through knowledge-based authentication question sets and scores below 88.0 resulting in biometric verification failure requiring alternative documentary verification pathways.

        5.1.3 Concurrent with Identity Verification Service processing, the system must initiate credit_bureau_validation through established data furnisher relationships with major consumer reporting agencies, submitting applicant_ssn_tax_id, applicant_full_name, applicant_date_of_birth, and applicant_residential_address through specialized identity verification products distinct from full credit report pulls to avoid triggering hard inquiry impacts on applicant credit scores. The credit bureau must return validation response indicating whether the submitted combination of personally identifiable information corresponds to established credit file records within their database, with responses categorized as "verified" when all four elements match existing records with high confidence, "not_verified" when submitted data lacks sufficient bureau file history or contains conflicting elements, or "insufficient_data" when the applicant represents a credit-invisible population lacking adequate bureau presence to confirm identity. The credit_bureau_validation_status field must be populated with the returned classification, with "verified" outcomes contributing positively to aggregate CIP verification confidence while "not_verified" results require supplementary documentary evidence collection such as utility bill submission for proof_of_address_type validation and "insufficient_data" responses necessitate alternative verification methodologies acceptable under risk-based CIP frameworks including government database cross-referencing where available or notarized affidavit acceptance in limited circumstances.

        5.1.4 Upon completion of all parallel identity verification processes including document authentication scoring, facial biometric confidence calculation when applicable, and credit bureau validation status determination, the system must execute the CIP Verification Decision Matrix algorithm that synthesizes verification outcomes into a definitive cip_verification_status classification. The decision logic must evaluate whether document_authenticity_score achieved the 85.0 minimum threshold, whether facial_match_confidence_score met the 92.0 requirement for digital applications or whether in-branch physical document examination occurred for manual applications, and whether credit_bureau_validation_status returned "verified" or acceptable alternative verification was completed. When all applicable verification components achieve passing thresholds, the cip_verification_status must be populated as "verified" enabling workflow progression to risk screening stages; when one or more verification components fail to meet minimum standards but no definitive fraud indicators are present, the status must be set to "pending_manual_review" with manual_review_required set to true and manual_review_reason documenting the specific verification component requiring compliance officer assessment; when multiple verification failures occur or document_authenticity_score falls below 50.0 indicating probable counterfeit credential presentation, the status must be populated as "failed" with immediate workflow termination and decline_reason documentation specifying "Unable to verify customer identity pursuant to Customer Identification Program requirements."

        5.2 Regulatory Watchlist Screening and Negative Database Risk Assessment Execution

        5.2.1 Applications achieving cip_verification_status of "verified" must immediately trigger invocation of the institutional Watchlist Screening Tool which maintains subscription access to the Office of Foreign Assets Control Specially Designated Nationals List, the Bureau of Industry and Security Denied Persons List, the Financial Crimes Enforcement Network 314(a) request database, and commercial watchlist aggregation services encompassing global sanctions regimes, politically exposed person databases, and adverse media monitoring feeds. The screening engine must receive applicant_full_name, applicant_date_of_birth when available for disambiguation purposes, applicant_residential_address for geographic risk assessment, and applicant_ssn_tax_id as a unique identifier, executing fuzzy matching algorithms that account for name variations, transliteration inconsistencies, and typographical errors through phonetic encoding and edit distance calculations. The screening process must generate individual similarity scores for each potential match detected, with algorithmic logic comparing submitted name components against watchlist entries and calculating percentage-based confidence that the applicant and the listed individual represent the same person. The watchlist_screening_result field must be populated according to the following threshold-based classification rules: when no potential matches are identified with similarity scores exceeding 70.0 percent, the result must be recorded as "clear" permitting unimpeded workflow continuation; when one or more potential matches generate similarity scores between 85.0 and 94.9 percent without corroborating secondary identifier alignment such as matching date of birth or address components, the result must be classified as "potential_match" requiring manual disambiguation by compliance personnel; when similarity scores reach or exceed 95.0 percent or when scores between 85.0 and 94.9 percent are accompanied by date of birth alignment within thirty days or address geographic concordance at the city level, the result must be designated as "confirmed_match" triggering immediate application decline and suspicious activity reporting obligations pursuant to 31 CFR 1020.320.

        5.2.2 For applications where watchlist_screening_result returns "potential_match," the system must automatically populate manual_review_required as true, assign the case to the compliance officer queue with lowest current workload as determined by pending case counts, record the assigned compliance_officer_id, and generate detailed match reports displaying side-by-side comparison of applicant information against each watchlist entry triggering the potential match classification including similarity score quantification, matching algorithm methodology explanation, and recommended disambiguation investigation procedures. The assigned compliance officer must access the case within institutional service level agreement parameters not to exceed four business hours from assignment, review all available applicant documentation including identification images, proof of address documents, and any supplementary information provided during application submission, and conduct enhanced research through commercial database services that may include LexisNexis public records searches, social media profile examination, employment verification through LinkedIn or similar professional networking platforms, and when necessary, direct applicant contact requesting additional clarifying documentation such as recent passport pages displaying international travel stamps or notarized statements of non-affiliation with sanctioned individuals sharing similar names. The compliance officer must document all investigation steps, findings, and analytical reasoning within the compliance_review_notes free-text field with minimum documentation standards requiring identification of specific databases consulted, description of disambiguating factors discovered, and clear articulation of the basis for concluding that the applicant is or is not the same individual appearing on the watchlist. Upon completion of investigation, the compliance officer must manually override the watchlist_screening_result field to either "clear" when sufficient evidence establishes the applicant is not the listed individual, or "confirmed_match" when investigation reveals the applicant is indeed subject to sanctions, with the latter outcome mandating immediate population of final_approval_status as "declined," decline_reason as "Applicant identified on OFAC Specially Designated Nationals List," and initiation of blocking procedures preventing any account activation or funds transfer.

        5.2.3 Concurrently with OFAC watchlist screening execution, the system must initiate ChexSystems Consumer Score retrieval for all individual retail and digital remote applications where customer_type equals "individual," transmitting applicant_full_name, applicant_ssn_tax_id, applicant_date_of_birth, and applicant_residential_address to the ChexSystems reporting agency through established batch file transmission or real-time API integration channels. ChexSystems must return the three-digit risk score derived from analysis of the applicant's deposit account history across participating financial institutions, quantifying the likelihood of future account mismanagement based on historical patterns of check kiting, suspected fraud incidents, returned deposited items, negative balance charge-offs, and account closures for cause. The returned chexsystems_score integer value ranging from 100 to 899 must be recorded in the application database, with institutional risk policy establishing the following score interpretation and routing framework: scores at or above 600 represent acceptable risk for standard processing through automated approval pathways when combined with satisfactory performance on other risk metrics; scores between 500 and 599 indicate moderate risk requiring enhanced scrutiny with additional weight applied to fraud_risk_score calculations and potential limitation of initial account features such as delayed debit card issuance or restricted mobile deposit availability during initial relationship monitoring periods; scores between 400 and 499 necessitate manual_review_required population as true with compliance officer assessment of specific negative events contributing to score depression and evaluation of whether mitigating circumstances such as identity theft victimization or medical financial hardship justify approval despite elevated risk indicators; scores below 400 create presumptive decline recommendations that may only be overridden through documented compliance officer authorization with enhanced due diligence completion and ongoing transaction monitoring calibration to detect potential account abuse patterns.

        5.2.4 For corporate business applications where business_type equals "corporate_business," the ChexSystems scoring component must be supplemented with commercial entity verification procedures that validate business_legal_name registration with the applicable Secretary of State through integration with commercial entity database services, confirm business_ein validity through Internal Revenue Service verification systems where available or alternative tax authority validation, and assess business_industry classification against the institution's risk appetite framework to identify high-risk industry associations requiring enhanced_edd designation. The system must cross-reference business_industry string values against a predefined taxonomy of elevated-risk sectors including money services businesses requiring state licensing verification, cannabis-related enterprises operating in jurisdictional regulatory ambiguity, virtual currency exchangers subject to FinCEN money services business registration, international trade facilitators with sanctions screening exposure, politically exposed person-owned entities requiring source of wealth documentation, and businesses incorporated in high-risk jurisdictions identified on Financial Action Task Force monitoring lists. When business_industry mapping identifies alignment with high-risk classifications, the due_diligence_level field must be automatically populated as "enhanced_edd," triggering additional documentation requirements including business plan submission, source of funds attestation letters, detailed transaction activity forecasting, and potentially site visit completion by business relationship management personnel to verify physical business operations correspond to stated activities. Conversely, when business_industry represents low-risk professional services, established franchise operations with national brand affiliation, or locally-recognized small businesses with minimal transactional complexity indicated by business_monthly_transaction_volume_estimate below $50,000, the due_diligence_level may be classified as "standard_cdd" permitting streamlined documentation review and approval workflows.

        5.3 Algorithmic Fraud Risk Scoring and Multi-Factor Auto-Decision Engine Processing

        5.3.1 Upon successful completion of CIP verification processes yielding cip_verification_status of "verified" and watchlist screening procedures returning watchlist_screening_result of "clear" or resolved "potential_match" outcomes, the application must progress to the Fraud Risk Scoring Engine which synthesizes multiple behavioral, demographic, and transactional risk indicators into a unified fraud_risk_score quantitative metric. The scoring algorithm must implement a weighted multi-factor model incorporating the following components with specified contribution percentages: analysis of consistency between id_ocr_extracted_data JSON elements and manually-entered application fields contributes fifteen percent weight, with perfect alignment scoring 0 risk points and each inconsistency adding 10 risk points to this component; evaluation of applicant_residential_address alignment with credit bureau-reported addresses or utility billing address validation contributes twenty percent weight, with exact matches scoring 0 risk points, same ZIP code but different street addresses adding 8 risk points, different cities within same state adding 15 risk points, and different states or post office box addresses adding 25 risk points; assessment of applicant_employment_status and applicant_annual_income logical coherence contributes fifteen percent weight, with employed status claiming income above $200,000 without corroborating high-end residential address characteristics adding 12 risk points, unemployed status claiming income above $30,000 adding 18 risk points, and student status claiming income above $60,000 adding 10 risk points; analysis of initial_deposit_amount relationship to stated income contributes ten percent weight, with deposits exceeding twenty-five percent of annualized income for new customer relationships adding 15 risk points; device fingerprinting analysis for digital applications contributes twenty percent weight, with known device profiles associated with the applicant_email or applicant_phone scoring 0 risk points, new device profiles from residential IP addresses scoring 5 risk points, VPN or proxy detection adding 20 risk points, and evidence of device emulation or automation tooling adding 40 risk points; and behavioral analytics assessment of application completion patterns contributes twenty percent weight, with natural human keystroke dynamics and mouse movement patterns scoring 0 risk points, suspiciously rapid form completion suggesting auto-fill scripting adding 12 risk points, and detection of copy-paste operations for all fields suggesting synthetic identity template usage adding 25 risk points.

        5.3.2 The Fraud Risk Scoring Engine must aggregate the weighted component scores into a composite fraud_risk_score floating-point value ranging from 0.0 to 100.0, with the final score recorded in the application database and subjected to threshold-based classification that influences auto_decision_result determination. The institutional fraud risk tolerance framework establishes the following score interpretation guidelines: fraud_risk_score values at or below 25.0 represent acceptable risk for streamlined processing when combined with satisfactory performance on other decisioning factors; scores between 25.1 and 40.0 indicate moderate fraud concern requiring additional verification steps such as outbound telephone contact to applicant_phone to verbally confirm application submission and account opening intent, with successful verbal confirmation permitting continued processing while inability to reach applicant or verification failure triggering manual_review_required population; scores between 40.1 and 60.0 create elevated fraud suspicion mandating manual compliance officer review with detailed investigation of specific risk factors contributing to score elevation, collection of supplementary documentation addressing identified inconsistencies, and documented justification required for any approval decision; scores exceeding 60.0 generate presumptive fraud determination resulting in automatic decline absent extraordinary mitigating circumstances such as verified identity theft reporting with law enforcement case numbers or documented data entry errors correctable through amendment processes.

        5.3.3 With fraud_risk_score calculation completed and all upstream verification processes finalized, the Multi-Factor Auto-Decision Engine must execute comprehensive risk evaluation synthesizing cip_verification_status, watchlist_screening_result, chexsystems_score when applicable, fraud_risk_score, credit_bureau_validation_status, and facial_match_confidence_score for digital applications into a holistic risk profile subjected to decision matrix processing. The auto-decision algorithm must implement the following hierarchical evaluation logic: first-level screening eliminates applications where cip_verification_status equals "failed" or watchlist_screening_result equals "confirmed_match," immediately populating auto_decision_result as "auto_declined" and final_approval_status as "declined" with appropriate decline_reason documentation; second-level screening identifies applications where any individual risk metric falls into manual review threshold ranges including chexsystems_score between 400-499, fraud_risk_score between 40.1-60.0, facial_match_confidence_score between 88.0-91.9, or credit_bureau_validation_status equals "not_verified," populating auto_decision_result as "refer_for_manual_review" and manual_review_required as true; third-level screening applies the optimistic approval pathway for applications where all of the following conditions are satisfied: cip_verification_status equals "verified," watchlist_screening_result equals "clear," chexsystems_score greater than or equal to 600 or application represents corporate account not subject to ChexSystems scoring, fraud_risk_score less than or equal to 25.0, credit_bureau_validation_status equals "verified," and for digital applications facial_match_confidence_score greater than or equal to 92.0, populating auto_decision_result as "auto_approved" and proceeding directly to account generation workflows without human intervention; fourth-level screening addresses residual applications not meeting auto-approval criteria but lacking disqualifying factors, applying weighted composite scoring where chexsystems_score contributes forty percent weight with normalization to 100-point scale through formula (chexsystems_score minus 100) divided by 7.99, fraud_risk_score contributes thirty percent weight with inversion through formula 100 minus fraud_risk_score, facial_match_confidence_score contributes twenty percent weight directly when available or proportional reweighting to other factors when not applicable, and credit_bureau verification contributes ten percent weight with "verified" scoring 100 points and "insufficient_data" scoring 60 points, calculating aggregate composite_risk_score where values at or above 70.0 result in auto_decision_result population as "auto_approved" while values below 70.0 yield "refer_for_manual_review" classification.

        5.3.4 Applications receiving auto_decision_result classification of "refer_for_manual_review" must be routed to the compliance officer queue with case assignment based on workload balancing algorithms and specialized expertise matching where applicable, such as routing corporate business applications to officers with commercial banking due diligence experience or routing high-net-worth individual applications indicating applicant_annual_income exceeding $500,000 to officers with private banking regulatory knowledge. The assigned compliance_officer_id must be recorded and compliance_review_timestamp initialized upon case access by the reviewing officer. The compliance officer must conduct comprehensive file review examining all source documentation within the documents_submitted list, validating internal consistency of reported information across multiple fields, assessing the credibility of unusual circumstances documented in application notes such as large cash deposits from stated inheritance proceeds or business sale transactions, and exercising expert judgment regarding whether identified risk factors represent genuine fraud indicators or explainable anomalies consistent with legitimate customer profiles. For corporate applications requiring beneficial_owners verification, the compliance officer must validate that all individuals meeting the twenty-five percent ownership threshold or exercising control have been identified, that personal identification documentation has been collected for each beneficial owner, and that the ownership structure disclosed represents a complete and accurate depiction of control relationships rather than nominee arrangements obscuring ultimate beneficial ownership. The officer must document detailed analytical conclusions within compliance_review_notes, explicitly addressing each risk factor that triggered manual review and articulating the specific evidence or reasoning supporting the ultimate approval or decline determination. Upon completion of review, the compliance officer must populate final_approval_status as either "approved" when risk factors are satisfactorily resolved and the application meets institutional risk appetite, "declined" when unacceptable risk or material misrepresentation is identified with decline_reason providing customer-facing explanation of the adverse action, or "conditional_approval" when approval is granted subject to operational restrictions such as delayed debit card access, limited daily transaction thresholds, or enhanced ongoing monitoring protocols with such conditions documented in compliance_review_notes for operational implementation.

        5.4 Account Establishment in Core Banking System and Customer Relationship Activation

        5.4.1 Applications achieving final_approval_status of "approved" whether through auto_decision_result of "auto_approved" or compliance officer manual adjudication must trigger the Account Generation Orchestration Process which interfaces with the institution's Core Banking System to create the formal account relationship and establish the system of record for deposit activity. The orchestration engine must compile all validated customer data elements including applicant_full_name, applicant_date_of_birth, applicant_ssn_tax_id, applicant_residential_address, applicant_email, applicant_phone, applicant_employment_status, and account_type_requested, format them according to the Core Banking System's data ingestion specifications which may require field length truncations, character set conversions, or standardized abbreviations for state names and street type designations, and transmit them through secure API integration endpoints or batch file processing channels depending on the core system's technical architecture. The Core Banking System must execute its internal account creation procedures including assignment of a unique account_number_generated conforming to the institution's account numbering schema which may incorporate branch_code prefixes, product type identifiers, and check digit validation algorithms, association of the standard routing_number representing the financial institution's American Bankers Association routing transit number for ACH and wire transfer processing, establishment of the initial account balance as zero pending funding completion, configuration of fee schedule parameters corresponding to the selected account product type, and initialization of interest accrual rates when applicable for savings and money market account types.

        5.4.2 Upon successful account creation in the Core Banking System with confirmation response including the assigned account_number_generated, the system must record account_creation_timestamp capturing the precise moment of system of record establishment, calculate total_processing_time_minutes as the integer difference in minutes between application_start_timestamp and account_creation_timestamp to populate process efficiency metrics, and evaluate whether rework_count equals zero to populate first_time_yield as true indicating quality processing without corrections or supplemental documentation requests. The system must then initiate parallel processes for customer communication, account funding facilitation, and ancillary product provisioning. For customer communication, the system must generate and transmit account opening confirmation messages via applicant_email containing account_number_generated, routing_number, summary of account terms including fee schedules and interest rates when applicable, instructions for accessing online banking services, and legal disclosures required by Regulation DD including Annual Percentage Yield calculations, compounding frequency specifications, and fee structure transparency disclosures. When application_channel equals "in_branch" and banker_id is populated, the system must generate branch-side notifications enabling the personal banker to present physical welcome kit materials to the customer including printed account confirmation documents, signature cards for physical records archiving, and promotional materials for cross-sell products aligned with the product_bundle_offered field.

        5.4.3 Account funding procedures must be initiated based on the initial_deposit_method specified during application submission, with distinct processing pathways for each funding mechanism. When initial_deposit_method equals "cash" or "check," the system must interface with branch teller platforms to create a pending deposit transaction awaiting physical instrument presentment, with account_funding_completed remaining false until the teller completes the deposit posting transaction and confirms funds availability; when the method equals "external_transfer" and external_bank_account_linked is true, the system must invoke Plaid API or equivalent account aggregation service to initiate an ACH pull transaction from the verified external account for the amount specified in initial_deposit_amount, with transaction status monitoring occurring over the standard ACH settlement timeline of one to three business days and account_funding_completed populating as true upon successful funds receipt and posting to the new account; when the method equals "debit_card," the system must execute immediate card-not-present debit authorization and settlement for initial_deposit_amount, with near-instantaneous funds availability enabling account_funding_completed to be set true within minutes of approval. The account_activation_status field must be populated based on funding completion, with values of "pending_funding" assigned when account_number_generated exists but account_funding_completed remains false, "active" assigned when funding is successfully completed enabling full account functionality, and "dormant" assigned when funding is not completed within institutional timelines such as seven business days for in-branch applications or three business days for digital applications, triggering automated reminder communications to applicant_email and applicant_phone encouraging deposit completion to activate services.

        5.4.4 Ancillary product provisioning must proceed based on the account_type_requested and any cross_sell_products_accepted during the application process, with distinct fulfillment workflows for various product components. Debit card issuance requires submission of card manufacturing orders to the institution's card processor including account_number_generated for embossing and magnetic stripe encoding, customer name and address for physical delivery via postal service, and card design template selection based on account type such as standard designs for basic checking accounts or premium metallic cards for high-balance relationship packages, with debit_card_number_issued recording the masked sixteen-digit card number upon successful card production and the physical card typically arriving at applicant_residential_address within seven to ten business days. Online banking credential creation requires generation of unique username and temporary password combinations transmitted securely to applicant_email, with online_banking_credentials_created populated as true upon successful credential provisioning and customer able to activate access immediately through web portal or mobile application login workflows requiring password reset on first use and multi-factor authentication enrollment for ongoing security. Check stock ordering for checking accounts requires determination of initial checkbook specifications including standard quantities such as one hundred checks for personal accounts or two hundred checks for business accounts, selection of check design templates, incorporation of account_number_generated and routing_number in magnetic ink character recognition line encoding, printing of customer name and applicant_residential_address in upper left corner positioning, and processing through third-party check printing vendors with delivery timelines typically ranging from ten to fifteen business days, with temporary_checks_issued field indicating whether counter checks or starter checks with limited quantities were provided for immediate transaction needs during permanent checkbook production and delivery intervals.

        5.5 Process Completion Documentation and Performance Metrics Recording

        5.5.1 Upon finalization of all account establishment procedures including Core Banking System account creation, funding processing, and ancillary product provisioning, the system must execute comprehensive process completion documentation activities to support operational reporting, regulatory examination preparation, and continuous improvement analytics. The welcome_kit_issued field must be populated as true when all applicable customer onboarding materials have been delivered whether through physical branch presentation for in_branch applications or electronic transmission via email for digital channels, with the welcome kit components including account terms summary, fee schedule disclosure, privacy notice pursuant to Gramm-Leach-Bliley Act requirements, funds availability policy disclosure pursuant to Regulation CC, electronic fund transfer agreement when debit cards or online banking are activated, and any promotional offer confirmations related to new account bonuses or relationship pricing benefits. The system must verify that customer_consent_forms_signed list contains all regulatory-required acknowledgments including signature card or electronic signature capture for account ownership establishment, Truth in Savings Act disclosure acknowledgment, privacy notice receipt confirmation, electronic communications consent when applicable for paperless statement delivery, and any product-specific terms such as overdraft protection agreements or linked account authorizations.

        5.5.2 Process performance metrics must be calculated and recorded to support service level monitoring and process improvement initiatives aligned with the institutional continuous improvement methodology. The total_processing_time_minutes field representing elapsed time from application_start_timestamp to account_creation_timestamp must be analyzed against service level agreement targets which typically specify maximum processing times of forty-five minutes for in-branch applications with banker assistance, fifteen minutes for digital remote applications achieving auto-approval, and four business hours for applications requiring manual compliance review, with deviations from these targets triggering management escalation and root cause analysis to identify process bottlenecks or resource constraint issues. The first_time_yield boolean must aggregate across all applications to calculate institutional FTY percentage through formula (count of applications where first_time_yield equals true) divided by (total application count) multiplied by 100, with target benchmarks typically seeking ninety-five percent or higher FTY performance indicating effective process design, adequate staff training, and appropriate system validation rules preventing common data entry errors. The rework_count integer must be analyzed to quantify process inefficiency costs with each rework iteration representing duplicative effort, extended cycle times, and degraded customer experience, driving operational excellence initiatives focused on enhancing upfront data validation, clarifying customer instructions, and streamlining documentation requirements.

        5.5.3 For digital channel applications specifically, abandonment analysis must be conducted to identify friction points in the self-service application workflow that cause customers to discontinue the process before completion. When application_abandoned equals true indicating the customer initiated but did not complete the application within institutional timeout thresholds such as seven days from application_start_timestamp, the abandonment_stage field must be analyzed to determine where in the workflow the customer discontinued participation, with common abandonment points including the ID document capture stage when customers experience difficulty photographing identification cards in adequate lighting conditions or acceptable focus quality, the liveness detection stage when customers feel uncomfortable performing facial movements or experience technical difficulties with camera activation, the external bank account linking stage when customers express reluctance to provide login credentials to third-party aggregation services or encounter connection failures with their existing financial institutions, and the funding stage when customers reconsider the initial_deposit_amount commitment or prefer to delay deposit until physical branch visit. Aggregation of abandonment_stage data across multiple applications enables product management to prioritize user experience enhancements such as implementing progressive disclosure techniques to reduce perceived application length, providing real-time guidance overlays during ID capture to improve photo quality, offering alternative verification methods for customers uncomfortable with biometric processes, and creating "save and resume" functionality with omnichannel session persistence enabling application completion across multiple devices and sessions.

        5.5.4 Regulatory compliance metrics must be calculated and monitored to ensure ongoing adherence to Customer Identification Program requirements, sanctions screening effectiveness, and beneficial ownership documentation completeness. The percentage of applications achieving cip_verification_status of "verified" without manual intervention must be tracked as an indicator of identity verification process robustness, with target performance typically exceeding ninety percent for individual retail applications and seventy percent for corporate business applications acknowledging greater complexity in entity verification workflows. The watchlist_screening_result distribution must be analyzed to quantify "clear" versus "potential_match" versus "confirmed_match" outcomes, with unexpectedly high potential match rates potentially indicating overly sensitive screening algorithm parameters requiring threshold recalibration to reduce false positive compliance workload while maintaining regulatory effectiveness. For corporate business applications, the completeness of beneficial_owners documentation must be validated ensuring all ownership structures claiming four or fewer beneficial owners meeting the twenty-five percent threshold have supporting ownership percentage documentation totaling one hundred percent of entity equity and all authorized_signers lists include signature specimen captures for all individuals granted transaction authority, with deficiency rates tracked and addressed through enhanced relationship manager training and digital workflow validations preventing incomplete application submission.

        6. Output

        Upon successful completion of all procedures specified herein, the account opening process shall yield the following definitive, quantifiable outputs recorded within the application database and integrated downstream systems. The cip_verification_status field shall contain a terminal classification of "verified," "failed," or "pending_manual_review" representing the institution's determination of whether the applicant's identity has been satisfactorily confirmed pursuant to 31 CFR 1020.220 requirements. The watchlist_screening_result field shall document "clear," "potential_match," or "confirmed_match" indicating whether the applicant appears on Office of Foreign Assets Control sanctions lists or other restricted party databases requiring transaction blocking. The chexsystems_score field shall contain an integer value between 100 and 899 quantifying the applicant's deposit account risk profile based on historical banking behavior patterns, or shall remain null for corporate entity applications not subject to consumer reporting agency scoring. The fraud_risk_score field shall contain a floating-point value between 0.0 and 100.0 representing the algorithmically-calculated composite fraud probability based on weighted analysis of behavioral, demographic, and consistency risk factors. The auto_decision_result field shall specify "auto_approved," "auto_declined," or "refer_for_manual_review" documenting whether automated decisioning systems rendered definitive adjudication or escalated the application for human expert review.

        For applications requiring manual compliance intervention, the compliance_officer_id field shall identify the specific reviewing officer, the compliance_review_timestamp shall record the ISO 8601 formatted datetime of review completion, and the compliance_review_notes field shall contain detailed textual documentation of investigative procedures conducted, findings discovered, and analytical reasoning supporting the ultimate approval or decline decision. The final_approval_status field shall contain the terminal adjudication of "approved," "declined," or "conditional_approval" representing institutional willingness to establish the customer relationship, with decline_reason field populated with customer-facing adverse action explanation when final_approval_status equals "declined" to support Fair Credit Reporting Act disclosure requirements.

        For approved applications, the account_number_generated field shall contain the unique alphanumeric identifier assigned by the Core Banking System serving as the primary account reference for all future transactions, statements, and customer communications, formatted according to institutional numbering conventions and validated through check digit algorithms. The routing_number field shall contain the nine-digit American Bankers Association routing transit number enabling ACH and wire transfer processing to and from the established account. The account_creation_timestamp field shall document the precise moment of system of record establishment in ISO 8601 datetime format with UTC timezone specification. The account_activation_status field shall indicate "active," "pending_funding," "dormant," or "closed" representing the current operational state of the account relationship.

        Process performance outputs shall include the total_processing_time_minutes integer value quantifying elapsed time from application initiation to account creation, enabling service level agreement compliance monitoring and continuous improvement analytics. The first_time_yield boolean shall indicate whether the application was processed correctly without requiring rework, corrections, or supplemental documentation requests, with rework_count integer documenting the specific number of iteration cycles required for successful completion. The welcome_kit_issued boolean shall confirm delivery of all customer onboarding materials, while debit_card_number_issued shall document the masked card number for provisioned payment instruments, online_banking_credentials_created shall verify digital banking access establishment, and temporary_checks_issued shall indicate whether immediate transaction instruments were provided pending permanent checkbook delivery.

        For digital channel applications, the application_abandoned boolean shall indicate whether the customer discontinued the self-service workflow prior to completion, with abandonment_stage field specifying the precise workflow step at which discontinuation occurred to inform user experience optimization initiatives. Aggregate metrics derived from individual application outputs shall include abandonment rate calculated as (count of applications where application_abandoned equals true) divided by (total applications initiated) multiplied by 100, auto-approval rate calculated as (count of applications where auto_decision_result equals "auto_approved") divided by (total applications submitted) multiplied by 100, and first time yield rate calculated as (count of applications where first_time_yield equals true) divided by (total applications processed) multiplied by 100, providing quantitative key performance indicators supporting operational excellence management and strategic resource allocation decisions.

        </SOP>'''
    additional_context = '''
    ### 1. Business Overview

    *   **Scenario Name:** Retail & Corporate Bank Account Opening
    *   **Industry:** Financial Services / Banking
    *   **Primary Purpose / Value:** To formally establish a contractual relationship between a customer (individual or entity) and the bank, enabling the customer to deposit funds, facilitate transactions, and access financial products. The core value lies in customer acquisition, regulatory compliance (KYC/AML), and setting the foundation for cross-selling future banking services.

    ### 2. Business Types

    *   **Sub-business A: Retail Individual Account Opening (In-Branch):** The traditional process where an individual visits a physical branch to open a checking or savings account with the assistance of a bank teller or personal banker.
    *   **Sub-business B: Digital Remote Account Opening (Mobile/Web):** An automated, self-service process allowing individuals to open accounts via a smartphone app or website using biometric verification and OCR technology.
    *   **Sub-business C: Corporate/Business Account Opening:** A complex process for legal entities (LLCs, Corporations, Sole Proprietorships) requiring extensive documentation to verify business legitimacy, beneficial ownership, and authorized signatories.

    ### 3. Process Description

    *   **Sub-business A Process: Retail Individual Account Opening (In-Branch)**
        1.  **Customer Reception & Needs Assessment:** The banker greets the customer and conducts a brief interview to determine the appropriate account type (e.g., Student Checking, High-Yield Savings) based on their financial needs.
        2.  **Identity Verification (CIP):** The banker requests and validates physical identification documents (Passport, Driver's License) and proof of address (Utility Bill) to satisfy Customer Identification Program (CIP) requirements.
        3.  **Profile Creation & Data Entry:** The banker manually inputs customer demographic data (Name, SSN/Tax ID, Address, Employment status) into the Core Banking System.
        4.  **Regulatory Screening:** The system automatically runs the customer's data against watchlists (OFAC, Politically Exposed Persons) and checking history databases (e.g., ChexSystems) to assess risk.
        5.  **Account Selection & Disclosures:** The banker presents the specific account terms, fee schedules, and interest rates. The customer signs the signature card and consent forms physically or on a signature pad.
        6.  **Initial Deposit Processing:** The customer provides cash or a check for the minimum opening deposit, which the banker processes immediately at the teller station.
        7.  **Welcome Kit Issuance:** The banker provides a starter kit (temporary debit card, checkbook, online banking credentials) and the account number is generated.

    *   **Sub-business B Process: Digital Remote Account Opening (Mobile/Web)**
        1.  **User Registration:** The customer downloads the app or visits the site, entering a mobile number and email address to receive a One-Time Password (OTP) for device verification.
        2.  **ID Capture & OCR Extraction:** The customer takes a photo of the front and back of their government-issued ID. The system uses Optical Character Recognition (OCR) to auto-fill the application form.
        3.  **Liveness Check (Biometrics):** The customer is prompted to take a "video selfie" or perform specific facial movements (blink, turn head) to verify they are a live person and match the photo on the ID.
        4.  **Data Validation & E-KYC:** The system validates the extracted data against external credit bureaus and government databases in real-time to confirm identity and residency.
        5.  **Risk Decisioning:** An automated decision engine evaluates the application. It returns a result: "Approved," "Declined," or "Refer for Manual Review" (based on fraud scores or credit history).
        6.  **Electronic Signature (e-Sign):** The customer reviews the terms and conditions and taps to "sign" the agreement digitally.
        7.  **Funding the Account:** The customer links an external bank account (via Plaid or similar API) or uses a debit card to transfer the initial deposit.
        8.  **Account Activation:** The system generates the account number and routing number, displaying them on the screen immediately.

    *   **Sub-business C Process: Corporate/Business Account Opening**
        1.  **Application Initiation:** The authorized company representative requests an account opening meeting and submits the initial application form outlining the business structure and expected transaction volume.
        2.  **Document Collection:** The banker collects required legal documents: Articles of Incorporation, EIN/Tax ID certification, Operating Agreements, and Business License.
        3.  **Beneficial Ownership Verification (UBO):** The banker identifies all individuals who own 25% or more of the equity interests or have significant control. Personal identification is collected for all Beneficial Owners.
        4.  **Due Diligence Review (CDD/EDD):** The compliance team reviews the nature of the business to ensure it fits the bank's risk appetite (e.g., screening against high-risk industries like gambling or cannabis, if restricted).
        5.  **Signatory Authority Setup:** The banker establishes who has the authority to sign checks, initiate wires, and manage the account. Specimen signatures are collected for all authorized signers.
        6.  **System Entry & Approval:** All data and scanned documents are entered into the onboarding platform. A branch manager or back-office operations officer reviews the file for completeness and approves the account.
        7.  **Account Generation:** The system generates the business account number and initiates the order for business debit cards and check stock.

    ### 4. Key Roles and Responsibilities

    *   **Retail Personal Banker:** Responsible for face-to-face customer interaction, data entry accuracy, initial KYC checks, and cross-selling banking products.
    *   **Compliance / AML Officer:** Responsible for reviewing "Refer for Manual Review" cases, investigating potential watchlist hits (OFAC), and ensuring the bank adheres to the Bank Secrecy Act (BSA).
    *   **Business Relationship Manager:** Responsible for managing corporate client onboarding, understanding complex business structures, and gathering necessary legal documentation.
    *   **System Administrator:** Responsible for maintaining the availability of the digital onboarding platform and configuring the rules for the automated risk decision engine.
    *   **Back-Office Operations:** Responsible for final quality assurance (QA) on document scans and archiving physical files.

    ### 5. System and Tool Support

    *   **Core Banking System (e.g., Fiserv, Jack Henry):** The central database where the account is actually created and the "system of record" for balances and customer data.
    *   **Digital Onboarding Platform (e.g., nCino, Backbase):** The front-end interface that customers or bankers use to input data, which orchestrates the workflow.
    *   **Identity Verification Service (e.g., Jumio, Onfido):** A tool integrated via API to perform ID document authentication, facial comparison, and liveness checks.
    *   **Watchlist Screening Tool (e.g., LexisNexis, World-Check):** Used to screen names against global sanctions lists, terrorist watchlists, and adverse media.
    *   **e-Signature Platform (e.g., DocuSign, Adobe Sign):** Enables the legal capturing of consent and signatures without physical paper.

    ### 6. Risks and Control Points

    *   **Risk Point 1: Identity Theft / Synthetic Fraud:** A bad actor uses stolen or fabricated details to open an account.
        *   *Control:* Implementation of multi-factor authentication (MFA) during signup and biometric liveness detection to ensure the applicant is physically present.
    *   **Risk Point 2: Money Laundering (Placement Phase):** Criminals open accounts to deposit illicit cash.
        *   *Control:* Strict Customer Due Diligence (CDD) and Enhanced Due Diligence (EDD) for high-risk customers; automated transaction monitoring thresholds set upon account opening.
    *   **Risk Point 3: Data Entry Errors:** Incorrect typing of SSN or Name by a banker results in failed tax reporting.
        *   *Control:* Automated validation rules (e.g., SSN format check) and "four-eyes" review (secondary approval) for manual business account entries.
    *   **Risk Point 4: Regulatory Non-Compliance:** Failure to provide required disclosures (e.g., Truth in Savings Act).
        *   *Control:* System-enforced workflows that prevent account finalization until the customer has acknowledged receipt of disclosures via a checkbox or signature.

    ### 7. Typical Metrics and Measurement Methods

    *   **Metric 1: Average Handling Time (AHT) / Cycle Time:** The elapsed time from the start of the application to the generation of the account number.
        *   *Measurement:* Timestamp of "Start Application" minus Timestamp of "Account Created" in system logs.
    *   **Metric 2: Abandonment Rate:** The percentage of potential customers who start the digital application but drop out before completion.
        *   *Measurement:* (Number of incomplete applications / Total applications started) * 100.
    *   **Metric 3: Auto-Approval Rate:** The percentage of digital applications approved by the system without human intervention.
        *   *Measurement:* (Number of auto-approved applications / Total digital applications) * 100.
    *   **Metric 4: First Time Yield (FTY):** The percentage of account applications that are processed correctly the first time without being returned for missing documents or errors.
        *   *Measurement:* (Applications with 0 rework loops / Total applications).

    ### 8. Best Practices or Improvement Suggestions

    *   **Suggestion 1: Implement "Save and Resume" Functionality:** In digital channels, allow customers to save their progress and resume later across different devices (omnichannel experience) to reduce abandonment rates.
    *   **Suggestion 2: Integration of Government Data APIs:** Instead of manual entry or OCR reliance, integrate directly with government tax or motor vehicle databases (where legally permissible) to pull verified data, reducing fraud and entry errors.
    *   **Suggestion 3: Dynamic Questioning:** Configure the onboarding form to show/hide questions based on previous answers (e.g., only ask for "Source of Wealth" if the initial deposit exceeds a certain threshold) to minimize friction for low-risk customers.
    *   **Suggestion 4: Pre-fill for Existing Customers:** If a customer already has a credit card with the bank and wants to open a checking account, pre-fill all known data to shorten the process to just a few clicks.

    '''
    sample_schema = '''
        <schema>
        <Existing_fields>
            application_id - string, unique identifier for the account opening application (e.g., "APP_2025041812345").
            application_channel - enum, channel through which the application was submitted (e.g., "in_branch", "mobile_app", "web_portal").
            business_type - enum, type of account opening business (e.g., "retail_individual", "digital_remote", "corporate_business").
            application_start_timestamp - datetime, timestamp when the application was initiated (e.g., "2025-04-18T10:15:30Z").
            customer_type - enum, classification of customer (e.g., "individual", "sole_proprietorship", "llc", "corporation").
            account_type_requested - enum, type of account requested (e.g., "checking", "savings", "student_checking", "high_yield_savings", "business_checking").
            applicant_full_name - string, full legal name of the applicant (e.g., "John Michael Smith").
            applicant_date_of_birth - date, date of birth of the applicant (e.g., "1985-06-15").
            applicant_ssn_tax_id - string, Social Security Number or Tax ID (encrypted/masked) (e.g., "XXX-XX-1234").
            applicant_email - string, email address of the applicant (e.g., "john.smith@email.com").
            applicant_phone - string, phone number of the applicant (e.g., "+1-555-123-4567").
            applicant_residential_address - string, current residential address (e.g., "123 Main St, Apt 4B, New York, NY 10001").
            applicant_employment_status - enum, employment status (e.g., "employed", "self_employed", "unemployed", "student", "retired").
            applicant_annual_income - integer, annual income in USD (e.g., 65000).
            id_document_type - enum, type of identification document provided (e.g., "passport", "drivers_license", "state_id").
            id_document_number - string, identification document number (e.g., "DL123456789").
            id_document_expiry_date - date, expiration date of the ID document (e.g., "2028-06-15").
            id_document_issuing_country - string, country that issued the ID (e.g., "US").
            proof_of_address_type - enum, type of proof of address document (e.g., "utility_bill", "bank_statement", "lease_agreement").
            initial_deposit_amount - float, amount of initial deposit in USD (e.g., 500.00).
            initial_deposit_method - enum, method of initial deposit (e.g., "cash", "check", "external_transfer", "debit_card").
            external_bank_account_linked - boolean, whether an external bank account was linked (e.g., True).
            otp_verification_completed - boolean, whether One-Time Password verification was completed (e.g., True).
            id_ocr_extracted_data - JSON, data extracted from ID via OCR (e.g., {"name": "John Michael Smith", "dob": "1985-06-15", "id_number": "DL123456789", "address": "123 Main St, New York, NY"}).
            biometric_liveness_check_completed - boolean, whether biometric liveness check was performed (e.g., True).
            facial_match_confidence_score - float, confidence score of facial comparison (0-100) (e.g., 96.5).
            business_legal_name - string, legal name of the business entity (for corporate accounts) (e.g., "Acme Technologies LLC").
            business_ein - string, Employer Identification Number (for corporate accounts) (e.g., "12-3456789").
            business_structure - enum, legal structure of the business (e.g., "llc", "corporation", "sole_proprietorship", "partnership").
            business_industry - string, industry classification of the business (e.g., "technology_services").
            business_registration_date - date, date when business was registered (e.g., "2020-03-15").
            business_monthly_transaction_volume_estimate - integer, estimated monthly transaction volume in USD (e.g., 150000).
            beneficial_owners - list[JSON], list of beneficial owners with 25%+ ownership (e.g., [{"name": "Jane Doe", "ownership_percentage": 60, "ssn": "XXX-XX-5678", "dob": "1978-11-20"}]).
            authorized_signers - list[JSON], list of authorized signers for the account (e.g., [{"name": "John Doe", "title": "CEO", "signature_captured": true}]).
            documents_submitted - list[JSON], list of all documents submitted (e.g., [{"doc_type": "articles_of_incorporation", "file_path": "/docs/APP_123_articles.pdf", "upload_timestamp": "2025-04-18T10:30:00Z"}]).
            banker_id - string, identifier of the banker/employee handling the application (e.g., "EMP_8765").
            branch_code - string, code of the branch where application was submitted (e.g., "BR_NYC_001").
            customer_consent_forms_signed - list[string], list of consent forms signed (e.g., ["signature_card", "terms_and_conditions", "truth_in_savings_disclosure", "privacy_notice"]).
            esignature_timestamp - datetime, timestamp of electronic signature (e.g., "2025-04-18T10:45:22Z").
            esignature_ip_address - string, IP address from which e-signature was captured (e.g., "192.168.1.100").
            marketing_opt_in - boolean, whether customer opted in for marketing communications (e.g., False).
            referral_source - enum, how customer learned about the bank (e.g., "online_search", "friend_referral", "branch_walk_in", "advertisement").
            product_bundle_offered - string, any product bundle offered during account opening (e.g., "premium_checking_with_rewards").
            cross_sell_products_accepted - list[string], additional products customer accepted (e.g., ["debit_card", "online_banking", "mobile_app"]).
            application_data_entry_method - enum, how data was entered (e.g., "manual_banker_entry", "ocr_auto_fill", "customer_self_service").
        </Existing_fields>
        
        <Field_to_be_obtained>
            cip_verification_status - enum, result of Customer Identification Program verification (e.g., "verified", "failed", "pending_manual_review").
            watchlist_screening_result - enum, result of OFAC/sanctions screening (e.g., "clear", "potential_match", "confirmed_match").
            chexsystems_score - integer, ChexSystems risk score (300-899 scale) (e.g., 625).
            credit_bureau_validation_status - enum, result of credit bureau identity validation (e.g., "verified", "not_verified", "insufficient_data").
            fraud_risk_score - float, automated fraud risk score (0-100, higher = riskier) (e.g., 15.3).
            auto_decision_result - enum, automated decision engine result (e.g., "auto_approved", "auto_declined", "refer_for_manual_review").
            manual_review_required - boolean, whether manual compliance review is required (e.g., False).
            manual_review_reason - string, reason for manual review if applicable (e.g., "High-risk industry flagged").
            compliance_officer_id - string, identifier of compliance officer who reviewed (e.g., "COMP_4521").
            compliance_review_timestamp - datetime, timestamp of compliance review completion (e.g., "2025-04-18T14:22:15Z").
            compliance_review_notes - string, notes from compliance review (e.g., "Verified source of funds documentation. Approved.").
            due_diligence_level - enum, level of due diligence applied (e.g., "standard_cdd", "enhanced_edd", "simplified_dd").
            final_approval_status - enum, final approval decision (e.g., "approved", "declined", "conditional_approval").
            decline_reason - string, reason for decline if applicable (e.g., "Failed identity verification").
            account_number_generated - string, generated bank account number (e.g., "1234567890").
            routing_number - string, bank routing number assigned (e.g., "021000021").
            account_creation_timestamp - datetime, timestamp when account was created in core banking system (e.g., "2025-04-18T14:30:45Z").
            total_processing_time_minutes - integer, total time from application start to account creation in minutes (e.g., 18).
            first_time_yield - boolean, whether application was processed correctly without rework (e.g., True).
            rework_count - integer, number of times application was returned for corrections (e.g., 0).
            welcome_kit_issued - boolean, whether welcome kit was provided to customer (e.g., True).
            debit_card_number_issued - string, debit card number issued (masked) (e.g., "XXXX-XXXX-XXXX-1234").
            online_banking_credentials_created - boolean, whether online banking access was set up (e.g., True). 
            temporary_checks_issued - boolean, whether temporary checks were provided (e.g., False).  
            account_funding_completed - boolean, whether initial deposit was successfully processed (e.g., True).    
            account_activation_status - enum, current status of the account (e.g., "active", "pending_funding", "dormant", "closed").
            application_abandoned - boolean, whether the application was started but not completed (e.g., False).
            abandonment_stage - string, stage at which application was abandoned if applicable (e.g., "ID_verification").
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
    yaml_file = "/Account/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Dataset_Generation.yaml"
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
