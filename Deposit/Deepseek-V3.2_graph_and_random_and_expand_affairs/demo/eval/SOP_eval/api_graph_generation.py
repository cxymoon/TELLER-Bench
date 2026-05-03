import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI
def load_yaml_prompt(file_path, sop_title, api_code, additional_context):
    """读取YAML并替换占位符"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    text = data["prompts"]["text"]
    text = text.replace("{{sop_title}}", sop_title)
    text = text.replace("{{api_code}}", api_code)
    text = text.replace("{{additional_context}}", additional_context)
    return text


def main():
    # === 输入参数 ===
    sop_title = "Banking Deposit Business Handling Process"
    api_code = '''

        # ==================== API FUNCTIONS ====================
        def get_transaction_data(transaction_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve transaction data by transaction_id"""
            for data in TRANSACTION_DATA:
                if data.get('transaction_id') == transaction_id:
                    return data
            return None

        def update_transaction_data(transaction_id: str, updates: Dict[str, Any]) -> bool:
            """Update transaction data for a given transaction_id"""
            for data in TRANSACTION_DATA:
                if data.get('transaction_id') == transaction_id:
                    data.update(updates)
                    return True
            return False


        # ==================== API函数实现 ====================
        def KYC_Verification_API(
            transaction_id: str,
            customer_id: str,
            customer_name: str,
            customer_id_type: str,
            customer_id_number: str,
            customer_address: str,
            customer_tax_id: str,
            customer_phone: str,
            customer_email: str
        ) -> Dict[str, Any]:
            """Executes comprehensive Know Your Customer identity verification workflows"""

        def AML_Sanctions_Screening_API(
            transaction_id: str,
            customer_id: str,
            customer_name: str,
            customer_address: str,
            customer_tax_id: str,
            kyc_verification_status: str,
            deposit_amount: float,
            deposit_channel: str,
            branch_code: str,
            funding_source_type: str
        ) -> Dict[str, Any]:
            """Performs automated Anti-Money Laundering screening"""

        def CTR_Evaluation_API(
            transaction_id: str,
            customer_id: str,
            deposit_amount: float,
            funding_source_type: str,
            transaction_timestamp: str,
            ctr_threshold_amount: float
        ) -> Dict[str, Any]:
            """Executes real-time Currency Transaction Report filing obligation evaluation"""

        def Currency_Validation_API(
            transaction_id: str,
            funding_source_type: str,
            deposit_amount: float,
            cash_denomination_breakdown: dict,
            cash_counter_machine_id: str
        ) -> Dict[str, Any]:
            """Interfaces with certified currency discriminator hardware"""

        def Fraud_Detection_Scoring_API(
            transaction_id: str,
            customer_id: str,
            account_number: str,
            deposit_amount: float,
            deposit_channel: str,
            deposit_type: str,
            branch_code: str,
            transaction_timestamp: str
        ) -> Dict[str, Any]:
            """Executes real-time behavioral analytics fraud detection"""

        def Supervisor_Approval_Routing_API(
            transaction_id: str,
            deposit_amount: float,
            teller_id: str,
            kyc_verification_status: str,
            aml_screening_result: str,
            aml_risk_score: int,
            fraud_detection_score: float,
            counterfeit_detection_result: str,
            ctr_filing_required: bool,
            transaction_notes: str
        ) -> Dict[str, Any]:
            """Evaluates multiple conditional criteria for supervisor approval"""


        def Term_Deposit_Processing_API(
            transaction_id: str,
            deposit_type: str,
            deposit_amount: float,
            term_length_months: int,
            transaction_timestamp: str,
            interest_rate_apy: float,
            early_withdrawal_penalty_rate: float
        ) -> Dict[str, Any]:
            """Executes specialized term deposit product processing logic"""

        def Compliance_Aggregation_API(
            transaction_id: str,
            kyc_verification_status: str,
            aml_screening_result: str,
            aml_risk_score: int,
            counterfeit_detection_result: str,
            fraud_detection_score: float,
            supervisor_approval_required: bool,
            supervisor_id: str,
            supervisor_approval_timestamp: str,
            ctr_filing_required: bool
        ) -> Dict[str, Any]:
            """Performs comprehensive compliance verification aggregation"""

        def Account_Ledger_Posting_API(
            transaction_id: str,
            account_number: str,
            deposit_amount: float,
            previous_account_balance: float,
            funding_source_type: str,
            funding_source_account: str,
            branch_code: str,
            account_type: str,
            transaction_timestamp: str,
            compliance_check_passed: bool,
            transaction_approved: bool,
            cd_certificate_number: str,
            debit_card_issued: bool,
            online_banking_enabled: bool,
            mobile_app_enabled: bool,
            checkbook_ordered: bool
        ) -> Dict[str, Any]:
            """Executes atomic transaction commit operation"""

        def Receipt_Generation_API(
            transaction_id: str,
            customer_name: str,
            account_number: str,
            deposit_amount: float,
            new_account_balance: float,
            transaction_reference_id: str,
            transaction_timestamp: str,
            branch_code: str,
            receipt_delivery_method: str,
            customer_email: str,
            customer_phone: str,
            cd_certificate_number: str,
            maturity_date: str,
            interest_rate_apy: float,
            calculated_interest_at_maturity: float,
            transaction_notes: str,
            transaction_approved: bool,
            minimum_balance_requirement: float,
            monthly_maintenance_fee: float,
            early_withdrawal_penalty_rate: float,
            final_transaction_fee: float
        ) -> Dict[str, Any]:
            """Generates comprehensive transaction documentation"""

        def Audit_Trail_Recording_API(
            transaction_id: str,
            customer_id: str,
            account_number: str,
            transaction_timestamp: str,
            deposit_type: str,
            deposit_amount: float,
            teller_id: str,
            branch_code: str,
            kyc_verification_status: str,
            kyc_verification_timestamp: str,
            aml_screening_result: str,
            aml_risk_score: int,
            sanctions_screening_timestamp: str,
            fraud_detection_score: float,
            counterfeit_detection_result: str,
            supervisor_approval_required: bool,
            supervisor_id: str,
            supervisor_approval_timestamp: str,
            compliance_check_passed: bool,
            transaction_approved: bool,
            approval_rejection_reason: str,
            new_account_balance: float,
            transaction_reference_id: str,
            account_status: str,
            processing_time_seconds: int,
            receipt_number: str,
            previous_account_balance: float,
            ctr_filing_required: bool,
            cash_drawer_total_after_transaction: float
        ) -> Dict[str, Any]:
            """Creates comprehensive immutable audit trail documentation"""


        def DocumentImageResolutionValidation(application_id: str, submitted_documents: List[str]) -> Dict[str, Any]:
            """Document image resolution validation API"""

        def MaterialIntegrityScoreCalculation(application_id: str, submitted_documents: List[str], 
                                            image_clarity_scores: Dict[str, float], 
                                            text_recognition_accuracy: Dict[str, float],
                                            loan_type: str, customer_occupation_type: str) -> Dict[str, Any]:
            """Material integrity score calculation API"""

        def IdentityAuthenticationAndFaceMatching(application_id: str, customer_id: str,
                                                id_card_number: str, id_card_name: str,
                                                id_card_images: Dict[str, str], 
                                                face_photo: str) -> Dict[str, Any]:
            """Identity authentication and face matching API"""

        def FraudPreventionValidation(application_id: str, customer_id: str, application_source: str,
                                    identity_authentication_result: bool, face_matching_passed: bool,
                                    device_id: Optional[str] = None, ip_location: Optional[str] = None,
                                    gps_location: Optional[str] = None) -> Dict[str, Any]:
            """Fraud prevention validation API"""

        def ApplicationDataConsistencyCheck(application_id: str, material_integrity_score: float,
                                        requested_amount: float, requested_term: int,
                                        loan_type: str, customer_age_years: int, 
                                        customer_occupation_type: str, customer_monthly_salary: float,
                                        purpose_receipt_amount: Optional[float] = None) -> Dict[str, Any]:
            """Application data consistency check API"""

        def CreditReportRetrievalAndAnalysis(application_id: str, customer_id: str,
                                            identity_authentication_result: bool, fraud_prevention_passed: bool,
                                            credit_authorization_confirmed: bool,
                                            customer_id_number: str, customer_full_name: str) -> Dict[str, Any]:
            """Credit report retrieval and analysis API"""

        def InternalExternalDataCrossValidation(application_id: str, customer_id: str,
                                            customer_occupation_type: str, customer_monthly_salary: float,
                                            customer_current_debt: float, internal_account_balance: float,
                                            internal_history_months: int,
                                            total_loan_outstanding: float,
                                            total_creditcard_balance: float) -> Dict[str, Any]:
            """Internal and external data cross-validation API"""

        def DebtIncomeRatioPreciseCalculation(application_id: str, verified_monthly_salary: float,
                                            verified_current_debt: float, total_loan_outstanding: float,
                                            total_creditcard_balance: float, requested_amount: float,
                                            requested_term: int, 
                                            estimated_rate: Optional[float] = None) -> Dict[str, Any]:
            """Debt-to-income ratio precise calculation API"""

        def CreditScoreAndRiskGradeAssessment(application_id: str, customer_age_years: int,
                                            customer_occupation_type: str, credit_history_length: int,
                                            past_overdue_count: int, active_credit_accounts: int,
                                            credit_query_count_6m: int, debt_income_ratio: float,
                                            internal_account_balance: float, 
                                            internal_history_months: int,
                                            has_negative_records: bool, overdue_status: str,
                                            policy_version: str) -> Dict[str, Any]:
            """Credit score and risk grade assessment API"""

        def TelephoneVerificationRecording(application_id: str, assigned_reviewer_id: str, customer_id: str,
                                        verification_timestamp: str, verification_result: bool,
                                        customer_phone: Optional[str] = None,
                                        questions_asked: Optional[List[str]] = None,
                                        responses_consistent: Optional[bool] = None,
                                        employer_verification_done: Optional[bool] = None,
                                        employer_phone: Optional[str] = None,
                                        employer_confirmation: Optional[bool] = None,
                                        notes: Optional[str] = None,
                                        recording_id: Optional[str] = None) -> Dict[str, Any]:
            """Telephone verification recording API"""

        def ReviewReportDocumentGeneration(application_id: str, assigned_reviewer_id: str,
                                        material_integrity_score: float, identity_authentication_result: bool,
                                        fraud_prevention_passed: bool, credit_score_value: int, risk_grade: str,
                                        debt_income_ratio: float, salary_verification_result: bool,
                                        verification_result: bool, reviewer_suggestion: str,
                                        suggested_amount: Optional[float] = None,
                                        suggestion_rationale: Optional[str] = None) -> Dict[str, Any]:
            """Review report document generation API"""

        def ApprovalWorkflowRouting(application_id: str, requested_amount: float, credit_score_value: int,
                                risk_grade: str, debt_income_ratio: float, past_overdue_count: int,
                                fraud_prevention_passed: bool, reviewer_suggestion: str) -> Dict[str, Any]:
            """Approval workflow routing API"""

        def RiskOrientedPricingCalculation(application_id: str, credit_score_value: int, risk_grade: str,
                                        internal_account_balance: float, internal_history_months: int,
                                        current_benchmark_rate: float, requested_term: int) -> Dict[str, Any]:
            """Risk-oriented pricing calculation API"""

        def ApprovalDecisionImplementation(application_id: str, assigned_approver_id: str, auto_decision_flag: bool,
                                        suggested_rate: float, requested_amount: float,
                                        requested_term: int, decision: str,
                                        approved_amount: Optional[float] = None,
                                        approved_rate: Optional[float] = None,
                                        approved_term: Optional[int] = None,
                                        payment_method: Optional[str] = None,
                                        decision_conditions: Optional[List[str]] = None,
                                        decision_reason: str = "") -> Dict[str, Any]:
            """Approval decision implementation API"""

        def MonthlyPaymentAmountCalculation(application_id: str, approved_amount: float,
                                        approved_rate: float, approved_term: int,
                                        payment_method: str) -> Dict[str, Any]:
            """Monthly payment amount calculation API"""

        def DigitalContractCreation(application_id: str, customer_id: str, approved_amount: float,
                                approved_rate: float, approved_term: int,
                                payment_method: str, monthly_payment: float,
                                payment_schedule: List[Dict], loan_type: str,
                                customer_full_name: str, customer_id_number: str,
                                customer_residence: str, customer_phone: str) -> Dict[str, Any]:
            """Digital contract creation API"""

        def ContractSignatureValidation(application_id: str, contract_id: str,
                                        signing_timestamp: str, signing_method: str,
                                        contract_hash: str, digital_certificate: Optional[str] = None,
                                        customer_signature: Optional[str] = None,
                                        video_recording_id: Optional[str] = None) -> Dict[str, Any]:
            """Contract signature validation API"""

        def PreDisbursementFinalCheck(application_id: str, customer_id: str, contract_id: str,
                                    signature_verified: bool, approved_amount: float,
                                    target_account: str, disbursement_type: Optional[str] = None,
                                    payee_info: Optional[Dict] = None) -> Dict[str, Any]:
            """Pre-disbursement final check API"""

        def FundTransferExecution(application_id: str, contract_id: str, compliance_verified: bool,
                                approved_amount: float, target_account: str,
                                transfer_timestamp: str, transfer_method: str,
                                operator_user_id: str) -> Dict[str, Any]:
            """Fund transfer execution API"""

        def ApplicationStatusUpdateAndAlert(application_id: str, submission_timestamp: str,
                                        decision: str, decision_timestamp: str,
                                        review_begin_timestamp: str,
                                        transfer_timestamp: Optional[str] = None,
                                        transfer_status: Optional[str] = None) -> Dict[str, Any]:
            """Application status update and alert API"""

        def adopt_a_pet(location: str, animal_type: str) -> dict:

        def get_axolotl_info(fact_id: int) -> dict:

        def get_cat_fact() -> dict:

        def get_cat_image(tag: Optional[str] = None) -> dict:

        def search_anime(query: str) -> dict:

        def get_anime_info(anime_id: int) -> dict:

        def get_anime_list(status: str) -> dict:

        def get_anime_quote() -> dict:

        def check_ip_reputation(ip_address: str) -> dict:

        def check_threat_intelligence(indicator: str) -> dict:

        def analyze_malware(file_hash: str) -> dict:

        def check_safe_browsing(url: str) -> dict:

        def generate_discord_image(user_id: str, effect: str) -> dict:

        def get_artwork(artwork_id: int) -> dict:

        def generate_color_scheme() -> dict:

        def get_color_palette(palette_id: int) -> dict:

        def authenticate_user(username: str, password: str) -> dict:

        def generate_otp(phone_number: str) -> dict:

        def create_user(username: str, email: str) -> dict:

        def passwordless_login(email: str) -> dict:

        def query_blockchain(chain: str, query: str) -> dict:

        def get_chainlink_data(feed_id: str) -> dict:

        def anchor_to_blockchain(data: str) -> dict:

        def get_blockchain_data(chain: str, address: str) -> dict:

        def get_bhagavad_gita_verse(chapter: int, verse: int) -> dict:

        def get_bhagavad_gita_telugu(chapter: int, verse: int) -> dict:

        def get_bible_verse(reference: str) -> dict:

        def get_persian_poetry(poet: str) -> dict:

        def manage_dashboard(dashboard_id: int, action: str) -> dict:

        def search_charity(name: str) -> dict:

        def get_company_logo(domain: str) -> dict:

        def search_domains(keyword: str) -> dict:

        def get_public_holidays(country: str, year: int) -> dict:

        def get_holidays_calendarific(country: str, year: int) -> dict:

        def get_national_holiday(date: str) -> dict:

        def get_church_calendar(date: str) -> dict:

        def upload_file_anon(file_content: str) -> dict:

        def upload_to_box(file_name: str, file_content: str) -> dict:

        def upload_to_fileio(file_content: str) -> dict:

        def check_azure_health(resource_id: str) -> dict:

        def trigger_bitrise_build(app_id: str, workflow: str) -> dict:

        def trigger_buddy_pipeline(pipeline_id: int) -> dict:

        def trigger_circleci_build(project: str, branch: str) -> dict:

        def get_crypto_rate(crypto: str, fiat: str) -> dict:

        def get_binance_price(symbol: str) -> dict:

        def get_bitcambio_assets() -> dict:

        def get_bitcoin_average(symbol: str) -> dict:

        def convert_currency(from_currency: str, to_currency: str, amount: float) -> dict:

        def get_exchange_rate(base: str, target: str) -> dict:

        def get_currency_rate(base: str, target: str) -> dict:

        def get_currency_freaks_rate(base: str) -> dict:

        def postman_echo_test(method: str, data: dict) -> dict:

        def check_profanity(text: str) -> dict:

        def autocomplete_address(partial: str) -> dict:

        def extract_address(text: str) -> dict:

        def get_pull_requests(year: int) -> dict:

        def take_screenshot(url: str) -> dict:

        def estimate_age(name: str) -> dict:

        def get_api_gratis(service: str) -> dict:

        def get_chinese_character(character: str) -> dict:

        def search_chinese_text(query: str) -> dict:

        def lookup_collins(word: str, lang_from: str, lang_to: str) -> dict:

        def get_free_dictionary(word: str) -> dict:

        def manage_asana_task(task_id: str, action: str) -> dict:

        def manage_clickup_task(task_id: str, action: str) -> dict:

        def get_clockify_time_entries(user_id: str) -> dict:

        def convert_file(input_format: str, output_format: str, file_content: str) -> dict:

        def validate_email_mailboxlayer(email: str) -> dict:

        def validate_email_address(email: str) -> dict:

        def validate_cloudmersive(value: str, validation_type: str) -> dict:

        def check_disposable_email(email: str) -> dict:

        def get_chuck_norris_joke() -> dict:

        def get_corporate_buzzword() -> dict:

        def get_excuse(category: str) -> dict:

        def get_fun_fact() -> dict:

        def get_pollen_forecast(lat: float, lon: float) -> dict:

        def calculate_carbon_emissions(activity: str, amount: float) -> dict:

        def get_emission_factor(activity: str) -> dict:

        def offset_carbon(kg_co2: float) -> dict:

        def search_events_eventbrite(query: str, location: str) -> dict:

        def search_events_seatgeek(query: str) -> dict:

        def search_events_ticketmaster(keyword: str, city: str) -> dict:

        def validate_vat(vat_number: str, country_code: str) -> dict:

        def get_insider_trading(ticker: str) -> dict:

        def get_alpaca_market_data(symbol: str) -> dict:

        def get_alpha_vantage_stock(symbol: str) -> dict:

        def get_marketstack_data(symbol: str) -> dict:

        def get_banco_brasil_transactions(account_id: str) -> dict:

        def validate_iban(iban: str) -> dict:

        def create_billplz_bill(amount: float, description: str) -> dict:

        def get_bin_info(bin_number: str) -> dict:

        def generate_boleto(amount: float, payer_name: str) -> dict:

        def get_citi_account(account_id: str) -> dict:

        def get_econdb_data(dataset: str, country: str) -> dict:

        def get_fed_treasury_data(series: str) -> dict:

        def get_finage_stock(symbol: str) -> dict:

        def get_financial_modeling_prep(symbol: str) -> dict:

        def get_finnhub_quote(symbol: str) -> dict:

        def get_fred_data(series_id: str) -> dict:

        def front_accounting_api(operation: str) -> dict:

        def get_hotstoks_data(ticker: str) -> dict:

        def get_iex_cloud_quote(symbol: str) -> dict:

        def get_ig_market_data(epic: str) -> dict:

        def get_indian_mutual_fund(scheme_code: str) -> dict:

        def get_intrinio_data(ticker: str, data_point: str) -> dict:

        def create_klarna_session(amount: int, currency: str) -> dict:

        def create_mercadopago_payment(amount: float, description: str) -> dict:

        def connect_mono_account(account_id: str) -> dict:

        def create_moov_transfer(amount: float, destination: str) -> dict:

        def connect_nordigen_bank(bank_id: str) -> dict:

        def get_openfigi(id_value: str, id_type: str) -> dict:

        def connect_plaid_account(public_token: str) -> dict:

        def get_polygon_data(ticker: str, date: str) -> dict:

        def optimize_portfolio(tickers: list, weights: list) -> dict:

        def get_ifsc_details(ifsc_code: str) -> dict:

        def get_realtime_finance(symbol: str) -> dict:

        def get_sec_edgar(cik: str, filing_type: str) -> dict:

        def smartapi_trade(symbol: str, action: str, quantity: int) -> dict:

        def get_stockdata(ticker: str) -> dict:

        def get_styvio_data(symbol: str) -> dict:

        def validate_tax_number(tax_number: str, country: str) -> dict:

        def get_tradier_quote(symbol: str) -> dict:

        def get_twelve_data(symbol: str, interval: str) -> dict:

        def get_wallstreetbets_sentiment(ticker: str) -> dict:

        def get_yahoo_finance(symbol: str) -> dict:

        def get_ynab_budget(budget_id: str) -> dict:

        def get_zoho_books_invoice(invoice_id: str) -> dict:

        def get_bacon_image(width: int, height: int) -> dict:

        def get_food_product(barcode: str) -> dict:

        def get_brewery(brewery_id: int) -> dict:

        def get_food_facts(barcode: str) -> dict:

        def get_aoe2_resource(resource_id: int) -> dict:

        def get_amiibo(amiibo_id: str) -> dict:

        def get_autochess_data(player_id: str) -> dict:

        def get_barter_game(game_id: int) -> dict:

        def geolocate_ip(ip_address: str) -> dict:

        def process_grass_gis(operation: str, parameters: dict) -> dict:

        def get_administrative_division(country: str, level: int) -> dict:

        def geocode_address_france(address: str) -> dict:

        def get_bc_law(law_id: str) -> dict:

        def get_brazil_data(dataset: str) -> dict:

        def get_brazil_company(cnpj: str) -> dict:

        def get_federal_code(agency: str) -> dict:

        def get_cms_data(dataset: str) -> dict:

        def get_covid_data(country: str) -> dict:

        def get_covid_datenhub(region: str) -> dict:

        def get_government_response(country: str) -> dict:

        def search_job_adzuna(query: str, location: str) -> dict:

        def search_job_arbeitnow(query: str) -> dict:

        def search_job_arbeitsamt(occupation: str, region: str) -> dict:

        def search_job_careerjet(query: str, location: str) -> dict:

        def process_thai_ai(text: str, operation: str) -> dict:

        def analyze_image_cloudmersive(image_url: str, analysis_type: str) -> dict:

        def process_dialogflow(text: str, session_id: str) -> dict:

        def filter_text_exude(text: str, operation: str) -> dict:

        def get_7digital_track(track_id: int) -> dict:

        def master_audio(audio_url: str) -> dict:

        def get_audiomack_track(track_id: str) -> dict:

        def search_bandcamp(query: str) -> dict:

        def get_news_mediastack(category: str, country: str) -> dict:

        def search_news_ap(query: str) -> dict:

        def search_historical_newspapers(query: str, year: int) -> dict:

        def get_currents_news(category: str, language: str) -> dict:

        def get_api_setu_service(service: str) -> dict:

        def get_black_history_fact(fact_id: int) -> dict:

        def get_telegram_bot(bot_username: str) -> dict:

        def get_carto_prediction(lat: float, lon: float, data_type: str) -> dict:

        def get_countly_analytics(app_id: str, metric: str) -> dict:

        def search_creative_commons(query: str, license_type: str) -> dict:

        def search_datamuse(word: str, relation: str) -> dict:

        def query_metabase(query_id: int) -> dict:

        def search_epo_patent(query: str) -> dict:

        def search_patentsview(query: str) -> dict:

        def search_tipo_patent(query: str) -> dict:

        def search_uspto_patent(query: str) -> dict:

        def get_advice() -> dict:

        def get_biriyani_image() -> dict:

        def get_dev_article(article_id: int) -> dict:

        def get_dictum_quote(quote_id: int) -> dict:

        def validate_phone_number(phone: str, country_code: str) -> dict:

        def validate_numverify(phone: str) -> dict:

        def validate_phone_cloudmersive(phone: str) -> dict:

        def get_phone_specs(brand: str, model: str) -> dict:

        def generate_image_bruzu(text: str, template: str) -> dict:

        def optimize_image_cheetaho(image_url: str, width: int, height: int) -> dict:

        def manipulate_image_dagpi(image_url: str, effect: str) -> dict:

        def generate_image_duply(template_id: str, data: dict) -> dict:

        def get_codeforces_user(handle: str) -> dict:

        def compile_code_hackerearth(code: str, language: str) -> dict:

        def get_coding_contests() -> dict:

        def generate_docs_mintlify(code: str, language: str) -> dict:

        def get_astronomy_data(object_name: str) -> dict:

        def search_arxiv(query: str, category: str) -> dict:

        def search_core(query: str) -> dict:

        def get_gbif_species(species_name: str) -> dict:

        def scan_binaryedge(target: str) -> dict:

        def get_bitwarden_vault(user_id: str) -> dict:

        def detect_bot(user_agent: str, ip: str) -> dict:

        def get_bugcrowd_submission(submission_id: str) -> dict:

        def search_bestbuy_product(query: str) -> dict:

        def search_digikey_part(part_number: str) -> dict:

        def get_dummy_product(product_id: int) -> dict:

        def search_ebay(query: str, category: str) -> dict:

        def get_4chan_board(board: str) -> dict:

        def post_to_social_ayrshare(message: str, platforms: list) -> dict:

        def get_horoscope_aztro(sign: str, day: str) -> dict:

        def get_blogger_posts(blog_id: str) -> dict:

        def get_football_info(league: str, season: int) -> dict:

        def check_symptoms(symptoms: list, age: int, gender: str) -> dict:

        def get_nba_stats(player: str) -> dict:

        def get_city_bikes(city: str) -> dict:

        def generate_bacon_ipsum(paragraphs: int) -> dict:

        def generate_avatar_dicebear(seed: str, style: str) -> dict:

        def get_random_word(count: int) -> dict:

        def generate_fake_data(schema: dict) -> dict:

        def detect_code_language(code: str) -> dict:

        def detect_language_apilayer(text: str) -> dict:

        def analyze_text_aylien(text: str, analysis_type: str) -> dict:

        def detect_language_api(text: str) -> dict:

        def track_shipment_aftership(tracking_number: str) -> dict:

        def track_correios(tracking_code: str) -> dict:

        def track_habit_pixela(user: str, graph_id: str, date: str) -> dict:

        def get_pincode_info(pincode: str) -> dict:

        def get_adsb_flight(icao24: str) -> dict:

        def get_airport_info(icao_code: str) -> dict:

        def get_ais_vessel(mmsi: str) -> dict:

        def search_flights_amadeus(origin: str, destination: str, date: str) -> dict:

        def shorten_url_cleanuri(url: str) -> dict:

        def track_link_clickmeter(link_id: str) -> dict:

        def shorten_url_drivet(url: str) -> dict:

        def shorten_url_free(url: str) -> dict:

        def get_helipad(helipad_id: str) -> dict:

        def get_kbb_vehicle(vin: str) -> dict:

        def get_mercedes_vehicle(vin: str) -> dict:

        def get_nhtsa_vehicle(vin: str) -> dict:

        def get_doctor_who_info(episode_id: int) -> dict:

        def get_next_episode(show: str) -> dict:

        def get_czech_tv_schedule(channel: str, date: str) -> dict:

        def get_dailymotion_video(video_id: str) -> dict:

        def get_weather_micro(lat: float, lon: float) -> dict:

        def get_odweather_forecast(location: str) -> dict:

        def get_oikolab_weather(lat: float, lon: float, date: str) -> dict:

        def get_open_meteo_forecast(lat: float, lon: float) -> dict:
            '''
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

    yaml_file = "/Deposit/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Bench_API_Graph_Generation.yaml"
    system_prompt = "You are a professional AI system specializing in analyzing complex business processes and deducing API dependency relationships. Your task is to select all valid APIs from the provided implemented API interfaces (which contain many redundant APIs) based on the business context, and then generate a comprehensive directed graph of API dependencies. This graph must accurately reflect the data flows and prerequisite dependencies between the APIs."



    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, api_code, additional_context)

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
        model="deepseek-r1-250528",  # 指定模型
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
