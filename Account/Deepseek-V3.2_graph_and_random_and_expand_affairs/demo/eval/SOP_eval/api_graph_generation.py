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
    sop_title = "Banking Account Opening Business Handling Process"
    api_code = '''

        # ==================== API FUNCTIONS ====================
        def get_application_by_id(application_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve application data by application_id"""

        def update_application_data(application_id: str, updates: Dict[str, Any]) -> bool:
            """Update application data"""

        # ==================== API函数实现 ====================
        def ValidateApplicationData(
            application_id: str,
            applicant_full_name: str,
            applicant_date_of_birth: str,
            applicant_ssn_tax_id: str,
            applicant_residential_address: str,
            applicant_email: str,
            applicant_phone: str,
            id_document_type: str,
            id_document_number: str,
            id_document_expiry_date: str,
            application_data_entry_method: str,
            id_ocr_extracted_data: Dict[str, Any],
            customer_type: str,
            business_type: str
        ) -> Dict[str, Any]:
            """Validates application data format, completeness, and internal consistency"""

        def ExecuteCIPVerification(
            application_id: str,
            applicant_full_name: str,
            applicant_date_of_birth: str,
            applicant_ssn_tax_id: str,
            applicant_residential_address: str,
            id_document_type: str,
            id_document_number: str,
            id_document_issuing_country: str,
            biometric_liveness_check_completed: bool,
            application_channel: str,
            data_consistency_score: float
        ) -> Dict[str, Any]:
            """Executes CIP verification including document authentication and biometric checks"""

        def ScreenWatchlists(
            application_id: str,
            applicant_full_name: str,
            applicant_date_of_birth: str,
            applicant_residential_address: str,
            applicant_ssn_tax_id: str,
            cip_verification_status: str,
            business_legal_name: str,
            beneficial_owners: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Performs OFAC sanctions screening and watchlist matching"""

        def AssessRiskProfile(
            application_id: str,
            applicant_full_name: str,
            applicant_ssn_tax_id: str,
            applicant_date_of_birth: str,
            applicant_residential_address: str,
            applicant_employment_status: str,
            applicant_annual_income: float,
            initial_deposit_amount: float,
            customer_type: str,
            cip_verification_status: str,
            credit_bureau_validation_status: str,
            facial_match_confidence_score: float,
            id_ocr_extracted_data: Dict[str, Any],
            data_consistency_score: float,
            application_channel: str,
            business_industry: str,
            business_monthly_transaction_volume_estimate: Optional[float]
        ) -> Dict[str, Any]:
            """Calculates comprehensive risk assessment including ChexSystems and fraud risk scoring"""

        def ExecuteAutoDecision(
            application_id: str,
            cip_verification_status: str,
            watchlist_screening_result: str,
            chexsystems_score: Optional[float],
            fraud_risk_score: float,
            credit_bureau_validation_status: str,
            facial_match_confidence_score: float,
            due_diligence_level: str,
            customer_type: str,
            application_channel: str
        ) -> Dict[str, Any]:
            """Synthesizes risk metrics to render automated approval decision"""

        def ProcessManualReview(
            application_id: str,
            auto_decision_result: str,
            manual_review_required: bool,
            manual_review_reason: str,
            watchlist_screening_result: str,
            fraud_risk_score: float,
            chexsystems_score: Optional[float],
            due_diligence_level: str,
            documents_submitted: List[Dict[str, Any]],
            beneficial_owners: List[Dict[str, Any]],
            compliance_officer_id: str
        ) -> Dict[str, Any]:
            """Facilitates compliance officer adjudication for manual review cases"""

        def CreateBankAccount(
            application_id: str,
            final_approval_status: str,
            applicant_full_name: str,
            applicant_date_of_birth: str,
            applicant_ssn_tax_id: str,
            applicant_residential_address: str,
            applicant_email: str,
            applicant_phone: str,
            account_type_requested: str,
            customer_type: str,
            branch_code: str,
            business_legal_name: str,
            business_ein: str,
            authorized_signers: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Generates new account in Core Banking System"""

        def ProcessAccountFunding(
            application_id: str,
            account_number_generated: str,
            initial_deposit_amount: float,
            initial_deposit_method: str,
            external_bank_account_linked: bool,
            account_activation_status: str
        ) -> Dict[str, Any]:
            """Executes initial deposit processing"""

        def ProvisionAncillaryProducts(
            application_id: str,
            account_number_generated: str,
            routing_number: str,
            applicant_full_name: str,
            applicant_residential_address: str,
            applicant_email: str,
            account_type_requested: str,
            cross_sell_products_accepted: List[str],
            customer_type: str
        ) -> Dict[str, Any]:
            """Coordinates issuance of ancillary banking products"""
            
        def CalculateProcessMetrics(
            application_id: str,
            application_start_timestamp: str,
            account_creation_timestamp: str,
            account_funding_completed: bool,
            final_approval_status: str,
            auto_decision_result: str,
            manual_review_required: bool,
            customer_consent_forms_signed: List[str],
            application_abandoned: bool,
            abandonment_stage: str
        ) -> Dict[str, Any]:
            """Computes process performance metrics"""

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

    yaml_file = "/Account/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Bench_API_Graph_Generation.yaml"
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
