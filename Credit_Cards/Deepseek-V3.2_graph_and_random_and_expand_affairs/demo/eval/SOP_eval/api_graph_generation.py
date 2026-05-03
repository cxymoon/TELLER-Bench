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
    sop_title = "Banking Credit Card Issuance Scenario"
    api_code = '''

        # ==================== API FUNCTIONS ====================
        def get_application_by_id(application_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve application data by application_id"""

        def update_application_data(application_id: str, updates: Dict[str, Any]):
            """Update application data with new fields"""

        # ==================== API函数实现 ====================
        def ValidateApplicationData(
            application_id: str,
            application_timestamp: str,
            application_channel: str,
            business_type: str,
            product_type: str,
            applicant_id: str,
            applicant_age: int,
            applicant_employment_status: str,
            applicant_annual_income: float,
            applicant_housing_status: str,
            applicant_monthly_housing_cost: float,
            applicant_years_at_address: int,
            applicant_years_employed: int,
            business_entity_name: Optional[str] = None,
            business_tin_ein: Optional[str] = None,
            business_incorporation_date: Optional[str] = None,
            business_annual_revenue: Optional[float] = None,
            business_industry_code: Optional[str] = None,
            guarantor_personal_income: Optional[float] = None,
            security_deposit_amount: Optional[float] = None,
            deposit_verification_status: Optional[str] = None,
            deposit_account_id: Optional[str] = None
        ) -> Dict[str, Any]:
            """Performs comprehensive data quality validation on credit card applications"""

        def VerifyIdentity(
            application_id: str,
            applicant_id: str,
            applicant_age: int,
            device_fingerprint_id: str,
            ip_address_location: str,
            fraud_alert_flags: List[str]
        ) -> Dict[str, Any]:
            """Executes multi-layered identity verification per USA PATRIOT Act Section 326"""

        def ScreenSanctionsWatchlist(
            application_id: str,
            applicant_id: str,
            identity_verification_method: str,
            business_entity_name: Optional[str] = None
        ) -> Dict[str, Any]:
            """Screens applicant against OFAC SDN list and consolidated sanctions lists"""

        def DetectFraudIndicators(
            application_id: str,
            watchlist_check_result: str,
            device_fingerprint_id: str,
            ip_address_location: str,
            fraud_alert_flags: List[str]
        ) -> Dict[str, Any]:
            """Analyzes application for fraud risk patterns using multi-factor detection logic"""

        def RetrieveCreditBureauData(
            application_id: str,
            applicant_id: str,
            business_type: str,
            fraud_risk_level: str,
            business_tin_ein: Optional[str] = None
        ) -> Dict[str, Any]:
            """Executes hard inquiry to credit bureaus based on business type"""

        def CalculateDebtToIncomeRatio(
            application_id: str,
            business_type: str,
            applicant_annual_income: float,
            applicant_monthly_housing_cost: float,
            total_outstanding_debt: float,
            business_annual_revenue: Optional[float] = None,
            guarantor_personal_income: Optional[float] = None
        ) -> Dict[str, Any]:
            """Computes debt-to-income ratio as primary measure of ability to service credit"""

        def EvaluateCreditScorecard(
            application_id: str,
            business_type: str,
            credit_score: int,
            credit_utilization_ratio: float,
            debt_to_income_ratio: float,
            number_of_open_accounts: int,
            number_of_delinquent_accounts: int,
            bankruptcy_flag: bool,
            applicant_years_employed: int,
            applicant_years_at_address: int,
            months_since_last_delinquency: Optional[int] = None
        ) -> Dict[str, Any]:
            """Executes proprietary credit scorecard model to predict default probability"""

        def DetermineCreditDecision(
            application_id: str,
            business_type: str,
            product_type: str,
            internal_risk_score: int,
            scorecard_reason_codes: List[str],
            watchlist_check_result: str,
            identity_verification_method: str,
            bankruptcy_flag: bool,
            fraud_alert_flags: List[str],
            applicant_annual_income: float,
            business_industry_code: Optional[str] = None
        ) -> Dict[str, Any]:
            """Applies hierarchical decision rule logic to render final credit decision"""

        def AssignCreditTerms(
            application_id: str,
            final_decision: str,
            business_type: str,
            product_type: str,
            internal_risk_score: int,
            applicant_annual_income: float,
            business_annual_revenue: Optional[float] = None,
            security_deposit_amount: Optional[float] = None,
            deposit_verification_status: Optional[str] = None
        ) -> Dict[str, Any]:
            """Calculates approved credit limit and APR for approved applications"""

        def GenerateAdverseActionNotice(
            application_id: str,
            final_decision: str,
            decline_reason_code: Optional[str],
            applicant_id: str,
            credit_bureau_source: str,
            application_timestamp: str
        ) -> Dict[str, Any]:
            """Produces FCRA and ECOA compliant adverse action notices"""

        def FinalizeApplicationOutputs(
            application_id: str,
            processing_start_timestamp: str,
            final_decision: str,
            adverse_action_letter_generated: bool,
            manual_review_required: bool,
            fraud_alert_flags: List[str],
            uploaded_documents: List[Dict[str, str]],
            approved_credit_limit: Optional[float] = None,
            assigned_apr: Optional[float] = None,
            underwriter_id: Optional[str] = None,
            deposit_verification_status: Optional[str] = None
        ) -> Dict[str, Any]:
            """Consolidates all output fields to complete application processing lifecycle"""

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
    *   **Scenario Name:** Banking Credit Card Issuance & Origination
    *   **Industry:** Financial Services / Retail Banking
    *   **Primary Purpose / Value:** To systematically evaluate applicant creditworthiness, mitigate financial risk, and issue credit facilities (cards) to qualified customers. This process balances the bank's need for asset growth and customer acquisition with the strict requirement for regulatory compliance and risk management. The ultimate value is the creation of a profitable lending relationship and a frictionless onboarding experience for the customer.

    ### 2. Business Types (Sub-businesses)
    *   **Sub-business A: Retail Credit Card Application Processing:** Handling applications from individual consumers for personal use credit cards (e.g., Rewards, Cashback, Low Interest).
    *   **Sub-business B: Small Business/Corporate Card Issuance:** Processing applications from business entities for corporate expense management or working capital.
    *   **Sub-business C: Secured Credit Card Issuance:** Processing applications for customers with limited or poor credit history, requiring a cash collateral deposit.

    ### 3. Process Description
    *Note: These processes focus strictly on the issuance phase, ending with the final decision and credit limit assignment (deterministic outputs).*

    *   **Sub-business A: Retail Credit Card Application Processing**
        1)  **Step 1: Application Intake & Data Capture:** The system captures applicant PII (Personally Identifiable Information), employment data, income details, and housing status via digital channels (web/app) or branch entry.
        2)  **Step 2: Identity Verification (KYC/CIP):** The system automatically validates the applicant's identity against government databases and internal watchlists (OFAC, AML) to prevent fraud.
        3)  **Step 3: Credit Bureau Inquiry:** The system triggers a hard inquiry to major credit bureaus (Experian, TransUnion, Equifax) to retrieve the credit score, utilization ratios, and payment history.
        4)  **Step 4: Debt-to-Income (DTI) Calculation:** The engine calculates the applicant's ability to pay by comparing declared income against debt obligations found in the credit report.
        5)  **Step 5: Automated Decisioning & Scorecard Evaluation:** The application is run through the bank’s internal credit scorecard model. It generates a final risk score (e.g., 0-1000).
        6)  **Step 6: Final Output Generation:** The system outputs a binary decision (Approve/Decline) and, if approved, calculates the specific **Credit Limit (e.g., $5,000)** and **APR (e.g., 18.99%)**.

    *   **Sub-business B: Small Business/Corporate Card Issuance**
        1)  **Step 1: Entity & Guarantor Data Collection:** Collecting legal business documents (Articles of Incorporation, TIN/EIN) and personal details of the beneficial owner/guarantor.
        2)  **Step 2: Business Verification (KYB):** Validating the existence and active status of the business entity via Secretary of State registries and commercial credit bureaus (e.g., Dun & Bradstreet).
        3)  **Step 3: Blended Credit Assessment:** Pulling both the commercial credit report of the business and the personal credit report of the guarantor.
        4)  **Step 4: Cash Flow Analysis:** Analyzing business bank statements or tax returns (if uploaded) to determine average monthly revenue and volatility.
        5)  **Step 5: Underwriting Decision:** A credit underwriter (or automated system for small limits) reviews the blended risk profile.
        6)  **Step 6: Final Output Generation:** Deterministic output of **Approved Credit Line (e.g., $50,000)** and **Cardholder Limit Distribution** (how much limit per employee card).

    *   **Sub-business C: Secured Credit Card Issuance**
        1)  **Step 1: Application & Collateral Request:** Applicant submits personal details and agrees to provide a security deposit.
        2)  **Step 2: Deposit Verification:** The system verifies the receipt of the security deposit into a holding account (locking funds).
        3)  **Step 3: Risk Hygiene Check:** Basic checks for bankruptcy or active fraud alerts (credit score is less relevant here, but "derogatory marks" are checked).
        4)  **Step 4: Ratio Configuration:** The system applies the bank's LTV (Loan-to-Value) policy (usually 100% or 90% of the deposit).
        5)  **Step 5: Final Output Generation:** Outputs the **Approved Credit Limit** strictly equal to or representing a percentage of the **Verified Deposit Amount**.

    ### 4. Key Roles and Responsibilities
    *   **Applicant (Customer):** Provides accurate personal and financial data; consents to credit checks.
    *   **Credit Underwriter:** Reviews complex or borderline applications (manual review queues) that the automated system cannot decide; validates income documentation.
    *   **Fraud Analyst:** Investigates alerts generated by the identity verification system (e.g., address mismatches, synthetic identity suspicion).
    *   **Product Manager:** Defines the scorecard criteria, APR ranges, and credit limit strategies for specific card products.
    *   **Compliance Officer:** Ensures the application process adheres to regulations like ECOA (Equal Credit Opportunity Act) and FCRA (Fair Credit Reporting Act).

    ### 5. System and Tool Support
    *   **LOS (Loan Origination System):** The central platform that manages the workflow from application ingestion to final decisioning.
    *   **Decision Engine (BRE - Business Rule Engine):** Executes logic (e.g., "If Credit Score < 600, then Decline") and calculates the internal scorecard.
    *   **External Data Interface (API Gateway):** Connects to credit bureaus (Experian/Equifax) and fraud databases (LexisNexis/ChexSystems).
    *   **Document Management System (DMS):** Stores uploaded proofs of income (pay stubs) or identity (driver's license images) securely.

    ### 6. Risks and Control Points
    *   **Risk Point 1: Identity Fraud (Synthetic IDs):**
        *   *Control Measure:* Multi-factor authentication (OTP) during application; Knowledge-Based Authentication (KBA); device fingerprinting analysis.
    *   **Risk Point 2: Inaccurate Credit Decisioning (Bad Debt):**
        *   *Control Measure:* Regular back-testing of credit scorecards against actual default rates; automated DTI threshold enforcement.
    *   **Risk Point 3: Regulatory Non-Compliance (Fair Lending violations):**
        *   *Control Measure:* Automated auditing of decline reasons to ensure no decisions are based on protected class attributes (race, gender, etc.); generation of Adverse Action Letters.
    *   **Risk Point 4: Data Privacy Leakage:**
        *   *Control Measure:* Encryption of PII at rest and in transit; strict role-based access control (RBAC) within the LOS.

    ### 7. Typical Metrics and Measurement Methods
    *   **Approval Rate:** (Total Approved Applications / Total Applications Received) × 100. Indicates the quality of the applicant pool and the strictness of credit policy.
    *   **Turnaround Time (TAT):** Average time elapsed from "Application Submit" to "Final Decision Output." Measured in seconds for auto-decisions and hours/days for manual review.
    *   **Cost per Acquisition (CPA):** Total marketing and processing costs divided by the number of funded/activated accounts.
    *   **Auto-Decision Rate:** Percentage of applications processed without any human intervention. High rates indicate efficient automation.

    ### 8. Best Practices or Improvement Suggestions
    *   **Suggestion 1: Implement Open Banking API Integration:** Instead of asking customers to upload PDF bank statements or enter income manually, allow them to connect their primary bank account securely to instantly verify income and cash flow, reducing fraud and manual review time.
    *   **Suggestion 2: Dynamic Friction:** Apply "lighter" KYC checks for low-risk applicants (trusted devices/IPs) and step-up verification (document scan/biometrics) only for high-risk signals to improve conversion rates.
    *   **Suggestion 3: Pre-Qualification (Soft Pull):** Allow customers to see if they are likely to be approved without impacting their credit score (Soft Inquiry) before the formal submission (Hard Inquiry), improving the user experience and the quality of final applications.

    '''

    yaml_file = "Credit_cards/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Bench_API_Graph_Generation.yaml"
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
