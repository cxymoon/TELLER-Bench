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
    sop_title = "Banking Invest Process"
    api_code = '''

        # ==================== API FUNCTIONS ====================
        def get_by_transaction_id(transaction_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve data entry by transaction_id"""

        def get_by_application_id(application_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve data entry by application_id (alias for transaction_id)"""

        # ==================== API函数实现 ====================
        def ValidateExistingRiskProfile(client_id: str, transaction_timestamp: str, 
                                        previous_risk_profile: Optional[str] = None,
                                        previous_risk_profile_expiry_date: Optional[str] = None) -> Dict[str, Any]:
            """
            Validates whether the client has an existing, unexpired risk profile assessment on file.
            Determines if a new risk assessment questionnaire administration is required.
            """

        def AdministerRiskQuestionnaire(client_id: str, client_age: int, client_annual_income: float,
                                        client_investment_experience_years: int, client_liquidity_needs: str,
                                        risk_questionnaire_responses: Dict[str, Any]) -> Dict[str, Any]:
            """
            Administers the regulatory-mandated Risk Tolerance Questionnaire to the client.
            Applies weighted scoring algorithm based on questionnaire responses.
            """

        def CalculateRiskProfile(risk_questionnaire_total_score: int) -> Dict[str, Any]:
            """
            Applies deterministic mapping algorithm to convert numerical risk_questionnaire_total_score 
            into categorical risk profile classification.
            """

        def AuthorizeAndActivateRiskProfile(client_id: str, calculated_risk_profile: str,
                                        client_authorization_method: str, transaction_timestamp: str) -> Dict[str, Any]:
            """
            Presents the calculated risk profile to the client for review and explicit consent.
            Computes risk profile expiry date by adding 730 days to current timestamp.
            Updates client master record to set investable_status to 'Active'.
            """

        def RecommendSuitableProducts(client_id: str, calculated_risk_profile: str, 
                                    investable_status: str) -> Dict[str, Any]:
            """
            Queries product master database applying hierarchical suitability filters based on 
            client calculated_risk_profile to return only suitable investment products.
            """

        def CreateInvestmentOrder(client_id: str, relationship_manager_id: str, product_code: str,
                                subscription_amount: float, settlement_account_number: str,
                                transaction_timestamp: str, branch_code: str, channel: str,
                                region: str) -> Dict[str, Any]:
            """
            Creates comprehensive new transaction record with system-generated transaction_id.
            Captures all order parameters and retrieves product master data.
            """

        def ValidateProductSuitability(transaction_id: str, calculated_risk_profile: str,
                                    product_risk_rating: str) -> Dict[str, Any]:
            """
            Executes automated real-time suitability check by comparing product_risk_rating 
            against client calculated_risk_profile using hierarchical matching matrix.
            """

        def ExecuteAMLSanctionsScreening(client_id: str, client_name: str, 
                                        transaction_id: str) -> Dict[str, Any]:
            """
            Interfaces with enterprise AML monitoring platform to perform real-time sanctions 
            list screening of client_id and client_name against all relevant sanctions databases.
            """

        def ValidateThresholdAndSOW(subscription_amount: float, monitoring_threshold_amount: float,
                                    source_of_wealth_documented: bool) -> Dict[str, Any]:
            """
            Compares subscription_amount against configured monitoring_threshold_amount to determine 
            if transaction exceeds regulatory large-transaction threshold requiring enhanced due diligence.
            """

        def VerifyFundsAvailability(settlement_account_number: str, 
                                subscription_amount: float) -> Dict[str, Any]:
            """
            Queries Core Banking System to retrieve real-time available_balance for specified 
            settlement_account_number and compares against subscription_amount.
            """

        def AggregateComplianceChecks(suitability_check_passed: bool, aml_sanctions_screening_status: str,
                                    transaction_threshold_exceeded: bool, source_of_wealth_documented: bool,
                                    funds_availability_verified: bool) -> Dict[str, Any]:
            """
            Aggregates validation results from all parallel pre-trade compliance checks.
            Applies boolean AND logic across all validation outcomes.
            """

        def CalculateTransactionFinancials(subscription_amount: float, front_end_load_percentage: float,
                                        annual_management_fee_percentage: float, current_nav: float,
                                        product_category: str, pre_trade_compliance_passed: bool) -> Dict[str, Any]:
            """
            Computes comprehensive transaction financial metrics with regulatory precision.
            Only executes when pre_trade_compliance_passed = TRUE.
            """

        def ManageCoolingOffPeriod(transaction_id: str, product_category: str, 
                                cooling_off_period_required: bool, cooling_off_period_hours: int,
                                client_authorization_timestamp: str) -> Dict[str, Any]:
            """
            Evaluates product_category against cooling-off period requirements to determine applicability.
            When applicable, calculates cooling-off expiry timestamp.
            """

        def ExecuteInvestmentTransaction(transaction_id: str, product_category: str, product_code: str,
                                        net_investment_amount: float, settlement_account_number: str,
                                        client_id: str, transaction_status: str, 
                                        subscription_amount: float) -> Dict[str, Any]:
            """
            Final execution step consummating investment transaction through appropriate channel routing.
            Only executes when transaction_status = 'Ready_for_Execution'.
            """

        def CalculateSTPFlag(transaction_id: str, suitability_mismatch_flag: bool,
                            supervisor_approval_required: bool, aml_sanctions_screening_status: str) -> Dict[str, Any]:
            """
            Evaluates entire transaction lifecycle to determine if straight-through processing was achieved.
            Sets straight_through_processing_flag to TRUE only for zero-touch transactions.
            """

        def ExecutePostTradeComplianceReview(transaction_id: str, audio_video_recording_id: Optional[str],
                                            suitability_mismatch_flag: bool, transaction_threshold_exceeded: bool,
                                            disclosure_documents_presented: List[str]) -> Dict[str, Any]:
            """
            Performs post-execution quality assurance and regulatory compliance verification.
            Only executes after transaction_status = 'Executed'.
            """

        def CalculateProcessingMetrics(transaction_timestamp: str, execution_timestamp: str) -> Dict[str, Any]:
            """
            Computes end-to-end transaction processing efficiency metrics by calculating 
            total_processing_time_minutes as elapsed time between transaction initiation and completion.
            """

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

    yaml_file = "/Invest/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Bench_API_Graph_Generation.yaml"
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
