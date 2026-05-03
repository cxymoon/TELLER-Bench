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
    sop_title = "Banking Business Handling Process"
    api_code = '''

        # ==================== API FUNCTIONS ====================
        def get_application_data(application_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve application data by application_id"""

        def update_application_data(application_id: str, updates: Dict[str, Any]) -> bool:
            """Update application data with new values"""

        def convert_value(value: str, target_type: str):
            """Convert string value to appropriate type"""

        def ValidateApplicantIdentity(
            application_id: str,
            applicant_name: str,
            applicant_ssn: str,
            applicant_dob: str,
            applicant_address: str,
            co_applicant_name: Optional[str] = None,
            co_applicant_ssn: Optional[str] = None
        ) -> Dict[str, Any]:
            """Validates applicant identity and performs OFAC screening"""

        def ValidateEmploymentAndIncome(
            application_id: str,
            employment_status: str,
            gross_monthly_income: float,
            pay_stubs_uploaded: bool,
            w2_forms_uploaded: bool,
            employer_name: Optional[str] = None,
            employment_years: Optional[float] = None,
            additional_income_monthly: Optional[float] = None,
            tax_returns_uploaded: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Validates employment and income information"""

        def ValidateLiquidAssets(
            application_id: str,
            liquid_assets: float,
            bank_statements_uploaded: bool,
            down_payment_amount: float,
            requested_loan_amount: float,
            applicant_dob: str,
            retirement_accounts: Optional[float] = None
        ) -> Dict[str, Any]:
            """Validates liquid assets and calculates reserves"""

        def PullCreditReports(
            application_id: str,
            verified_applicant_name: str,
            verified_applicant_ssn: str,
            verified_applicant_dob: str,
            applicant_address: str,
            credit_consent_signed: bool,
            credit_consent_timestamp: str,
            co_applicant_name: Optional[str] = None,
            co_applicant_ssn: Optional[str] = None
        ) -> Dict[str, Any]:
            """Pulls tri-bureau credit reports"""

        def CalculateRepresentativeCreditScore(
            application_id: str,
            fico_score_equifax: Optional[int] = None,
            fico_score_experian: Optional[int] = None,
            fico_score_transunion: Optional[int] = None,
            co_applicant_fico_equifax: Optional[int] = None,
            co_applicant_fico_experian: Optional[int] = None,
            co_applicant_fico_transunion: Optional[int] = None
        ) -> Dict[str, Any]:
            """Calculates representative credit score using tri-merge methodology"""

        def ExecutePrequalificationAnalysis(
            application_id: str,
            representative_credit_score: int,
            preliminary_gross_monthly_income: float,
            total_monthly_debt_obligations: float,
            requested_loan_amount: float,
            loan_term_months: int,
            estimated_property_value: float,
            down_payment_amount: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str
        ) -> Dict[str, Any]:
            """Executes pre-qualification analysis"""

        def VerifyEmploymentAndIncome(
            application_id: str,
            prequalification_decision: str,
            employment_status: str,
            gross_monthly_income: float,
            pay_stubs_uploaded: bool,
            w2_forms_uploaded: bool,
            employer_name: Optional[str] = None,
            employment_years: Optional[float] = None,
            additional_income_monthly: Optional[float] = None,
            tax_returns_uploaded: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Verifies employment and income through VOE and document analysis"""

        def VerifyAssetsAndReserves(
            application_id: str,
            prequalification_decision: str,
            liquid_assets: float,
            bank_statements_uploaded: bool,
            down_payment_amount: float,
            requested_loan_amount: float,
            applicant_dob: str,
            verified_gross_monthly_income: float,
            property_type: str,
            property_occupancy: str,
            retirement_accounts: Optional[float] = None
        ) -> Dict[str, Any]:
            """Verifies assets and calculates reserve months"""

        def AnalyzeCreditHistory(
            application_id: str,
            credit_report_full_data: Dict[str, Any],
            derogatory_marks_count: int,
            bankruptcy_history: bool
        ) -> Dict[str, Any]:
            """Analyzes credit history in detail"""

        def CalculateDebtRatios(
            application_id: str,
            verified_gross_monthly_income: float,
            total_monthly_debt_obligations: float,
            requested_loan_amount: float,
            loan_term_months: int,
            estimated_interest_rate_prequalification: float,
            property_address: str,
            estimated_property_value: float,
            representative_credit_score: int,
            preliminary_ltv_ratio: float
        ) -> Dict[str, Any]:
            """Calculates precise debt-to-income and debt service coverage ratios"""

        def CalculateProbabilityOfDefault(
            application_id: str,
            representative_credit_score: int,
            debt_to_income_ratio: float,
            preliminary_ltv_ratio: float,
            reserve_months_calculated: float,
            employment_years: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str,
            detailed_derogatory_marks_count: int
        ) -> Dict[str, Any]:

        def DetermineFinalInterestRate(
            application_id: str,
            credit_score_grade: str,
            representative_credit_score: int,
            preliminary_ltv_ratio: float,
            property_type: str,
            property_occupancy: str,
            loan_purpose: str,
            debt_to_income_ratio: float,
            loan_term_months: int
        ) -> Dict[str, Any]:
            """Determines final risk-adjusted interest rate"""

        def EvaluateCompensatingFactors(
            application_id: str,
            debt_to_income_ratio: float,
            reserve_months_calculated: float,
            payment_history_quality: str,
            verified_gross_monthly_income: float,
            monthly_housing_payment_total: float,
            total_monthly_debt_obligations: float,
            preliminary_ltv_ratio: float,
            employment_years: float,
            credit_depth_rating: str,
            applicant_address: str
        ) -> Dict[str, Any]:
            """Evaluates compensating factors for marginal applications"""

        def OrderPropertyAppraisal(
            application_id: str,
            prequalification_decision: str,
            property_address: str,
            property_type: str,
            property_occupancy: str,
            estimated_property_value: float,
            requested_loan_amount: float,
            loan_purpose: str
        ) -> Dict[str, Any]:
            """Orders property appraisal"""

        def ConductPropertyValuation(
            application_id: str,
            appraisal_order_number: str,
            property_address: str,
            property_type: str,
            estimated_property_value: float
        ) -> Dict[str, Any]:
            """Conducts property valuation through appraiser site inspection"""

        def ValidateAppraisalWithAVM(
            application_id: str,
            property_address: str,
            final_appraised_value: float,
            property_type: str
        ) -> Dict[str, Any]:
            """Validates appraisal using AVM"""

        def CalculateLoanToValueRatio(
            application_id: str,
            final_appraised_value: float,
            requested_loan_amount: float,
            estimated_property_value: float,
            loan_purpose: str,
            representative_credit_score: int,
            property_type: str,
            property_occupancy: str
        ) -> Dict[str, Any]:
            """Calculates loan-to-value ratio"""

        def RenderFinalLoanDecision(
            application_id: str,
            debt_to_income_ratio: float,
            debt_service_coverage_ratio: float,
            loan_to_value_ratio: float,
            probability_of_default_score: float,
            credit_score_grade: str,
            final_risk_adjusted_interest_rate: float,
            compensating_factors_present: bool,
            property_condition: str,
            bankruptcy_seasoning_compliant: bool,
            foreclosure_history_compliant: bool,
            verified_gross_monthly_income: float,
            requested_loan_amount: float,
            loan_term_months: int,
            application_timestamp: str
        ) -> Dict[str, Any]:
            """Renders final loan decision"""

        def ExecuteFairLendingMonitoring(
            application_id: str,
            final_decision: str,
            representative_credit_score: int,
            debt_to_income_ratio: float,
            loan_to_value_ratio: float,
            property_address: str,
            approved_interest_rate: Optional[float] = None,
            conditions_count: Optional[int] = None
        ) -> Dict[str, Any]:
            """Executes fair lending compliance monitoring"""

        def TrackLoanPerformanceMetrics(
            application_id: str,
            application_timestamp: str,
            final_decision: str,
            final_decision_timestamp: str,
            conditions_count: int,
            approved_loan_amount: Optional[float] = None
        ) -> Dict[str, Any]:
            """Tracks loan performance metrics"""

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

    yaml_file = "/Loans/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Bench_API_Graph_Generation.yaml"
    system_prompt = "You are a professional AI system specializing in analyzing complex business processes and deducing API dependency relationships. Your task is to select all valid APIs from the provided implemented API interfaces (which contain many redundant APIs) based on the business context, and then generate a comprehensive directed graph of API dependencies. This graph must accurately reflect the data flows and prerequisite dependencies between the APIs."



    # === 读取YAML并填充 ===
    user_prompt = load_yaml_prompt(yaml_file, sop_title, api_code, additional_context)

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
        model="deepseek-v3.2",  # 指定模型
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
