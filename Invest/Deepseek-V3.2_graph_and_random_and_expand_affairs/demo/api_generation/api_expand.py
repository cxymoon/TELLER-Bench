import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():
    # === 输入参数 ===
    system_prompt = "你是一个特定领域SOP相关API扩展生成者，你要根据<api_samples></api_samples>中间提供给你的API样例合成20个函数名和参数与他们类似的冗余API（内部可以仅使用很简单的逻辑），并完成langchain格式封装。"

    client = OpenAI(
        api_key="",  # 替换为你的API Key
        base_url="https://"  # API中转地址
    )
    # === 读取YAML并填充 ===
    user_prompt = '''
        你是一个冗余API扩展合成专家。  
        下面将提供给你当前的API样例：  
        <api_samples>
            def MaterialImageQualityInspection(application_id: str, submitted_documents: List[str]) -> Dict[str, Any]:
                """材料影像质量检测API"""

            def MaterialCompletenessScoring(application_id: str, submitted_documents: List[str], 
                                                image_quality_scores: Dict[str, float], 
                                                ocr_recognition_accuracy: Dict[str, float],
                                                loan_purpose: str, customer_employment_type: str) -> Dict[str, Any]:
                """材料完整性评分API"""

            def IdentityVerificationAndLivenessDetection(application_id: str, customer_id: str,
                                                                identity_card_number: str, identity_card_name: str,
                                                                identity_card_images: Dict[str, str], 
                                                                face_image: str) -> Dict[str, Any]:
                """身份验证与活体检测API"""

            def AntiFraudCheck(application_id: str, customer_id: str, application_channel: str,
                                    identity_verification_result: bool, liveness_detection_passed: bool,
                                    device_fingerprint: Optional[str] = None, ip_address: Optional[str] = None,
                                    geolocation: Optional[str] = None) -> Dict[str, Any]:
                """反欺诈检查API"""


            def ApplicationLogicValidation(application_id: str, material_completeness_score: float,
                                                requested_loan_amount: float, requested_loan_term: int,
                                                loan_purpose: str, customer_age: int, 
                                                customer_employment_type: str, customer_monthly_income: float,
                                                purpose_invoice_amount: Optional[float] = None) -> Dict[str, Any]:
                """申请信息逻辑一致性校验API"""

            def CreditReportQueryAndParsing(application_id: str, customer_id: str,
                                                identity_verification_result: bool, anti_fraud_check_passed: bool,
                                                credit_report_authorization_confirmed: bool,
                                                customer_identity_number: str, customer_name: str) -> Dict[str, Any]:
                """征信报告查询与解析API"""

            def InternalExternalDataCrossVerification(application_id: str, customer_id: str,
                                                            customer_employment_type: str, customer_monthly_income: float,
                                                            customer_existing_debt: float, internal_deposit_balance: float,
                                                            internal_transaction_history_months: int,
                                                            total_existing_loan_balance: float,
                                                            total_credit_card_used_limit: float) -> Dict[str, Any]:
                """内外部数据交叉验证API"""

            def DebtToIncomeRatioCalculation(application_id: str, verified_monthly_income: float,
                                                    verified_existing_debt: float, total_existing_loan_balance: float,
                                                    total_credit_card_used_limit: float, requested_loan_amount: float,
                                                    requested_loan_term: int, 
                                                    estimated_interest_rate: Optional[float] = None) -> Dict[str, Any]:
                """债务收入比精细化计算API"""

            def CreditScoringAndRiskLevelAssessment(application_id: str, customer_age: int,
                                                            customer_employment_type: str, credit_history_months: int,
                                                            historical_overdue_count: int, current_credit_accounts: int,
                                                            credit_inquiry_count_6months: int, debt_to_income_ratio: float,
                                                            internal_deposit_balance: float, 
                                                            internal_transaction_history_months: int,
                                                            has_public_negative_records: bool, current_overdue_status: str,
                                                            risk_policy_version: str) -> Dict[str, Any]:
                """信用评分与风险等级评定API"""

            def PhoneVerificationRecording(application_id: str, assigned_reviewer_id: str, customer_id: str,
                                                phone_verification_timestamp: str, phone_verification_result: bool,
                                                customer_phone_number: Optional[str] = None,
                                                verification_questions_asked: Optional[List[str]] = None,
                                                customer_responses_consistent: Optional[bool] = None,
                                                employer_verification_conducted: Optional[bool] = None,
                                                employer_phone_number: Optional[str] = None,
                                                employer_confirmation_result: Optional[bool] = None,
                                                verification_notes: Optional[str] = None,
                                                recording_file_id: Optional[str] = None) -> Dict[str, Any]:
                """电话核实记录API"""

            def ReviewReportGeneration(application_id: str, assigned_reviewer_id: str,
                                            material_completeness_score: float, identity_verification_result: bool,
                                            anti_fraud_check_passed: bool, credit_score: int, risk_level: str,
                                            debt_to_income_ratio: float, income_verification_result: bool,
                                            phone_verification_result: bool, reviewer_recommendation: str,
                                            recommended_loan_amount: Optional[float] = None,
                                            recommendation_rationale: Optional[str] = None) -> Dict[str, Any]:
                """审查报告生成API"""

            def ApprovalQueueRouting(application_id: str, requested_loan_amount: float, credit_score: int,
                                            risk_level: str, debt_to_income_ratio: float, historical_overdue_count: int,
                                            anti_fraud_check_passed: bool, reviewer_recommendation: str) -> Dict[str, Any]:
                """审批队列路由API"""

            def RiskBasedPricingCalculation(application_id: str, credit_score: int, risk_level: str,
                                                internal_deposit_balance: float, internal_transaction_history_months: int,
                                                current_lpr_rate: float, requested_loan_term: int) -> Dict[str, Any]:
                """风险定价计算API"""

            def ApprovalDecisionExecution(application_id: str, assigned_approver_id: str, auto_approval_flag: bool,
                                                suggested_annual_interest_rate: float, requested_loan_amount: float,
                                                requested_loan_term: int, approval_decision: str,
                                                approved_loan_amount: Optional[float] = None,
                                                approved_annual_interest_rate: Optional[float] = None,
                                                approved_loan_term: Optional[int] = None,
                                                repayment_method: Optional[str] = None,
                                                approval_conditions: Optional[List[str]] = None,
                                                approval_reason: str = "") -> Dict[str, Any]:
                """审批决策执行API"""

            def MonthlyRepaymentCalculation(application_id: str, approved_loan_amount: float,
                                                approved_annual_interest_rate: float, approved_loan_term: int,
                                                repayment_method: str) -> Dict[str, Any]:
                """月还款额计算API"""

            def ElectronicContractGeneration(application_id: str, customer_id: str, approved_loan_amount: float,
                                                    approved_annual_interest_rate: float, approved_loan_term: int,
                                                    repayment_method: str, monthly_repayment_amount: float,
                                                    repayment_schedule_table: List[Dict], loan_purpose: str,
                                                    customer_name: str, customer_identity_number: str,
                                                    customer_address: str, customer_phone: str) -> Dict[str, Any]:
                """电子合同生成API"""

            def ContractSigningVerification(application_id: str, contract_id: str,
                                                contract_signing_timestamp: str, contract_signing_method: str,
                                                contract_hash_value: str, digital_signature_certificate: Optional[str] = None,
                                                customer_signature_image: Optional[str] = None,
                                                video_recording_file_id: Optional[str] = None) -> Dict[str, Any]:
                """合同签署验证API"""

            def PreDisbursementFinalAudit(application_id: str, customer_id: str, contract_id: str,
                                                signature_validity_verified: bool, approved_loan_amount: float,
                                                disbursement_account: str, disbursement_method: Optional[str] = None,
                                                payee_account_info: Optional[Dict] = None) -> Dict[str, Any]:
                """放款前终审校验API"""

            def FundDisbursementExecution(application_id: str, contract_id: str, compliance_check_passed: bool,
                                                approved_loan_amount: float, disbursement_account: str,
                                                disbursement_instruction_timestamp: str, disbursement_method: str,
                                                operator_id: str) -> Dict[str, Any]:
                """资金划拨执行API"""

            def BusinessStatusUpdateAndNotification(application_id: str, application_timestamp: str,
                                                            approval_decision: str, approval_timestamp: str,
                                                            review_start_timestamp: str,
                                                            disbursement_timestamp: Optional[str] = None,
                                                            disbursement_status: Optional[str] = None) -> Dict[str, Any]:
                """业务状态更新与通知API"""

        </api_samples>
        请你模仿这些API的函数名和参数，合成20个简单的类似冗余API，内部只需要1、2行简单的逻辑实现。但函数名要和样例尽可能相似，每个函数的参数名和数量也尽量相似
        另外，你要在和合成完API后输出类封装和tools列表封装，例如下：
            def generate_mock_cc_transaction(customer_id: str, amount: float, merchant_category: str) -> dict:
                # 伪逻辑：生成交易ID “MOCKTX…”，类别 merchant_category
                return {"transaction_id": f"MOCKTX{customer_id}{int(amount)}", "customer_id": customer_id, "amount": amount, "merchant_category": merchant_category}

            def compute_reward_points(customer_id: str, transactions: list[dict]) -> int:
                # 伪逻辑：积分 = 交易笔数 * 10
                return len(transactions) * 10        
        对以上两个函数的类封装：
            class Tool_generate_mock_cc_transaction(BaseModel):
                customer_id: str = Field(..., description="")
                amount: float = Field(..., description="")
                merchant_category: str = Field(..., description="")

            class Tool_compute_reward_points(BaseModel):
                customer_id: str = Field(..., description="")
                transactions: List[Dict[str, Any]] = Field(..., description="")
        对其列表封装：
            StructuredTool.from_function(
                func=generate_mock_cc_transaction,
                name="generate_mock_cc_transaction",
                description="",
                args_schema=Tool_generate_mock_cc_transaction
            ),
            StructuredTool.from_function(
                func=compute_reward_points,
                name="compute_reward_points",
                description="",
                args_schema=Tool_compute_reward_points
            )        
        你要输出完整的冗余API代码，及对应的类封装和列表封装，不要输出其他内容。类封装和列表封装中的description都留空。

        '''

    # 创建对话
    chat_completion = client.chat.completions.create(
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
        max_tokens=32768
    )

    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
