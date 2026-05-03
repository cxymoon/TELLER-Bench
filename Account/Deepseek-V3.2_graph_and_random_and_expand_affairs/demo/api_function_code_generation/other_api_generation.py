
def API_021_check_account_eligibility(account_id: str, account_age_days: int) -> bool:
    # 伪逻辑：若账户年龄超过365天且状态“active”，则合规，否则不合规
    if account_age_days >= 365:
        return True
    return False


def API_022_list_inactive_accounts(days_inactive: int) -> list[str]:
    # 伪逻辑：返回假定账户ID列表 “acct1001”“acct1002”…共 N/10 个
    return [f"acct{1000 + i}" for i in range(days_inactive // 10)]


def API_023_flag_suspicious_transfer(from_account: str, to_account: str, amount: float, recipient_risk_score: float) -> dict:
    # 伪逻辑：若金额 >10000 或者收款人风险评分 >0.8，则标记为可疑
    flagged = amount > 10000 or recipient_risk_score > 0.8
    return {"from": from_account, "to": to_account, "amount": amount, "flagged": flagged}


def API_024_get_credit_score(customer_id: str) -> int:
    # 伪逻辑：根据 customer_id 的最后一位数字生成评分 300‐850
    last_digit = int(customer_id[-1]) if customer_id[-1].isdigit() else 0
    return 300 + last_digit * 50


def API_025_issue_temporary_overdraft(account_id: str, threshold: float) -> dict:
    # 伪逻辑：若账户余额低于阈值，则批准额度=threshold*0.5，否则额度=0
    # 这里只模拟“低于”情况
    return {"account_id": account_id, "approved_overdraft": threshold * 0.5}


def API_026_close_dormant_account(account_id: str) -> str:
    # 伪逻辑：返回 “closed” 状态
    return f"Account {account_id} closed due to dormancy"


def API_027_compute_loan_risk(applicant_id: str, loan_amount: float) -> float:
    # 伪逻辑：评级 = loan_amount / 10000 + 随机基数 1.0
    return loan_amount / 10000 + 1.0


def API_028_report_failed_logins(branch_id: str, date: str) -> dict:
    # 伪逻辑：假设失败次数 = branch_id 长度 * 2
    failures = len(branch_id) * 2
    return {"branch_id": branch_id, "date": date, "failed_logins": failures}


def API_029_verify_beneficiary_details(beneficiary_account: str, bank_code: str) -> bool:
    # 伪逻辑：若 bank_code 以 “BK” 开头，视为合法
    return bank_code.startswith("BK")


def API_030_generate_transaction_id() -> str:
    # 伪逻辑：生成 “TX” + 时间戳字符串
    import time
    return f"TX{int(time.time() * 1000)}"


def API_031_get_fx_rate(base_currency: str, quote_currency: str) -> float:
    # 伪逻辑：如果 base==“USD”且 quote==“EUR”，汇率=0.9；否则 =1.0
    if base_currency == "USD" and quote_currency == "EUR":
        return 0.9
    return 1.0


def API_032_compute_savings_interest(account_id: str, balance: float, annual_rate: float) -> float:
    # 伪逻辑：月利息 = balance * (annual_rate/12)
    return balance * (annual_rate / 12)


def API_033_flag_deposit_limit(account_id: str, deposit_amount: float, daily_limit: float) -> bool:
    # 伪逻辑：若 deposit_amount > daily_limit，则返回 True
    return deposit_amount > daily_limit


def API_034_notify_term_deposit_maturity(customer_id: str, deposit_id: str, maturity_date: str) -> dict:
    # 伪逻辑：生成通知字典
    return {"customer_id": customer_id, "deposit_id": deposit_id, "maturity_date": maturity_date, "message": "Your term deposit is maturing soon"}


def API_035_archive_transactions(account_id: str, older_than_days: int) -> dict:
    # 伪逻辑：返回归档条数 = older_than_days * 10
    archived_count = older_than_days * 10
    return {"account_id": account_id, "archived_transactions": archived_count}


def API_036_calculate_loan_payoff(loan_id: str, extra_payments: list[float]) -> str:
    # 伪逻辑：预计还清日期 = “2025-12-31”
    return "2025-12-31"


def API_037_branch_cash_on_hand(branch_id: str, date: str) -> float:
    # 伪逻辑：现金在手 = branch_id 长度 * 1000
    return float(len(branch_id) * 1000)


def API_038_verify_atm_status(atm_id: str) -> dict:
    # 伪逻辑：返回状态 “OK” 且 time = 当前时间减去3600秒
    import time
    return {"atm_id": atm_id, "status": "OK", "last_maintenance": int(time.time()) - 3600}


def API_039_log_audit_event(event_type: str, details: dict) -> bool:
    # 伪逻辑：打印或记录 (这里只返回 True)
    print(f"Audit event logged: {event_type} with details {details}")
    return True


def API_040_generate_mock_cc_transaction(customer_id: str, amount: float, merchant_category: str) -> dict:
    # 伪逻辑：生成交易ID “MOCKTX…”，类别 merchant_category
    return {"transaction_id": f"MOCKTX{customer_id}{int(amount)}", "customer_id": customer_id, "amount": amount, "merchant_category": merchant_category}

def API_041_compute_reward_points(customer_id: str, transactions: list[dict]) -> int:
    # 伪逻辑：积分 = 交易笔数 * 10
    return len(transactions) * 10
