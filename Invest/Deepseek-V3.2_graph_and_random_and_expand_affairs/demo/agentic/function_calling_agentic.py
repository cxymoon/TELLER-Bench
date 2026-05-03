from typing import Dict
from langchain.tools import StructuredTool  # 确保导入StructuredTool
# from langchain_core.tools.structured import StructuredTool

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import yaml
import pandas as pd
import json
import ast
from pydantic import BaseModel, Field
import hashlib
import random
from datetime import datetime, timedelta
import math

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import threading
import os
import sys
from io import StringIO
from decimal import Decimal
import re
from typing import Any, Dict, List, Optional, Union

# --- 载入 SOP-Bench 提示模板中的 text 字段 ---
PROMPT_YAML_PATH = "Invest/Deepseek-V3.2_graph_and_random_and_expand_affairs/prompts/Function_Calling_Agentic_Prompt.yaml"
sop_text = '''
        <SOP>

        1. Purpose

        The purpose of this Standard Operating Procedure is to establish a comprehensive, multi-layered framework for the execution of retail wealth management investment transactions within a banking and financial services environment, encompassing the full lifecycle from initial client risk profiling and investor suitability assessment through to the final execution and settlement of investment product subscriptions, including but not limited to open-ended mutual fund units and complex structured financial instruments such as Dual Currency Investments and Principal Protected Notes. This SOP ensures strict adherence to all applicable regulatory mandates governing investor protection, Know Your Customer (KYC) requirements, Anti-Money Laundering (AML) screening protocols, suitability and appropriateness assessments as prescribed under MiFID II equivalents and local banking supervisory frameworks, while simultaneously optimizing Assets Under Management (AUM) growth and fee-based revenue generation through systematic, auditable, and compliant sales processes. The procedure integrates automated risk-based decisioning engines, real-time compliance validation checkpoints, and multi-system orchestration across Wealth Management Systems (WMS), Customer Relationship Management (CRM) platforms, Sales Compliance Recording Systems (SCRS), and Risk Engines to achieve Straight-Through Processing (STP) rates exceeding industry benchmarks while maintaining zero-tolerance policies for miss-selling incidents and regulatory breaches.

        2. Scope

        This Standard Operating Procedure applies comprehensively to all Relationship Managers (RMs), Investment Product Specialists (IPSs), Compliance Officers, and Back Office Operations personnel who are directly or indirectly involved in the origination, processing, validation, execution, and post-trade settlement of investment product transactions across all channels including Branch, Phone Banking, Mobile Application, and Online Banking platforms within the organization's retail wealth management business unit. The scope explicitly encompasses three distinct but interconnected sub-business processes: (A) Risk Profiling and Investor Suitability Assessment, which establishes the foundational eligibility and investment parameters for each client; (B) Mutual Fund Subscription (Lump Sum), covering the direct purchase of open-ended mutual fund units through immediate debit from client settlement accounts; and (C) Structured Product Enrollment, addressing the sales, disclosure, and booking procedures for complex financial instruments requiring enhanced due diligence, mandatory audio-video recording, and regulatory cooling-off periods. The SOP governs all transactions originating from clients identified by unique client_id values within the system, processed by authorized relationship_manager_id personnel, and recorded with distinct transaction_id identifiers for complete audit trail purposes. Geographically, this procedure applies across all regions and branch_code locations where the organization maintains retail wealth management operations, with specific regional regulatory adaptations documented in supplementary annexes where local supervisory requirements impose additional or variant obligations beyond the baseline standards articulated herein.

        3. Definitions

        **Risk Tolerance Questionnaire** refers to the regulatory-mandated, multi-dimensional assessment instrument administered to each client to systematically capture quantitative and qualitative data pertaining to the client's age bracket, income stability characteristics, declared annual income (client_annual_income), total net worth (client_net_worth), investment experience measured in years (client_investment_experience_years), liquidity requirements categorized as Low, Medium, or High (client_liquidity_needs), investment time horizon preferences, psychological loss tolerance thresholds, and prior exposure to various asset classes, with all responses recorded in structured JSON format within the risk_questionnaire_responses field and subsequently processed through a proprietary weighted scoring algorithm to generate a risk_questionnaire_total_score ranging from zero to one hundred points.

        **Calculated Risk Profile** denotes the system-generated investor categorization output derived from the risk_questionnaire_total_score through algorithmic mapping logic whereby scores falling within the range of zero to thirty-five points inclusive are categorized as "Conservative," scores from thirty-six to sixty-five points inclusive are designated as "Balanced," and scores from sixty-six to one hundred points inclusive are classified as "Aggressive," with the resulting classification stored in the calculated_risk_profile field and serving as the primary reference parameter for all subsequent suitability matching operations and product recommendation constraints.

        **Product Risk Rating** signifies the standardized risk classification assigned to each investment product identified by product_code and product_name, expressed through a five-tier taxonomy comprising "Conservative," "Moderate," "Balanced," "Growth," and "Aggressive" designations, with the rating determined through quantitative analysis of historical volatility metrics, asset class composition, leverage factors, currency exposure, liquidity characteristics, and structural complexity, and permanently associated with each product record in the product master database to enable automated suitability comparison operations.

        **Suitability Check** represents the automated, real-time validation process executed by the Risk Engine middleware wherein the product_risk_rating of the selected investment instrument is algorithmically compared against the client's calculated_risk_profile using a hierarchical matching matrix such that a product rated "Conservative" is deemed suitable for all client risk profiles, "Moderate" products are suitable for "Balanced" and "Aggressive" profiles, "Balanced" products are suitable for "Balanced" and "Aggressive" profiles, "Growth" products are suitable only for "Aggressive" profiles, and "Aggressive" products are exclusively suitable for "Aggressive" client profiles, with the comparison outcome recorded as a boolean value in the suitability_check_passed field.

        **Suitability Mismatch** occurs when the hierarchical risk comparison logic determines that the product_risk_rating exceeds the permissible threshold relative to the client's calculated_risk_profile, triggering the suitability_mismatch_flag to assume a TRUE boolean state and initiating a mandatory exception workflow requiring the generation of a mismatch_waiver_document_id, explicit client acknowledgment of elevated risk exposure through documented signature, and supervisor_approval_required flag activation necessitating authorization from a supervisor_id personnel possessing appropriate delegation authority before transaction progression is permitted.

        **Net Asset Value (NAV)** constitutes the per-unit market valuation of a mutual fund investment product calculated by the fund administrator at the close of each trading day by dividing the total market value of all securities held within the fund portfolio, minus all liabilities and accrued expenses, by the total number of outstanding fund units, with the most recent NAV figure stored in the current_nav field and utilized as the basis for determining estimated_units to be allocated to the client through the mathematical operation of dividing the net_investment_amount by the current_nav with results typically expressed to two decimal places.

        **Front-End Load** refers to the sales charge or commission assessed at the point of initial investment purchase, expressed as a percentage of the gross subscription_amount in the front_end_load_percentage field, with the absolute dollar value of this charge computed as the product of subscription_amount multiplied by front_end_load_percentage divided by one hundred and recorded in the front_end_load_amount field, thereby reducing the actual capital deployed into the investment product which is represented by the net_investment_amount calculated as subscription_amount minus front_end_load_amount.

        **Annual Management Fee** represents the ongoing administrative and portfolio management charge levied by the fund house or investment manager, expressed as an annualized percentage rate in the annual_management_fee_percentage field, with the estimated first-year dollar impact calculated as net_investment_amount multiplied by annual_management_fee_percentage divided by one hundred and recorded in the annual_management_fee_amount field to provide clients with transparent disclosure of the recurring cost burden associated with the investment product over the holding period.

        **Straight-Through Processing (STP)** designates the operational paradigm wherein a transaction progresses from initial order entry by the Relationship Manager through all validation checkpoints, compliance screening layers, execution operations, and booking confirmations without requiring any manual intervention, data correction, exception handling, or workflow rerouting by Back Office Operations personnel, with achievement of this state indicated by the straight_through_processing_flag assuming a TRUE boolean value and serving as a key performance metric for operational efficiency measurement.

        **Cooling-Off Period** signifies a regulatory-mandated temporal interval during which a client who has authorized the purchase of a complex or high-risk structured product retains the unconditional right to cancel the transaction without incurring penalties, surrender charges, or adverse financial consequences, with the duration specified in hours within the cooling_off_period_hours field (typically twenty-four or forty-eight hours depending on product category and jurisdiction), and the transaction maintained in a "Pending_Cooling_Off" status within the transaction_status field until the designated period expires or the client exercises their cancellation right.

        **Audio-Video Recording (VR)** refers to the mandatory electronic capture, encryption, and archival of the complete sales interaction between the Relationship Manager and the client for transactions involving complex structured products where audio_video_recording_required assumes a TRUE boolean value, with each recording session assigned a unique audio_video_recording_id identifier and stored within the Sales Compliance Recording System (SCRS) for subsequent compliance review, quality assurance sampling, and regulatory audit purposes, ensuring complete documentation of risk disclosure script delivery and client verbal acknowledgment of understanding.

        **Trade Reference Number** constitutes the unique system-generated alphanumeric identifier assigned to each successfully executed investment transaction at the moment of order booking, stored in the trade_reference_number field, serving as the primary reference key for all downstream settlement operations, client confirmation communications, portfolio accounting entries, and regulatory transaction reporting obligations.

        **Contract Note Number** represents the specialized confirmation document identifier generated exclusively for structured product transactions processed through Treasury or Global Markets desks, stored in the contract_note_number field, containing comprehensive details including notional amount booked, strike price parameters, value date specifications, payoff scenario matrices, and counterparty information necessary for client record-keeping and regulatory compliance.

        4. Input

        The initiation of any investment transaction workflow governed by this Standard Operating Procedure requires the complete availability and validation of the following input data elements and contextual parameters drawn from multiple source systems and client interaction touchpoints. First, the transaction must be associated with a valid and active client_id representing a customer whose identity has been previously verified through government-issued identification documents and whose profile data is maintained within the Customer Relationship Management (CRM) system, including but not limited to client_name, client_age measured in integer years, client_annual_income declared in United States Dollar equivalents, client_net_worth representing total liquid and illiquid asset valuations, and client_investment_experience_years quantifying the duration of prior involvement with capital markets instruments. Second, the transaction must be originated and sponsored by an authorized relationship_manager_id whose relationship_manager_name, delegation authorities, and product certification credentials are recorded in the human resources and access control systems, with the initial transaction_timestamp capturing the precise date and time of workflow initiation in ISO 8601 format with timezone designation. Third, the client's settlement account infrastructure must be established with a valid settlement_account_number associated with the client_id and containing a verifiable available_balance in United States Dollar denomination sufficient to support the intended subscription_amount inclusive of all applicable fees and charges. Fourth, for transactions involving investment products, the system must have access to complete product master data including product_code, product_name, product_risk_rating, product_category classification (Mutual_Fund, Bond, Structured_Note, Dual_Currency_Investment, or Principal_Protected_Note), current_nav for unit-based instruments, front_end_load_percentage applicable to the specific client relationship tier, annual_management_fee_percentage as stipulated in the product prospectus, and disclosure_documents_presented list containing references to Fund_Fact_Sheet, Prospectus, Key Information Document (KID), and any supplementary risk disclosure materials. Fifth, the client's current compliance and eligibility status must be retrievable including the previous_risk_profile classification if any prior assessment exists, the previous_risk_profile_expiry_date indicating when the existing assessment loses validity, the aml_sanctions_screening_status reflecting the outcome of most recent sanctions list screening operations, the source_of_wealth_documented boolean flag indicating whether SOW documentation is on file, and the monitoring_threshold_amount parameter defining the trigger point for enhanced AML scrutiny which is standardly configured at one hundred thousand United States Dollars but may be adjusted based on client segment or regional regulatory requirements. Sixth, for structured products specifically, the input data must include the audio_video_recording_required boolean flag determined by product classification rules, the cooling_off_period_required flag indicating regulatory mandate applicability, and the cooling_off_period_hours duration specification. Seventh, the transaction must be tagged with operational metadata including branch_code identifying the originating business location, channel designation (Branch, Phone_Banking, Mobile_App, or Online_Banking), region code for geographic reporting purposes, and assignment of a compliance_officer_id responsible for oversight and exception review. Eighth, system access to market_data_source platforms such as Bloomberg or Reuters must be confirmed operational to enable real-time NAV verification, pricing validation, and market condition assessment. Finally, the client_authorization_method infrastructure must be configured and functional whether utilizing One-Time Password (OTP) delivery systems, biometric signature capture devices, physical signature verification protocols, or digital signature certificate frameworks to enable legally binding transaction consent capture with associated client_authorization_timestamp recording.

        5. Main Procedure

        5.1 Client Risk Profile Assessment and Establishment Phase

        5.1.1 Upon initiation of an investment transaction request or commencement of a new client investment relationship, the Relationship Manager designated by relationship_manager_id must first verify that the client identified by client_id possesses a current and valid risk profile by interrogating the CRM system to retrieve the previous_risk_profile classification and associated previous_risk_profile_expiry_date timestamp, comparing the expiry date against the current transaction_timestamp to determine temporal validity whereby if the previous_risk_profile_expiry_date is null indicating no prior assessment exists or if the previous_risk_profile_expiry_date precedes the current transaction_timestamp indicating expiry of the existing assessment, then the RM must initiate a complete Risk Tolerance Questionnaire administration workflow, whereas if a valid and unexpired previous_risk_profile exists then the RM may proceed directly to product selection and suitability validation stages while documenting the continued reliance on the existing risk categorization in the transaction audit trail.

        5.1.2 For scenarios requiring new or refreshed risk assessment, the Relationship Manager must administer the standardized Risk Tolerance Questionnaire instrument through either digital tablet-based presentation in branch environments or secure web-form delivery for remote channel transactions, ensuring the client personally completes all mandatory questions without RM influence or suggestion that might compromise assessment objectivity, with the questionnaire systematically collecting structured responses covering age bracket classification derived from client_age such that clients aged eighteen to thirty years receive zero baseline points, thirty-one to forty years receive five points, forty-one to fifty years receive ten points, fifty-one to sixty years receive fifteen points, and sixty-one years or greater receive twenty points reflecting increasing conservative orientation with age; income stability evaluation based on client_annual_income and employment sector whereby salaried employees in stable industries receive fifteen points, self-employed or commission-based income sources receive ten points, and retired individuals with fixed income streams receive twenty points; investment time horizon assessment whereby horizons less than one year receive zero points, one to three years receive ten points, three to five years receive fifteen points, five to ten years receive twenty points, and greater than ten years receive twenty-five points reflecting capacity to weather market volatility; loss tolerance psychological evaluation presented through scenario-based questions whereby willingness to accept potential loss of zero to five percent over one year receives five points, five to fifteen percent tolerance receives fifteen points, fifteen to thirty percent tolerance receives twenty-five points, and acceptance of greater than thirty percent loss potential receives thirty points; and prior investment experience scoring wherein client_investment_experience_years of zero years receives zero points, one to three years receives five points, four to seven years receives ten points, eight to fifteen years receives fifteen points, and greater than fifteen years receives twenty points, with all individual question responses recorded in JSON format within the risk_questionnaire_responses field maintaining complete granular detail for audit and quality review purposes.

        5.1.3 Upon completion of questionnaire data capture, the Risk Engine component of the Wealth Management System must aggregate all individual question scores to compute the risk_questionnaire_total_score as an integer summation ranging from zero to one hundred points, subsequently applying deterministic mapping logic to derive the calculated_risk_profile classification such that total scores in the range zero through thirty-five points inclusive result in "Conservative" classification reflecting low risk tolerance, preference for capital preservation, and suitability primarily for fixed income and money market instruments; scores in the range thirty-six through sixty-five points inclusive result in "Balanced" classification indicating moderate risk tolerance, acceptance of measured volatility, and suitability for diversified portfolios combining equity and fixed income exposures; and scores in the range sixty-six through one hundred points inclusive result in "Aggressive" classification signifying high risk tolerance, focus on capital appreciation over income generation, and suitability for concentrated equity positions, alternative investments, and complex structured products, with the calculated_risk_profile value recorded in the designated field and simultaneously triggering system updates to the client's master profile record in the CRM database.

        5.1.4 The system must automatically present the calculated_risk_profile classification to the client through the RM's interface along with a comprehensive narrative description explaining the characteristics, typical investment strategies, and product categories associated with the determined profile, requiring explicit client acknowledgment through one of the configured client_authorization_method protocols such that if the client agrees with the calculated classification they must provide affirmative consent via OTP validation, biometric signature capture, physical signature on printed assessment summary, or digital signature application with the client_authorization_timestamp precisely recorded, whereas if the client disputes the calculated classification expressing belief that their risk tolerance is either higher or lower than the system determination, the RM must initiate a documented re-evaluation workflow involving completion of a supplementary questionnaire with additional scenario-based questions, consultation with an Investment Product Specialist referenced by IPS identifier for expert second opinion, and mandatory submission of written justification narrative explaining the basis for override request which is subject to compliance_officer_id review and approval before an adjusted calculated_risk_profile may be recorded in the system with all override documentation forming part of the permanent client file.

        5.1.5 Following successful client acceptance and authorization of the calculated_risk_profile, the system must automatically compute the risk_profile_expiry_date by adding a duration of exactly seven hundred thirty days (equivalent to two calendar years) to the current transaction_timestamp to establish the validity period for the assessment, update the client's master record in the CRM with the new calculated_risk_profile value replacing any previous_risk_profile designation, set the risk_profile_expiry_date in the client profile, and most critically modify the client's investable_status field to assume the value "Active" thereby removing any systemic blocks that would prevent investment transaction processing and enabling the client to proceed with product purchases within their suitability parameters, while also triggering automated calendar reminders to the assigned relationship_manager_id at intervals of sixty days prior to expiry, thirty days prior to expiry, and at expiry date to ensure proactive risk profile renewal before eligibility lapses and transactions become blocked.

        5.2 Investment Product Selection and Suitability Validation Phase

        5.2.1 With a valid and active calculated_risk_profile established and the investable_status confirmed as "Active," the Relationship Manager must proceed to product recommendation and selection activities by first retrieving from the product master database all investment instruments whose product_risk_rating falls within the permissible range aligned with the client's calculated_risk_profile utilizing the hierarchical suitability matrix logic whereby clients with "Conservative" calculated profiles may only be presented products rated "Conservative" or "Moderate," clients with "Balanced" calculated profiles may be presented products rated "Conservative," "Moderate," or "Balanced," and clients with "Aggressive" calculated profiles may be presented products across all rating categories including "Conservative," "Moderate," "Balanced," "Growth," and "Aggressive," with the RM particularly focusing recommendation efforts on products generating optimal fee-based revenue through front_end_load_percentage and annual_management_fee_percentage structures while remaining within client suitability constraints and appropriateness for the client's stated investment objectives, time horizon, and current portfolio composition.

        5.2.2 Upon identifying a specific target product characterized by product_code, product_name, product_category, and product_risk_rating, the Relationship Manager must conduct a comprehensive product presentation session with the client encompassing mandatory disclosure of all material information contained within the regulatory-required disclosure_documents_presented list, specifically including the Fund Fact Sheet containing one-page summary of investment objective, asset allocation strategy, historical performance metrics for one-year, three-year, five-year, and since-inception periods with explicit disclaimer that past performance does not guarantee future results, the complete Prospectus or offering document detailing investment policies, restrictions, risk factors, management team biographies, and financial statements, and the Key Information Document (KID) formatted in standardized layout presenting objectives, risk and reward profile using the Synthetic Risk and Reward Indicator (SRRI) scale, cost disclosures showing reduction in yield due to charges, and performance scenarios under stressed, unfavorable, moderate, and favorable market conditions, with the RM required to explicitly draw client attention to the product_risk_rating classification, the front_end_load_percentage expressed both as percentage rate and absolute dollar amount calculated as subscription_amount multiplied by front_end_load_percentage divided by one hundred, the annual_management_fee_percentage similarly expressed in both percentage and estimated annual dollar impact, and specific sections of the risk disclosure content addressing market risk, credit risk for fixed income products, currency risk for internationally-invested funds, liquidity risk, and concentration risk where applicable.

        5.2.3 For transactions involving structured products where product_category is designated as "Structured_Note," "Dual_Currency_Investment," or "Principal_Protected_Note," additional procedural requirements beyond standard mutual fund disclosure become mandatory, specifically requiring the Relationship Manager to conduct a detailed payoff scenario analysis presentation utilizing market data sourced from the designated market_data_source platform such as Bloomberg or Reuters to demonstrate to the client through numerical examples and graphical illustrations how the structured product's return profile varies under different market conditions, for instance explaining that a Dual Currency Investment may deliver a five percent coupon if the underlying exchange rate remains above a specified strike level at maturity but could result in principal repayment in a depreciated foreign currency if the strike is breached thereby crystallizing foreign exchange loss, and for any structured product transaction where audio_video_recording_required assumes TRUE boolean value indicating regulatory mandate for sales process documentation, the RM must activate the Sales Compliance Recording System (SCRS) to capture complete audio and video recording of the presentation and client interaction, with the system generating a unique audio_video_recording_id identifier that becomes permanently associated with the transaction_id and stored in encrypted format on secure archival storage infrastructure for minimum retention periods typically extending seven years to satisfy regulatory audit and examination requirements.

        5.2.4 Following completion of all mandatory disclosure presentations and client review of documentation, the Relationship Manager must proceed to formal order entry by accessing the Wealth Management System's transaction module and creating a new transaction record populated with transaction_id generated through system-assigned sequential alphanumeric format (e.g., INV_YYYYMMDDNNNNN), capturing the current transaction_timestamp with precision to the second in UTC format, associating the client_id and relationship_manager_id with the transaction, specifying the transaction_type as "Risk_Profiling" if this is solely a profile establishment transaction, "Mutual_Fund_Subscription" for open-ended fund purchases, or "Structured_Product_Enrollment" for complex instrument bookings, recording the selected product_code and confirming automatic population of related product_name, product_risk_rating, product_category, current_nav for NAV-based funds, front_end_load_percentage, and annual_management_fee_percentage from the product master database, capturing the client's desired subscription_amount in United States Dollar value, identifying the designated settlement_account_number from which funds will be debited, and tagging the transaction with branch_code, channel, region, and assigned compliance_officer_id based on transaction value thresholds and risk characteristics of the product.

        5.2.5 At the moment of order entry completion but prior to final client authorization and execution, the Wealth Management System must invoke its integrated Risk Engine middleware to perform automated suitability validation by executing a comparative evaluation between the product_risk_rating of the selected investment instrument and the client's established calculated_risk_profile using the hierarchical matching algorithm such that the suitability_check_passed field assumes TRUE boolean value if the product rating is equal to or lower than the client profile on the Conservative-Moderate-Balanced-Growth-Aggressive hierarchy (specifically: Conservative products pass for all profiles; Moderate products pass for Balanced and Aggressive profiles; Balanced products pass for Balanced and Aggressive profiles; Growth products pass only for Aggressive profiles; Aggressive products pass only for Aggressive profiles), whereas if the product_risk_rating is hierarchically superior to the client's calculated_risk_profile indicating a suitability violation, the suitability_check_passed field assumes FALSE boolean value and the suitability_mismatch_flag is set to TRUE triggering immediate workflow diversion to an exception handling process that displays a hard-block error message to the Relationship Manager preventing order progression unless the RM initiates a formal Risk Mismatch Waiver procedure requiring generation of a mismatch_waiver_document_id identifier, electronic or physical presentation of a standardized waiver form to the client explicitly stating in clear and unambiguous language that the recommended product exceeds the client's assessed risk tolerance and may result in investment losses greater than the client indicated willingness to accept during risk profiling, capture of client signature acknowledging and accepting the elevated risk exposure and confirming understanding that they are proceeding against the bank's standard suitability recommendation, and critically the activation of supervisor_approval_required flag to TRUE boolean state necessitating routing of the transaction to a supervisory personnel identified by supervisor_id possessing appropriate delegation authority who must review the complete risk assessment documentation, product disclosure records, waiver acknowledgment, and RM justification narrative before either approving the transaction which permits continuation to execution or rejecting the transaction which terminates the workflow and requires the RM to recommend an alternative suitable product to the client.

        5.3 Pre-Trade Compliance and Financial Validation Phase

        5.3.1 For all transactions regardless of product type where the suitability_check_passed field confirms TRUE boolean value either through direct matching or through approved supervisor override following mismatch waiver procedures, the system must proceed to execute comprehensive pre-trade compliance screening operations beginning with Anti-Money Laundering (AML) sanctions list verification by interfacing with the bank's enterprise AML monitoring platform to submit the client_id, client_name, and transaction details for real-time screening against all relevant sanctions lists including OFAC (Office of Foreign Assets Control) Specially Designated Nationals (SDN) list, EU consolidated sanctions list, UN Security Council sanctions designations, and national sanctions registries applicable to the bank's operating jurisdictions, with the screening outcome recorded in the aml_sanctions_screening_status field assuming values of "Clear" if no matches are detected indicating permission to proceed, "Flagged" if potential name matches or watchlist hits are identified requiring immediate escalation to the compliance_officer_id for manual review and disposition decision before transaction can advance, or maintaining "Pending" status during the interval while screening systems process the query with typical resolution timeframes under five seconds for unambiguous clear results.

        5.3.2 Concurrent with AML sanctions screening, the system must evaluate whether the transaction subscription_amount exceeds the configured monitoring_threshold_amount parameter which is standardly established at one hundred thousand United States Dollars (100000.00) for retail wealth management transactions, setting the transaction_threshold_exceeded field to TRUE boolean value when subscription_amount is strictly greater than monitoring_threshold_amount, and when this threshold breach condition is detected, the system must automatically verify that the source_of_wealth_documented field associated with the client_id assumes TRUE boolean value indicating that current and valid Source of Wealth (SOW) documentation such as employment contracts, business ownership records, inheritance documentation, real estate sale proceeds verification, or investment portfolio liquidation statements have been collected and verified by the compliance team and are stored in the document management system, such that if source_of_wealth_documented is FALSE when transaction_threshold_exceeded is TRUE, the system must generate a hard-block preventing transaction execution and display to the Relationship Manager explicit instruction that SOW documentation must be obtained, submitted for compliance review, and approved before the transaction can proceed with the workflow entering suspended state pending documentation completion.

        5.3.3 Following successful completion of AML screening yielding "Clear" status and confirmation of SOW documentation adequacy for threshold-exceeding transactions, the system must proceed to funds availability verification by executing a real-time balance inquiry against the Core Banking System using the settlement_account_number to retrieve the current available_balance in United States Dollar equivalents, comparing this available_balance against the required subscription_amount to determine sufficiency such that the funds_availability_verified field assumes TRUE boolean value only when available_balance is greater than or equal to subscription_amount thereby confirming that adequate liquid funds exist to settle the transaction without creating overdraft conditions or requiring supplementary funding arrangements, whereas if available_balance is less than subscription_amount indicating insufficient funds, the funds_availability_verified field assumes FALSE boolean value triggering generation of an insufficient funds notification to the Relationship Manager who must either arrange for the client to transfer additional funds into the settlement_account_number before proceeding or reduce the subscription_amount to fit within the available_balance constraint with the revised amount requiring fresh suitability validation if the reduction is substantial enough to change fee structures or minimum investment thresholds.

        5.3.4 Aggregating all component validation outcomes, the system must compute an overall pre_trade_compliance_passed field assuming TRUE boolean value only when all of the following conditions are simultaneously satisfied: suitability_check_passed equals TRUE (either directly or through approved override), aml_sanctions_screening_status equals "Clear," if transaction_threshold_exceeded equals TRUE then source_of_wealth_documented must equal TRUE, and funds_availability_verified equals TRUE, such that failure of any single validation component results in pre_trade_compliance_passed assuming FALSE boolean value and preventing workflow advancement to execution stages, with the system generating detailed diagnostic messages identifying which specific validation component failed to enable the Relationship Manager to take corrective action such as obtaining missing SOW documentation, waiting for funds transfer completion, or selecting an alternative product that passes suitability requirements.

        5.3.5 For mutual fund transactions specifically where product_category equals "Mutual_Fund" and pre_trade_compliance_passed confirms TRUE boolean value, the system must proceed to calculate precise financial metrics beginning with computation of the front_end_load_amount expressed in United States Dollars by multiplying the subscription_amount by the product's front_end_load_percentage and dividing by one hundred with results rounded to two decimal places using standard banker's rounding convention (round half to even), subsequently calculating the net_investment_amount representing the actual capital deployed into the fund by subtracting the front_end_load_amount from the subscription_amount, utilizing the current_nav value retrieved from the product master database (which itself is updated nightly through automated feeds from fund administrators or sourced real-time from market_data_source platforms during trading hours) to compute the estimated_units that will be allocated to the client's portfolio by dividing the net_investment_amount by the current_nav with precision extended to four decimal places to ensure accurate unit allocation given that mutual fund units are typically fractional, and additionally calculating an estimated annual_management_fee_amount by multiplying the net_investment_amount by the annual_management_fee_percentage and dividing by one hundred to project the first year's ongoing fee burden, with all calculated financial metrics recorded in their respective fields and presented to the client in a comprehensive Fee Impact Sheet disclosure summary enabling transparent understanding of all costs associated with the investment prior to final authorization.

        5.4 Client Authorization and Transaction Execution Phase

        5.4.1 With all pre-trade compliance validations successfully passed yielding pre_trade_compliance_passed equals TRUE and all financial calculations completed including front_end_load_amount, net_investment_amount, estimated_units, and annual_management_fee_amount determined and disclosed, the Relationship Manager must present the complete transaction summary to the client for final review and authorization, utilizing the Wealth Management System's transaction preview interface to display a consolidated summary screen showing the product_name, subscription_amount, front_end_load_amount with explicit notation of the percentage and dollar impact, net_investment_amount representing actual capital invested, estimated_units based on current_nav with clear disclaimer that final unit allocation will be determined based on the official NAV calculated at the close of business on the transaction date which may differ from the estimated NAV displayed, annual_management_fee_amount with explanation of ongoing fee structure, settlement_account_number from which funds will be debited, expected settlement_date typically following T+1 (trade date plus one business day) or T+2 settlement cycles depending on product category and fund house processing rules, and all relevant risk disclosures and terms and conditions, requiring the client to explicitly confirm their understanding and acceptance of all displayed terms before authorization is granted.

        5.4.2 Upon client confirmation of transaction review and expression of intent to proceed, the system must invoke the configured client_authorization_method framework to capture legally binding consent whereby if the method is designated as "OTP" (One-Time Password), the system generates a six-digit numerical code transmitted to the client's registered mobile telephone number via SMS or banking application push notification requiring the client to manually enter the received code into the transaction interface within a validity window of three hundred seconds (five minutes) before code expiration; if the method is "Biometric" the system interfaces with fingerprint scanner, facial recognition camera, or iris scanning hardware to capture the client's biometric signature comparing the captured template against the reference template stored during account opening with matching threshold typically set at ninety-five percent confidence level or higher; if the method is "Physical_Signature" the client must apply their handwritten signature to a printed transaction document or deal ticket with the RM verifying signature consistency against the specimen signature on file; or if the method is "Digital_Signature" the client utilizes their personal digital certificate issued under Public Key Infrastructure (PKI) framework to cryptographically sign the transaction data creating a non-repudiable authorization record, with the successful completion of authorization via any of these methods recorded by capturing the precise client_authorization_timestamp in ISO 8601 format with timezone and millisecond precision establishing the exact moment of legally binding consent.

        5.4.3 Immediately following successful client authorization capture, the system must evaluate whether the transaction involves a structured product requiring a regulatory cooling-off period by examining the cooling_off_period_required boolean field, such that for transactions where cooling_off_period_required equals FALSE (typically mutual fund and standard bond purchases), the system proceeds directly to immediate execution operations, whereas for transactions where cooling_off_period_required equals TRUE (typically structured notes, dual currency investments, and principal protected notes in jurisdictions mandating investor protection cooling-off rights), the system must divert the transaction to a deferred execution workflow by setting the transaction_status field to "Pending_Cooling_Off," recording the cooling_off_period_hours duration (commonly twenty-four or forty-eight hours depending on product complexity and local regulations), calculating a cooling-off expiry timestamp by adding the cooling_off_period_hours to the client_authorization_timestamp thereby establishing the earliest moment at which execution may proceed, and placing the transaction into an automated batch queue monitored by scheduled system jobs that execute at hourly intervals to identify all transactions whose cooling-off expiry timestamp has passed and whose status indicates no client cancellation request has been received, at which point the system automatically updates transaction_status from "Pending_Cooling_Off" to "Ready_for_Execution" and routes the transaction to the appropriate execution subsystem.

        5.4.4 For transactions where cooling-off requirements are not applicable or where the cooling-off period has successfully elapsed without client cancellation, the system must proceed to transaction execution operations which vary based on product_category such that for product_category equal to "Mutual_Fund," the Wealth Management System interfaces with the fund house order routing gateway to transmit an electronic purchase order containing the client_id mapped to the fund house's investor account number, product_code mapped to the fund's ISIN or local fund identifier, net_investment_amount as the subscription value, estimated_units for reference purposes with notation that final allocation will be NAV-based, settlement_account_number details for fund receipt, and transaction_timestamp establishing trade date, with the fund house's automated order management system acknowledging receipt and returning a preliminary order confirmation that the system stores as the trade_reference_number serving as the primary tracking identifier for all downstream settlement and reconciliation activities; whereas for product_category equal to "Structured_Note," "Dual_Currency_Investment," or "Principal_Protected_Note," the Wealth Management System creates a structured product order record and transmits this via secure messaging protocols to the bank's Treasury or Global Markets desk which functions as the internal counterparty or arranger for such instruments, with the Treasury desk's trading personnel reviewing the order parameters including notional amount, tenor, strike prices, barrier levels, and underlying asset specifications before manually confirming acceptance and execution which triggers generation of a contract_note_number representing the formal trade confirmation documentation that includes all economic terms, value_date establishing when the structured note becomes effective, pricing parameters, and payoff formulas that the system associates with the transaction record.

        5.4.5 Concurrent with order transmission to external fund houses or internal Treasury desks, the system must execute settlement account debit operations by generating an immediate debit instruction to the Core Banking System specifying the settlement_account_number, the full subscription_amount (not merely net_investment_amount, as the gross amount including fees must be debited), a descriptive narration field containing transaction_id and product_name for client statement clarity, and value date designation matching the transaction execution date, with the Core Banking System returning a debit confirmation reference number that validates successful fund reservation or movement from available balance to blocked or transit status pending final settlement with the fund house or product issuer, and upon successful debit confirmation, the system updates the available_balance field in the transaction record to reflect the post-transaction balance status, sets the transaction_status field to "Executed" indicating successful completion of all execution operations, records the execution_timestamp capturing the exact date and time when execution operations concluded, calculates the settlement_date based on product-specific settlement cycles (T+1 for most mutual funds, T+2 for certain offshore funds, T+0 or T+1 for structured products depending on Treasury desk processing), and critically evaluates whether the entire transaction lifecycle from initiation through execution occurred without any manual intervention, exception handling, workflow rerouting, or data correction by setting the straight_through_processing_flag to TRUE only when zero manual touchpoints occurred or to FALSE if any Operations or Compliance personnel intervened at any stage thereby providing key operational efficiency metrics.

        5.5 Post-Execution Compliance Review and Documentation Phase

        5.5.1 Following successful transaction execution with transaction_status updated to "Executed" and all primary processing operations completed, the system must automatically route a subset of transactions to mandatory post-execution compliance review based on risk-based selection criteria whereby all transactions where audio_video_recording_required equals TRUE indicating complex product sales must be queued for compliance review, all transactions where suitability_mismatch_flag equals TRUE indicating risk override scenarios must be reviewed, all transactions where transaction_threshold_exceeded equals TRUE indicating large-value orders must be sampled, and additionally a statistically random sample of standard transactions equating to typically five percent of daily transaction volume must be selected to ensure ongoing quality monitoring of routine sales practices, with all selected transactions routed to the assigned compliance_officer_id whose workqueue interface displays the transaction details, provides direct playback access to audio_video_recording_id files where applicable, presents the complete audit trail of system decision points and validation outcomes, and enables the compliance officer to review for potential process violations, documentation deficiencies, inadequate risk disclosure, or signs of client pressure or miss-selling behaviors.

        5.5.2 The assigned Compliance Officer executing post-trade review must systematically evaluate each transaction against a standardized compliance checklist covering verification that disclosure_documents_presented list includes all mandatory materials appropriate to the product_category, confirmation that client_authorization_timestamp exists and authorization method was properly executed, validation that for any suitability_mismatch_flag equals TRUE scenario the associated mismatch_waiver_document_id is present and supervisor_approval_timestamp confirms supervisory review occurred, assessment of audio_video_recording_id content where applicable to verify that the Relationship Manager delivered the mandatory risk disclosure script in clear and comprehensible language without minimizing downside scenarios and that the client verbally acknowledged understanding without evidence of confusion or hesitation that might suggest insufficient comprehension, and evaluation of overall transaction coherence relative to client_age, client_annual_income, client_net_worth, and stated investment objectives to identify potentially unsuitable sales that passed automated screening but exhibit qualitative concerns, with the outcome of this review process documented through entry of compliance_review_notes text field containing detailed observations and conclusions, and critically the setting of compliance_defect_flag to TRUE boolean value if any material deficiencies are identified that constitute process violations requiring corrective action or to FALSE if the transaction is determined fully compliant with all regulatory and internal policy requirements.

        5.5.3 For any transaction where compliance_defect_flag assumes TRUE boolean value indicating identification of process violations or documentation deficiencies, the Compliance Officer must initiate corrective action workflows which may include requiring the Relationship Manager to obtain supplementary documentation from the client such as additional risk acknowledgment forms or missing SOW records, scheduling a mandatory follow-up client contact to address disclosure gaps or confirm understanding of product risks that may not have been adequately communicated during initial sale, imposing transaction reversal or cancellation where severe suitability violations or material misrepresentation is identified within allowable reversal timeframes typically extending three to five business days post-execution, generating formal incident reports that are escalated to senior management and legal/compliance committees documenting the nature of the defect and remediation actions taken, and where patterns of defects are associated with specific relationship_manager_id personnel, triggering performance management interventions ranging from mandatory retraining on product knowledge and sales process requirements to suspension of investment sales privileges pending demonstration of competency improvement, with all corrective actions tracked through case management systems and recorded in the transaction_id audit trail to establish complete regulatory accountability.

        5.5.4 Parallel to compliance review processes, the Operations or Back Office team identified by operational processing queues must execute post-execution reconciliation and settlement monitoring activities by comparing the trade_reference_number or contract_note_number generated during execution against subsequent confirmation messages received from fund houses, custodian banks, or internal Treasury systems to verify that external parties have accurately booked the transaction with matching details including client account references, product identifiers, subscription amounts, unit allocations based on official NAV calculations, and settlement_date expectations, with any discrepancies between internally recorded transaction parameters and external confirmations flagged as exceptions requiring investigation and resolution through formal breaks management processes that may involve contacting fund house operations teams, submitting transaction amendment requests, or in severe cases unwinding and rebooking transactions to correct errors, and once full reconciliation is achieved confirming that the estimated_units calculated during pre-execution based on preliminary current_nav align within acceptable tolerance ranges (typically plus or minus zero point five percent) with the final units allocated by the fund house based on official end-of-day NAV calculations, the Operations team updates the transaction record with final unit quantities and confirms that settlement accounting entries have been accurately posted to both the client's investment portfolio custody account and the bank's omnibus accounts with external fund houses or custodians.

        5.5.5 As the culminating step of the post-execution phase, the system must generate comprehensive client confirmation documentation by producing a trade confirmation letter or contract note that includes transaction_id for reference, client_name and client_id for identification, product_name and product_code for clarity on what was purchased, subscription_amount showing the total funds committed, front_end_load_amount transparently disclosing upfront fees charged, net_investment_amount showing capital actually deployed, estimated_units allocated pending final NAV confirmation with explicit notation that final unit allocation may vary slightly, current_nav used for estimation purposes, settlement_date indicating when the transaction will be finalized, settlement_account_number debited, trade_reference_number or contract_note_number for tracking purposes, comprehensive fee disclosures reiterating both front_end_load_percentage and annual_management_fee_percentage with projected annual_management_fee_amount, and regulatory disclosures including cooling-off rights if applicable, complaint procedures, and investor protection scheme information, with this confirmation document delivered to the client through their preferred communication channel whether postal mail, secure email, mobile application notification, or online banking message center within the regulatory-mandated timeframe typically requiring delivery within one business day of execution, and the system records the confirmation delivery timestamp and method to demonstrate compliance with client communication obligations, simultaneously calculating the total_processing_time_minutes field by computing the elapsed time in minutes between the original transaction_timestamp when the process initiated and the execution_timestamp when execution operations completed, providing a quantitative metric of processing efficiency that feeds into operational dashboards tracking cycle times and identifying process bottlenecks requiring optimization attention.

        6. Output

        Upon successful completion of all procedural stages encompassing Risk Profile Assessment and Establishment, Investment Product Selection and Suitability Validation, Pre-Trade Compliance and Financial Validation, Client Authorization and Transaction Execution, and Post-Execution Compliance Review and Documentation, the Banking Invest Process produces a comprehensive set of deterministic outputs recorded within the transaction data structure and associated systems enabling full auditability and operational measurement. The primary quantitative outputs include the calculated_risk_profile classification assuming one of three discrete values ("Conservative," "Balanced," or "Aggressive") derived through algorithmic scoring of risk_questionnaire_total_score with the corresponding risk_profile_expiry_date timestamp establishing validity period, the investable_status field updated to "Active" enabling future transaction processing, the suitability_check_passed boolean definitively indicating whether product selection aligns with client risk parameters, the pre_trade_compliance_passed boolean aggregating all validation outcomes, the precise financial calculations comprising front_end_load_amount expressed in United States Dollars with two decimal precision, net_investment_amount representing actual invested capital, estimated_units calculated to four decimal precision through NAV-based division, and annual_management_fee_amount projecting ongoing cost burden. Transaction execution outputs include the system-generated trade_reference_number or contract_note_number serving as permanent tracking identifiers, the definitive transaction_status value progressing through state transitions from "Pending" through "Pending_Cooling_Off" if applicable to final "Executed" state for successful completions or "Cancelled" or "Failed" states for unsuccessful outcomes, the execution_timestamp providing precise temporal record of transaction consummation, the settlement_date establishing when final settlement will occur typically following T+1 or T+2 cycles, and the straight_through_processing_flag boolean indicating whether zero-touch processing was achieved contributing to STP rate metrics calculated as the percentage of transactions with this flag set to TRUE. Compliance and quality outputs encompass the compliance_defect_flag boolean identifying transactions requiring corrective action, the detailed compliance_review_notes text field documenting officer observations, the supervisor_approval_timestamp for escalated transactions requiring management authorization, and the audio_video_recording_id providing immutable evidence of sales process execution for complex products. Operational efficiency metrics output includes the total_processing_time_minutes integer value enabling calculation of average cycle time performance indicators, the funds_availability_verified boolean confirming settlement readiness, and the available_balance post-transaction value enabling liquidity monitoring. Risk control outputs definitively record the suitability_mismatch_flag boolean and associated mismatch_waiver_document_id for exception tracking, the aml_sanctions_screening_status confirming compliance with sanctions regulations, the source_of_wealth_documented boolean validating enhanced due diligence for high-value transactions, and the transaction_threshold_exceeded flag identifying transactions subject to enhanced monitoring. Collectively these outputs enable computation of key performance indicators including Sales Conversion Rate calculated as (count of transactions with transaction_status equals "Executed") divided by (count of all initiated transactions) multiplied by one hundred percent, Straight-Through Processing Rate calculated as (count of transactions with straight_through_processing_flag equals TRUE) divided by (count of transactions with transaction_status equals "Executed") multiplied by one hundred percent, Suitability Pass Rate calculated as (count of transactions with suitability_check_passed equals TRUE and suitability_mismatch_flag equals FALSE) divided by (count of all transactions) multiplied by one hundred percent, Average Ticket Size calculated as sum of all subscription_amount values divided by count of executed transactions, Compliance Defect Ratio calculated as (count of transactions with compliance_defect_flag equals TRUE) divided by (count of transactions reviewed by compliance) multiplied by one hundred percent, and Average Processing Time calculated as mean of all total_processing_time_minutes values across executed transactions, with all metrics stored in data warehouse structures enabling trend analysis, performance benchmarking against industry standards, identification of process improvement opportunities, and regulatory reporting demonstrating adherence to investor protection obligations and operational excellence standards mandated by banking supervisory authorities.

        </SOP>'''

# --- 定义新的工具组 ---

# ==================== 数据存储层 ====================
# Global data storage
class DataStore:
    def __init__(self):
        self.data = []
        
    def parse_entry(self, entry_list: List[str]) -> Dict[str, Any]:
        """Parse a data entry from list format to dictionary"""
        entry_dict = {}
        for item in entry_list:
            if ':' in item:
                key, value = item.split(':', 1)
                entry_dict[key] = value
        return entry_dict
    
    def load_data(self, raw_data: List[List[str]]):
        """Load data from raw format"""
        self.data = [self.parse_entry(entry) for entry in raw_data]
    
    def get_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data entry by transaction_id"""
        for entry in self.data:
            if entry.get('transaction_id') == transaction_id:
                return entry
        return None
    
    def get_by_client_id(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data entry by client_id"""
        for entry in self.data:
            if entry.get('client_id') == client_id:
                return entry
        return None
    
    def update_entry(self, transaction_id: str, updates: Dict[str, Any]):
        """Update a data entry"""
        for entry in self.data:
            if entry.get('transaction_id') == transaction_id:
                entry.update(updates)
                return True
        return False
    
    def get_by_application_id(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve data entry by application_id (alias for transaction_id)"""
        return self.get_by_transaction_id(application_id)
        
def get_by_transaction_id(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve data entry by transaction_id"""
    for entry in data_store.data:
        if entry.get('transaction_id') == transaction_id:
            return entry
    return None
def get_by_application_id(application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve data entry by application_id (alias for transaction_id)"""
    return get_by_transaction_id(application_id)
# Initialize global data store
data_store = DataStore()

# Load sample data
raw_data = [
    # ['transaction_id:INV_2025041801317', 'client_id:CLI_867905', 'client_name:Christopher Bailey', 'client_age:40', 'client_annual_income:132000.00', 'client_net_worth:485000.00', 'client_investment_experience_years:10', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Christopher Lee', 'transaction_timestamp:2025-04-18T11:45:28Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:48000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_8679056789', 'available_balance:62000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Mutual_Funds_ETFs"}', 'risk_questionnaire_total_score:69', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-03-25T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T12:18:55Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Bloomberg', 'branch_code:BR_SF_008', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801322', 'client_id:CLI_412450', 'client_name:Laura Phillips', 'client_age:39', 'client_annual_income:138000.00', 'client_net_worth:525000.00', 'client_investment_experience_years:11', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Patricia Moore', 'transaction_timestamp:2025-04-18T09:52:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:55000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_4124501234', 'available_balance:72000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Real_Estate"}', 'risk_questionnaire_total_score:73', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-01-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T10:25:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Bloomberg', 'branch_code:BR_ATL_009', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801344', 'client_id:CLI_834682', 'client_name:Karen Fisher', 'client_age:33', 'client_annual_income:102000.00', 'client_net_worth:295000.00', 'client_investment_experience_years:6', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Patricia Moore', 'transaction_timestamp:2025-04-18T11:18:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:39000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_8346824567', 'available_balance:48000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Mutual_Funds"}', 'risk_questionnaire_total_score:55', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T11:52:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Internal_System', 'branch_code:BR_ATL_009', 'channel:Phone_Banking', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801341', 'client_id:CLI_501359', 'client_name:Steven Rogers', 'client_age:36', 'client_annual_income:125000.00', 'client_net_worth:395000.00', 'client_investment_experience_years:8', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Susan Chang', 'transaction_timestamp:2025-04-18T12:28:52Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:44000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_5013591234', 'available_balance:58000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_ETFs_Mutual_Funds"}', 'risk_questionnaire_total_score:69', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-12-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T13:05:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Bloomberg', 'branch_code:BR_ATL_009', 'channel:Mobile_App', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801403', 'client_id:CLI_923571', 'client_name:Rebecca Foster', 'client_age:39', 'client_annual_income:148000.00', 'client_net_worth:568000.00', 'client_investment_experience_years:11', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T13:28:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:55000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_9235713456', 'available_balance:72000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_ETFs"}', 'risk_questionnaire_total_score:70', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-09-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T14:05:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7821', 'market_data_source:Bloomberg', 'branch_code:BR_NYC_001', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801419', 'client_id:CLI_589139', 'client_name:Dennis Bryant', 'client_age:53', 'client_annual_income:155000.00', 'client_net_worth:1185000.00', 'client_investment_experience_years:22', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_9988', 'relationship_manager_name:Lisa Anderson', 'transaction_timestamp:2025-04-18T15:08:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:77000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_5891399012', 'available_balance:92000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:57', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T15:42:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Reuters', 'branch_code:BR_CHI_015', 'channel:Branch', 'region:US_Midwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801263', 'client_id:CLI_323457', 'client_name:Emily Zhang', 'client_age:31', 'client_annual_income:105000.00', 'client_net_worth:265000.00', 'client_investment_experience_years:5', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_5577', 'relationship_manager_name:Matthew Johnson', 'transaction_timestamp:2025-04-18T15:18:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:38000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_3234578901', 'available_balance:45000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_ETFs_Crypto"}', 'risk_questionnaire_total_score:70', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-08-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T15:42:52Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Bloomberg', 'branch_code:BR_SF_008', 'channel:Mobile_App', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801411', 'client_id:CLI_701360', 'client_name:Mildred Cooper', 'client_age:66', 'client_annual_income:95000.00', 'client_net_worth:1885000.00', 'client_investment_experience_years:35', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Susan Chang', 'transaction_timestamp:2025-04-18T10:28:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:54000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_7013600123', 'available_balance:72000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Low","q5_previous_investments":"Bonds_Mutual_Funds_Annuities"}', 'risk_questionnaire_total_score:42', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-06-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T11:05:22Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Reuters', 'branch_code:BR_ATL_009', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801448', 'client_id:CLI_478031', 'client_name:Donna Ward', 'client_age:46', 'client_annual_income:192000.00', 'client_net_worth:1025000.00', 'client_investment_experience_years:17', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Christopher Lee', 'transaction_timestamp:2025-04-18T11:52:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:87000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_4780311234', 'available_balance:105000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Options_Real_Estate"}', 'risk_questionnaire_total_score:75', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-12-05T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T12:28:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Bloomberg', 'branch_code:BR_SF_008', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801319', 'client_id:CLI_089127', 'client_name:Daniel Park', 'client_age:33', 'client_annual_income:115000.00', 'client_net_worth:345000.00', 'client_investment_experience_years:7', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Susan Chang', 'transaction_timestamp:2025-04-18T16:08:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:43000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_0891278901', 'available_balance:55000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_ETFs_Options"}', 'risk_questionnaire_total_score:71', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-12-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T16:42:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Bloomberg', 'branch_code:BR_ATL_009', 'channel:Online_Banking', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801401', 'client_id:CLI_701359', 'client_name:Norma Sanders', 'client_age:51', 'client_annual_income:152000.00', 'client_net_worth:1185000.00', 'client_investment_experience_years:20', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_1122', 'relationship_manager_name:Angela Davis', 'transaction_timestamp:2025-04-18T15:08:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:79000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_7013593456', 'available_balance:95000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:55', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T15:45:22Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Reuters', 'branch_code:BR_HOU_018', 'channel:Branch', 'region:US_South', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801331', 'client_id:CLI_401349', 'client_name:Brian Cooper', 'client_age:38', 'client_annual_income:128000.00', 'client_net_worth:445000.00', 'client_investment_experience_years:9', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Christopher Lee', 'transaction_timestamp:2025-04-18T12:42:28Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:76000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_4013490123', 'available_balance:95000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:58', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-10-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T13:18:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Bloomberg', 'branch_code:BR_SF_008', 'channel:Online_Banking', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801376', 'client_id:CLI_256804', 'client_name:Gary Thompson', 'client_age:44', 'client_annual_income:172000.00', 'client_net_worth:725000.00', 'client_investment_experience_years:13', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_3342', 'relationship_manager_name:Robert Chen', 'transaction_timestamp:2025-04-18T15:18:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:64000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_2568046789', 'available_balance:78000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Options"}', 'risk_questionnaire_total_score:73', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-01-28T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T15:52:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6542', 'market_data_source:Bloomberg', 'branch_code:BR_LA_005', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801420', 'client_id:CLI_690250', 'client_name:Christina Alexander', 'client_age:63', 'client_annual_income:108000.00', 'client_net_worth:1525000.00', 'client_investment_experience_years:29', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_6677', 'relationship_manager_name:Thomas Wright', 'transaction_timestamp:2025-04-18T10:52:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:69000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_6902500123', 'available_balance:88000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Low","q5_previous_investments":"Bonds_Mutual_Funds_Annuities"}', 'risk_questionnaire_total_score:45', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-07-12T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T11:28:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Internal_System', 'branch_code:BR_BOS_020', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801272', 'client_id:CLI_212347', 'client_name:Sophie Dubois', 'client_age:34', 'client_annual_income:118000.00', 'client_net_worth:365000.00', 'client_investment_experience_years:7', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_5577', 'relationship_manager_name:Anthony Lewis', 'transaction_timestamp:2025-04-18T13:35:22Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:55000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_2123478901', 'available_balance:68000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:56', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T14:02:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Reuters', 'branch_code:BR_SEA_022', 'channel:Online_Banking', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801343', 'client_id:CLI_723571', 'client_name:Donald Stewart', 'client_age:45', 'client_annual_income:178000.00', 'client_net_worth:785000.00', 'client_investment_experience_years:15', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_4521', 'relationship_manager_name:Sarah Johnson', 'transaction_timestamp:2025-04-18T15:42:18Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:66000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_7235713456', 'available_balance:88000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Options"}', 'risk_questionnaire_total_score:74', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-02-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T16:18:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7821', 'market_data_source:Bloomberg', 'branch_code:BR_NYC_001', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801345', 'client_id:CLI_945793', 'client_name:Joseph Morris', 'client_age:58', 'client_annual_income:165000.00', 'client_net_worth:1425000.00', 'client_investment_experience_years:26', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_2244', 'relationship_manager_name:Andrew Bennett', 'transaction_timestamp:2025-04-18T14:25:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:94000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_9457935678', 'available_balance:122000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Real_Estate"}', 'risk_questionnaire_total_score:58', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-11-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T15:02:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Bloomberg', 'branch_code:BR_BOS_020', 'channel:Online_Banking', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801441', 'client_id:CLI_701363', 'client_name:Nicole Coleman', 'client_age:42', 'client_annual_income:188000.00', 'client_net_worth:825000.00', 'client_investment_experience_years:15', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_1122', 'relationship_manager_name:Angela Davis', 'transaction_timestamp:2025-04-18T08:28:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:78000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_7013633456', 'available_balance:95000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:62', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-25T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T09:05:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Reuters', 'branch_code:BR_HOU_018', 'channel:Branch', 'region:US_South', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801416', 'client_id:CLI_256806', 'client_name:Virginia Bennett', 'client_age:31', 'client_annual_income:98000.00', 'client_net_worth:245000.00', 'client_investment_experience_years:5', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_3342', 'relationship_manager_name:Robert Chen', 'transaction_timestamp:2025-04-18T11:08:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:32000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_2568066789', 'available_balance:42000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_401k_Mutual_Funds"}', 'risk_questionnaire_total_score:50', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-07-05T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T11:42:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Internal_System', 'branch_code:BR_LA_005', 'channel:Phone_Banking', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801266', 'client_id:CLI_656790', 'client_name:Anna Kowalczyk', 'client_age:43', 'client_annual_income:142000.00', 'client_net_worth:575000.00', 'client_investment_experience_years:12', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_8811', 'relationship_manager_name:Charles Wright', 'transaction_timestamp:2025-04-18T16:25:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:68000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_6567901234', 'available_balance:82000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:64', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-10-05T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T16:52:15Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Reuters', 'branch_code:BR_CHI_015', 'channel:Online_Banking', 'region:US_Midwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801410', 'client_id:CLI_690249', 'client_name:Arthur Richardson', 'client_age:44', 'client_annual_income:192000.00', 'client_net_worth:925000.00', 'client_investment_experience_years:16', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_9922', 'relationship_manager_name:Brian Taylor', 'transaction_timestamp:2025-04-18T13:42:18Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:84000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_6902499012', 'available_balance:105000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Options_Real_Estate"}', 'risk_questionnaire_total_score:76', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-11-28T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T14:18:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Bloomberg', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    # ['transaction_id:INV_2025041801361', 'client_id:CLI_701359', 'client_name:Angela Brown', 'client_age:57', 'client_annual_income:125000.00', 'client_net_worth:1875000.00', 'client_investment_experience_years:26', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_1122', 'relationship_manager_name:Angela Davis', 'transaction_timestamp:2025-04-18T15:52:28Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:72000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_7013595678', 'available_balance:95000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Annuities"}', 'risk_questionnaire_total_score:53', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-05T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T16:28:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Reuters', 'branch_code:BR_HOU_018', 'channel:Branch', 'region:US_South', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801323', 'client_id:CLI_523561', 'client_name:George Edwards', 'client_age:66', 'client_annual_income:88000.00', 'client_net_worth:1625000.00', 'client_investment_experience_years:32', 'client_liquidity_needs:High', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T08:28:52Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_022', 'product_name:Short-Term Bond Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:48000.00', 'front_end_load_percentage:0.75', 'annual_management_fee_percentage:0.65', 'current_nav:9.98', 'settlement_account_number:ACC_5235612345', 'available_balance:75000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"1-3_years","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:31', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-02-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T08:58:25Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_9900', 'market_data_source:Reuters', 'branch_code:BR_PHX_025', 'channel:Branch', 'region:US_Southwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801280', 'client_id:CLI_090125', 'client_name:Isabella Ferrari', 'client_age:48', 'client_annual_income:162000.00', 'client_net_worth:815000.00', 'client_investment_experience_years:18', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_4466', 'relationship_manager_name:Peter Collins', 'transaction_timestamp:2025-04-18T15:05:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:76000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_0901256789', 'available_balance:92000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Options_Real_Estate"}', 'risk_questionnaire_total_score:83', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-03-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T15:38:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6655', 'market_data_source:Bloomberg', 'branch_code:BR_NYC_001', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801355', 'client_id:CLI_145793', 'client_name:Jennifer Martinez', 'client_age:42', 'client_annual_income:178000.00', 'client_net_worth:695000.00', 'client_investment_experience_years:13', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_4521', 'relationship_manager_name:Sarah Johnson', 'transaction_timestamp:2025-04-18T09:32:15Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:62000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_1457936789', 'available_balance:78000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:59', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T10:05:42Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7821', 'market_data_source:Bloomberg', 'branch_code:BR_NYC_001', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801445', 'client_id:CLI_145798', 'client_name:Aaron Edwards', 'client_age:28', 'client_annual_income:68000.00', 'client_net_worth:125000.00', 'client_investment_experience_years:3', 'client_liquidity_needs:High', 'relationship_manager_id:RM_3344', 'relationship_manager_name:James Wilson', 'transaction_timestamp:2025-04-18T10:52:48Z', 'transaction_type:Risk_Profiling', 'product_code:', 'product_name:', 'product_risk_rating:', 'product_category:', 'subscription_amount:0.00', 'front_end_load_percentage:0.0', 'annual_management_fee_percentage:0.0', 'current_nav:0.0', 'settlement_account_number:ACC_1457988901', 'available_balance:14000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"18-30","q2_income_stability":"Stable","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Savings_401k"}', 'risk_questionnaire_total_score:42', 'previous_risk_profile:None', 'previous_risk_profile_expiry_date:', 'disclosure_documents_presented:[]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T11:18:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5566', 'market_data_source:Internal_System', 'branch_code:BR_DEN_011', 'channel:Online_Banking', 'region:US_Mountain', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801324', 'client_id:CLI_634672', 'client_name:Amanda Scott', 'client_age:30', 'client_annual_income:82000.00', 'client_net_worth:185000.00', 'client_investment_experience_years:4', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_3355', 'relationship_manager_name:Karen White', 'transaction_timestamp:2025-04-18T14:18:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:32000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_6346723456', 'available_balance:38000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"18-30","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Mutual_Funds"}', 'risk_questionnaire_total_score:48', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-10-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T14:48:25Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Internal_System', 'branch_code:BR_SEA_022', 'channel:Phone_Banking', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801235', 'client_id:CLI_123456', 'client_name:Maria Elena Rodriguez', 'client_age:32', 'client_annual_income:85000.00', 'client_net_worth:250000.00', 'client_investment_experience_years:3', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_3342', 'relationship_manager_name:Robert Chen', 'transaction_timestamp:2025-04-18T11:22:15Z', 'transaction_type:Risk_Profiling', 'product_code:', 'product_name:', 'product_risk_rating:', 'product_category:', 'subscription_amount:0.00', 'front_end_load_percentage:0.0', 'annual_management_fee_percentage:0.0', 'current_nav:0.0', 'settlement_account_number:ACC_9876543210', 'available_balance:35000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks"}', 'risk_questionnaire_total_score:52', 'previous_risk_profile:None', 'previous_risk_profile_expiry_date:', 'disclosure_documents_presented:[]', 'aml_sanctions_screening_status:Pending', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T11:35:48Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6542', 'market_data_source:Internal_System', 'branch_code:BR_LA_005', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801414', 'client_id:CLI_034684', 'client_name:Joyce Morgan', 'client_age:72', 'client_annual_income:68000.00', 'client_net_worth:1925000.00', 'client_investment_experience_years:38', 'client_liquidity_needs:High', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T12:08:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_022', 'product_name:Short-Term Bond Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:52000.00', 'front_end_load_percentage:0.75', 'annual_management_fee_percentage:0.65', 'current_nav:9.98', 'settlement_account_number:ACC_0346844567', 'available_balance:72000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"<1_year","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:29', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-02-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T12:42:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_9900', 'market_data_source:Internal_System', 'branch_code:BR_PHX_025', 'channel:Branch', 'region:US_Southwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801312', 'client_id:CLI_312350', 'client_name:Susan Miller', 'client_age:56', 'client_annual_income:125000.00', 'client_net_worth:1150000.00', 'client_investment_experience_years:23', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_4455', 'relationship_manager_name:Kevin Park', 'transaction_timestamp:2025-04-18T10:22:18Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:78000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_3123501234', 'available_balance:95000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:56', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T10:55:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6655', 'market_data_source:Reuters', 'branch_code:BR_SEA_022', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801313', 'client_id:CLI_423461', 'client_name:Benjamin Turner', 'client_age:34', 'client_annual_income:105000.00', 'client_net_worth:328000.00', 'client_investment_experience_years:6', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Susan Chang', 'transaction_timestamp:2025-04-18T12:08:52Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:38000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_4234612345', 'available_balance:48000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_ETFs"}', 'risk_questionnaire_total_score:67', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-08-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T12:38:25Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Bloomberg', 'branch_code:BR_ATL_009', 'channel:Mobile_App', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801271', 'client_id:CLI_101236', 'client_name:George Williams', 'client_age:61', 'client_annual_income:112000.00', 'client_net_worth:1685000.00', 'client_investment_experience_years:27', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_4466', 'relationship_manager_name:Elizabeth King', 'transaction_timestamp:2025-04-18T09:52:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:75000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_1012367890', 'available_balance:108000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Low","q5_previous_investments":"Bonds_Mutual_Funds_Annuities"}', 'risk_questionnaire_total_score:45', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-06-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T10:25:48Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Internal_System', 'branch_code:BR_BOS_020', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801450', 'client_id:CLI_690253', 'client_name:Jack Griffin', 'client_age:57', 'client_annual_income:182000.00', 'client_net_worth:1425000.00', 'client_investment_experience_years:26', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_9922', 'relationship_manager_name:Brian Taylor', 'transaction_timestamp:2025-04-18T08:18:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:92000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_6902533456', 'available_balance:108000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Real_Estate"}', 'risk_questionnaire_total_score:61', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-15T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T08:52:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Reuters', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801432', 'client_id:CLI_812463', 'client_name:Larry Barnes', 'client_age:43', 'client_annual_income:158000.00', 'client_net_worth:625000.00', 'client_investment_experience_years:14', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_2244', 'relationship_manager_name:Andrew Bennett', 'transaction_timestamp:2025-04-18T12:18:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_GROW_099', 'product_name:International Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:65000.00', 'front_end_load_percentage:2.75', 'annual_management_fee_percentage:1.95', 'current_nav:24.56', 'settlement_account_number:ACC_8124633456', 'available_balance:82000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Bonds_Options"}', 'risk_questionnaire_total_score:73', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2026-11-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T12:52:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Bloomberg', 'branch_code:BR_BOS_020', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801326', 'client_id:CLI_856894', 'client_name:Samantha Torres', 'client_age:41', 'client_annual_income:148000.00', 'client_net_worth:645000.00', 'client_investment_experience_years:13', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_2244', 'relationship_manager_name:Andrew Bennett', 'transaction_timestamp:2025-04-18T11:32:18Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:84000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_8568945678', 'available_balance:102000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"41-50","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:59', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-25T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Biometric', 'client_authorization_timestamp:2025-04-18T12:08:42Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Bloomberg', 'branch_code:BR_BOS_020', 'channel:Branch', 'region:US_Northeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801365', 'client_id:CLI_145793', 'client_name:Patricia Campbell', 'client_age:72', 'client_annual_income:85000.00', 'client_net_worth:1950000.00', 'client_investment_experience_years:38', 'client_liquidity_needs:High', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T08:22:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_055', 'product_name:Conservative Income Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:62000.00', 'front_end_load_percentage:1.5', 'annual_management_fee_percentage:0.95', 'current_nav:10.85', 'settlement_account_number:ACC_1457939012', 'available_balance:98000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"<1_year","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:27', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-01-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T08:55:12Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_9900', 'market_data_source:Internal_System', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801342', 'client_id:CLI_612460', 'client_name:Christine Perry', 'client_age:60', 'client_annual_income:118000.00', 'client_net_worth:1585000.00', 'client_investment_experience_years:28', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_9922', 'relationship_manager_name:Brian Taylor', 'transaction_timestamp:2025-04-18T09:08:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:62000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_6124602345', 'available_balance:85000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:49', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-06-12T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T09:42:28Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Internal_System', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801284', 'client_id:CLI_434570', 'client_name:Fatima Hassan', 'client_age:64', 'client_annual_income:108000.00', 'client_net_worth:1765000.00', 'client_investment_experience_years:29', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_8811', 'relationship_manager_name:Timothy Wood', 'transaction_timestamp:2025-04-18T09:18:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_055', 'product_name:Conservative Income Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:52000.00', 'front_end_load_percentage:1.5', 'annual_management_fee_percentage:0.95', 'current_nav:10.85', 'settlement_account_number:ACC_4345701234', 'available_balance:118000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"1-3_years","q4_loss_tolerance":"Low","q5_previous_investments":"Bonds_CDs_Annuities"}', 'risk_questionnaire_total_score:35', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-01-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T09:45:18Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8821', 'market_data_source:Internal_System', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801346', 'client_id:CLI_056804', 'client_name:Martha Lewis', 'client_age:70', 'client_annual_income:75000.00', 'client_net_worth:2015000.00', 'client_investment_experience_years:38', 'client_liquidity_needs:High', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T08:18:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_022', 'product_name:Short-Term Bond Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:58000.00', 'front_end_load_percentage:0.75', 'annual_management_fee_percentage:0.65', 'current_nav:9.98', 'settlement_account_number:ACC_0568046789', 'available_balance:98000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"<1_year","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:29', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-01-28T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T08:52:25Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_9900', 'market_data_source:Reuters', 'branch_code:BR_PHX_025', 'channel:Branch', 'region:US_Southwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801308', 'client_id:CLI_878916', 'client_name:Mei Zhang', 'client_age:37', 'client_annual_income:142000.00', 'client_net_worth:465000.00', 'client_investment_experience_years:9', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_7799', 'relationship_manager_name:Patricia Moore', 'transaction_timestamp:2025-04-18T11:18:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_TECH_001', 'product_name:Global Tech Growth Fund', 'product_risk_rating:Growth', 'product_category:Mutual_Fund', 'subscription_amount:52000.00', 'front_end_load_percentage:2.5', 'annual_management_fee_percentage:1.75', 'current_nav:15.42', 'settlement_account_number:ACC_8789167890', 'available_balance:65000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"10+_years","q4_loss_tolerance":"High","q5_previous_investments":"Stocks_Mutual_Funds_Options"}', 'risk_questionnaire_total_score:70', 'previous_risk_profile:Aggressive', 'previous_risk_profile_expiry_date:2027-02-28T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T11:48:52Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Bloomberg', 'branch_code:BR_SF_008', 'channel:Mobile_App', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801451', 'client_id:CLI_701364', 'client_name:Jean Diaz', 'client_age:69', 'client_annual_income:92000.00', 'client_net_worth:2125000.00', 'client_investment_experience_years:38', 'client_liquidity_needs:High', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Susan Chang', 'transaction_timestamp:2025-04-18T14:08:42Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_022', 'product_name:Short-Term Bond Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:49000.00', 'front_end_load_percentage:0.75', 'annual_management_fee_percentage:0.65', 'current_nav:9.98', 'settlement_account_number:ACC_7013644567', 'available_balance:68000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"<1_year","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:25', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-01-08T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T14:42:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Internal_System', 'branch_code:BR_ATL_009', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801409', 'client_id:CLI_589138', 'client_name:Rita Howard', 'client_age:52', 'client_annual_income:142000.00', 'client_net_worth:965000.00', 'client_investment_experience_years:21', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_8811', 'relationship_manager_name:Rachel Martinez', 'transaction_timestamp:2025-04-18T08:42:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:67000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_5891388901', 'available_balance:85000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Low","q5_previous_investments":"Bonds_Mutual_Funds_Annuities"}', 'risk_questionnaire_total_score:47', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-07-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T09:18:48Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8899', 'market_data_source:Internal_System', 'branch_code:BR_CHI_015', 'channel:Branch', 'region:US_Midwest', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801330', 'client_id:CLI_290238', 'client_name:Nicole Foster', 'client_age:62', 'client_annual_income:108000.00', 'client_net_worth:1545000.00', 'client_investment_experience_years:31', 'client_liquidity_needs:High', 'relationship_manager_id:RM_5566', 'relationship_manager_name:Michelle Green', 'transaction_timestamp:2025-04-18T08:35:15Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_055', 'product_name:Conservative Income Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:52000.00', 'front_end_load_percentage:1.5', 'annual_management_fee_percentage:0.95', 'current_nav:10.85', 'settlement_account_number:ACC_2902389012', 'available_balance:78000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"1-3_years","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:34', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2026-12-22T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T09:08:38Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_8821', 'market_data_source:Reuters', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801366', 'client_id:CLI_256804', 'client_name:Steven Parker', 'client_age:25', 'client_annual_income:54000.00', 'client_net_worth:68000.00', 'client_investment_experience_years:1', 'client_liquidity_needs:High', 'relationship_manager_id:RM_3344', 'relationship_manager_name:James Wilson', 'transaction_timestamp:2025-04-18T10:52:33Z', 'transaction_type:Risk_Profiling', 'product_code:', 'product_name:', 'product_risk_rating:', 'product_category:', 'subscription_amount:0.00', 'front_end_load_percentage:0.0', 'annual_management_fee_percentage:0.0', 'current_nav:0.0', 'settlement_account_number:ACC_2568043456', 'available_balance:8500.00', 'risk_questionnaire_responses:{"q1_age_bracket":"18-30","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Savings_401k"}', 'risk_questionnaire_total_score:43', 'previous_risk_profile:None', 'previous_risk_profile_expiry_date:', 'disclosure_documents_presented:[]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T11:08:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5566', 'market_data_source:Internal_System', 'branch_code:BR_DEN_011', 'channel:Mobile_App', 'region:US_Mountain', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801302', 'client_id:CLI_212350', 'client_name:Nicole Stewart', 'client_age:40', 'client_annual_income:138000.00', 'client_net_worth:525000.00', 'client_investment_experience_years:10', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_8811', 'relationship_manager_name:Raymond Cooper', 'transaction_timestamp:2025-04-18T14:05:28Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:66000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_2123501234', 'available_balance:78000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:60', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T14:35:45Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Reuters', 'branch_code:BR_HOU_018', 'channel:Mobile_App', 'region:US_South', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801356', 'client_id:CLI_256804', 'client_name:Kevin Anderson', 'client_age:28', 'client_annual_income:67000.00', 'client_net_worth:125000.00', 'client_investment_experience_years:3', 'client_liquidity_needs:High', 'relationship_manager_id:RM_3342', 'relationship_manager_name:Robert Chen', 'transaction_timestamp:2025-04-18T14:22:48Z', 'transaction_type:Risk_Profiling', 'product_code:', 'product_name:', 'product_risk_rating:', 'product_category:', 'subscription_amount:0.00', 'front_end_load_percentage:0.0', 'annual_management_fee_percentage:0.0', 'current_nav:0.0', 'settlement_account_number:ACC_2568047890', 'available_balance:28000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"18-30","q2_income_stability":"Stable","q3_investment_horizon":"3-5_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks"}', 'risk_questionnaire_total_score:40', 'previous_risk_profile:None', 'previous_risk_profile_expiry_date:', 'disclosure_documents_presented:[]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T14:38:22Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6542', 'market_data_source:Internal_System', 'branch_code:BR_LA_005', 'channel:Online_Banking', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801430', 'client_id:CLI_690251', 'client_name:Terry Powell', 'client_age:55', 'client_annual_income:172000.00', 'client_net_worth:1325000.00', 'client_investment_experience_years:24', 'client_liquidity_needs:Low', 'relationship_manager_id:RM_9922', 'relationship_manager_name:Brian Taylor', 'transaction_timestamp:2025-04-18T15:52:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:85000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_6902511234', 'available_balance:102000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds"}', 'risk_questionnaire_total_score:56', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T16:28:48Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_5544', 'market_data_source:Reuters', 'branch_code:BR_MIA_012', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801442', 'client_id:CLI_812464', 'client_name:Philip Ross', 'client_age:65', 'client_annual_income:85000.00', 'client_net_worth:1725000.00', 'client_investment_experience_years:34', 'client_liquidity_needs:High', 'relationship_manager_id:RM_4455', 'relationship_manager_name:Kevin Park', 'transaction_timestamp:2025-04-18T15:08:48Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BOND_055', 'product_name:Conservative Income Fund', 'product_risk_rating:Conservative', 'product_category:Mutual_Fund', 'subscription_amount:56000.00', 'front_end_load_percentage:1.5', 'annual_management_fee_percentage:0.95', 'current_nav:10.85', 'settlement_account_number:ACC_8124644567', 'available_balance:75000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"61+","q2_income_stability":"Retired_Fixed","q3_investment_horizon":"1-3_years","q4_loss_tolerance":"Very_Low","q5_previous_investments":"Bonds_CDs_Money_Market"}', 'risk_questionnaire_total_score:32', 'previous_risk_profile:Conservative', 'previous_risk_profile_expiry_date:2027-02-18T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Physical_Signature', 'client_authorization_timestamp:2025-04-18T15:42:35Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6655', 'market_data_source:Internal_System', 'branch_code:BR_SEA_022', 'channel:Branch', 'region:US_West', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801404', 'client_id:CLI_034683', 'client_name:Samuel Griffin', 'client_age:54', 'client_annual_income:188000.00', 'client_net_worth:1325000.00', 'client_investment_experience_years:24', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_8899', 'relationship_manager_name:David Kim', 'transaction_timestamp:2025-04-18T10:42:35Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_BAL_045', 'product_name:Balanced Growth and Income Fund', 'product_risk_rating:Balanced', 'product_category:Mutual_Fund', 'subscription_amount:86000.00', 'front_end_load_percentage:2.0', 'annual_management_fee_percentage:1.45', 'current_nav:18.75', 'settlement_account_number:ACC_0346838901', 'available_balance:102000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"51-60","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds_Mutual_Funds_Real_Estate"}', 'risk_questionnaire_total_score:60', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-08-28T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:Digital_Signature', 'client_authorization_timestamp:2025-04-18T11:18:48Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_6677', 'market_data_source:Reuters', 'branch_code:BR_ATL_009', 'channel:Branch', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:'],
    ['transaction_id:INV_2025041801277', 'client_id:CLI_767892', 'client_name:Luis Hernandez', 'client_age:36', 'client_annual_income:96000.00', 'client_net_worth:242000.00', 'client_investment_experience_years:6', 'client_liquidity_needs:Medium', 'relationship_manager_id:RM_1133', 'relationship_manager_name:Nancy Roberts', 'transaction_timestamp:2025-04-18T14:52:45Z', 'transaction_type:Mutual_Fund_Subscription', 'product_code:MF_MOD_033', 'product_name:Moderate Allocation Fund', 'product_risk_rating:Moderate', 'product_category:Mutual_Fund', 'subscription_amount:32000.00', 'front_end_load_percentage:1.75', 'annual_management_fee_percentage:1.25', 'current_nav:12.34', 'settlement_account_number:ACC_7678923456', 'available_balance:42000.00', 'risk_questionnaire_responses:{"q1_age_bracket":"31-40","q2_income_stability":"Stable","q3_investment_horizon":"5-10_years","q4_loss_tolerance":"Moderate","q5_previous_investments":"Stocks_Bonds"}', 'risk_questionnaire_total_score:51', 'previous_risk_profile:Balanced', 'previous_risk_profile_expiry_date:2026-09-10T00:00:00Z', 'disclosure_documents_presented:["Fund_Fact_Sheet","Prospectus","KID"]', 'aml_sanctions_screening_status:Clear', 'source_of_wealth_documented:True', 'transaction_threshold_exceeded:False', 'monitoring_threshold_amount:100000.00', 'audio_video_recording_required:False', 'audio_video_recording_id:', 'cooling_off_period_required:False', 'cooling_off_period_hours:0', 'client_authorization_method:OTP', 'client_authorization_timestamp:2025-04-18T15:18:22Z', 'mismatch_waiver_required:False', 'mismatch_waiver_document_id:', 'supervisor_approval_required:False', 'supervisor_id:', 'compliance_officer_id:CO_7788', 'market_data_source:Reuters', 'branch_code:BR_ATL_009', 'channel:Mobile_App', 'region:US_Southeast', 'calculated_risk_profile:', 'risk_profile_expiry_date:', 'investable_status:', 'suitability_check_passed:', 'suitability_mismatch_flag:', 'pre_trade_compliance_passed:', 'funds_availability_verified:', 'estimated_units:', 'front_end_load_amount:', 'net_investment_amount:', 'annual_management_fee_amount:', 'trade_reference_number:', 'contract_note_number:', 'transaction_status:', 'execution_timestamp:', 'settlement_date:', 'straight_through_processing_flag:', 'compliance_defect_flag:', 'compliance_review_notes:', 'supervisor_approval_timestamp:', 'total_processing_time_minutes:']
]

data_store.load_data(raw_data)


# ==================== API函数实现 ====================
def ValidateExistingRiskProfile(client_id: str, transaction_timestamp: str, 
                                previous_risk_profile: Optional[str] = None,
                                previous_risk_profile_expiry_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates whether the client has an existing, unexpired risk profile assessment on file.
    Determines if a new risk assessment questionnaire administration is required.
    """
    # Parse transaction timestamp
    transaction_dt = datetime.fromisoformat(transaction_timestamp.replace('Z', '+00:00'))
    
    # Check if profile exists and is not null
    if previous_risk_profile is None or previous_risk_profile == "None" or previous_risk_profile == "":
        requires_new_assessment = True
        profile_validity_status = "No_Previous_Profile"
        days_until_expiry = None
    else:
        # Check expiry date
        if previous_risk_profile_expiry_date and previous_risk_profile_expiry_date != "":
            expiry_dt = datetime.fromisoformat(previous_risk_profile_expiry_date.replace('Z', '+00:00'))
            
            # Compare dates
            if transaction_dt >= expiry_dt:
                requires_new_assessment = True
                profile_validity_status = "Expired"
                days_until_expiry = 0
            else:
                requires_new_assessment = False
                profile_validity_status = "Valid"
                days_until_expiry = (expiry_dt - transaction_dt).days
        else:
            requires_new_assessment = True
            profile_validity_status = "No_Expiry_Date"
            days_until_expiry = None
    
    result = {
        "client_id": client_id,
        "previous_risk_profile": previous_risk_profile if previous_risk_profile and previous_risk_profile != "None" else None,
        "previous_risk_profile_expiry_date": previous_risk_profile_expiry_date if previous_risk_profile_expiry_date else None,
        "transaction_timestamp": transaction_timestamp,
        "requires_new_assessment": requires_new_assessment,
        "profile_validity_status": profile_validity_status,
        "days_until_expiry": days_until_expiry
    }
    
    return result


def AdministerRiskQuestionnaire(client_id: str, client_age: int, client_annual_income: float,
                                client_investment_experience_years: int, client_liquidity_needs: str,
                                risk_questionnaire_responses: Dict[str, Any]) -> Dict[str, Any]:
    """
    Administers the regulatory-mandated Risk Tolerance Questionnaire to the client.
    Applies weighted scoring algorithm based on questionnaire responses.
    """
    total_score = 0
    
    # Parse risk_questionnaire_responses if it's a string
    if isinstance(risk_questionnaire_responses, str):
        risk_questionnaire_responses = json.loads(risk_questionnaire_responses)
    
    # Age bracket scoring (from q1_age_bracket or client_age)
    if client_age <= 30:
        age_score = 0
    elif client_age <= 40:
        age_score = 5
    elif client_age <= 50:
        age_score = 10
    elif client_age <= 60:
        age_score = 15
    else:
        age_score = 20
    
    total_score += age_score
    
    # Investment experience scoring
    if client_investment_experience_years == 0:
        experience_score = 0
    elif client_investment_experience_years <= 3:
        experience_score = 5
    elif client_investment_experience_years <= 7:
        experience_score = 10
    elif client_investment_experience_years <= 15:
        experience_score = 15
    else:
        experience_score = 20
    
    total_score += experience_score
    
    # Income stability scoring (from q2_income_stability)
    income_stability = risk_questionnaire_responses.get("q2_income_stability", "")
    if "Stable" in income_stability:
        income_score = 15
    elif "Retired" in income_stability:
        income_score = 10
    else:
        income_score = 5
    
    total_score += income_score
    
    # Investment horizon scoring (from q3_investment_horizon)
    horizon = risk_questionnaire_responses.get("q3_investment_horizon", "")
    if "10+" in horizon or "10+_years" in horizon:
        horizon_score = 20
    elif "5-10" in horizon or "5-10_years" in horizon:
        horizon_score = 15
    elif "3-5" in horizon or "3-5_years" in horizon:
        horizon_score = 10
    elif "1-3" in horizon or "1-3_years" in horizon:
        horizon_score = 5
    else:
        horizon_score = 0
    
    total_score += horizon_score
    
    # Loss tolerance scoring (from q4_loss_tolerance)
    loss_tolerance = risk_questionnaire_responses.get("q4_loss_tolerance", "")
    if "High" in loss_tolerance:
        loss_score = 20
    elif "Moderate" in loss_tolerance:
        loss_score = 10
    elif "Low" in loss_tolerance:
        loss_score = 5
    else:  # Very_Low
        loss_score = 0
    
    total_score += loss_score
    
    # Previous investments scoring (from q5_previous_investments)
    previous_investments = risk_questionnaire_responses.get("q5_previous_investments", "")
    investment_types = previous_investments.split("_")
    
    if len(investment_types) >= 4 or "Derivatives" in previous_investments or "Options" in previous_investments or "Forex" in previous_investments:
        investment_score = 15
    elif len(investment_types) >= 3 or "Stocks" in previous_investments:
        investment_score = 10
    elif "Bonds" in previous_investments or "Mutual" in previous_investments:
        investment_score = 5
    else:
        investment_score = 0
    
    total_score += investment_score
    
    # Liquidity needs adjustment (inverse scoring)
    if client_liquidity_needs == "Low":
        liquidity_score = 10
    elif client_liquidity_needs == "Medium":
        liquidity_score = 5
    else:  # High
        liquidity_score = 0
    
    total_score += liquidity_score
    
    result = {
        "client_id": client_id,
        "risk_questionnaire_responses": risk_questionnaire_responses,
        "risk_questionnaire_total_score": total_score,
        "questionnaire_completion_status": "Completed"
    }
    
    # Update data store
    entry = data_store.get_by_client_id(client_id)
    if entry:
        data_store.update_entry(entry['transaction_id'], {
            'risk_questionnaire_total_score': str(total_score)
        })
    
    return result


def CalculateRiskProfile(risk_questionnaire_total_score: int) -> Dict[str, Any]:
    """
    Applies deterministic mapping algorithm to convert numerical risk_questionnaire_total_score 
    into categorical risk profile classification.
    """
    # Deterministic mapping: 0-35 = Conservative, 36-65 = Balanced, 66-100 = Aggressive
    if risk_questionnaire_total_score <= 35:
        calculated_risk_profile = "Conservative"
    elif risk_questionnaire_total_score <= 65:
        calculated_risk_profile = "Balanced"
    else:
        calculated_risk_profile = "Aggressive"
    
    result = {
        "risk_questionnaire_total_score": risk_questionnaire_total_score,
        "calculated_risk_profile": calculated_risk_profile,
        "score_range_applied": f"{0 if calculated_risk_profile == 'Conservative' else (36 if calculated_risk_profile == 'Balanced' else 66)}-{35 if calculated_risk_profile == 'Conservative' else (65 if calculated_risk_profile == 'Balanced' else 100)}"
    }
    
    return result


def AuthorizeAndActivateRiskProfile(client_id: str, calculated_risk_profile: str,
                                   client_authorization_method: str, transaction_timestamp: str) -> Dict[str, Any]:
    """
    Presents the calculated risk profile to the client for review and explicit consent.
    Computes risk profile expiry date by adding 730 days to current timestamp.
    Updates client master record to set investable_status to 'Active'.
    """
    # Parse transaction timestamp
    transaction_dt = datetime.fromisoformat(transaction_timestamp.replace('Z', '+00:00'))
    
    # Calculate expiry date (730 days from now)
    expiry_dt = transaction_dt + timedelta(days=730)
    risk_profile_expiry_date = expiry_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Set investable status
    investable_status = "Active"
    
    # Authorization captured flag
    authorization_captured = True
    
    result = {
        "client_id": client_id,
        "calculated_risk_profile": calculated_risk_profile,
        "client_authorization_method": client_authorization_method,
        "authorization_captured": authorization_captured,
        "risk_profile_expiry_date": risk_profile_expiry_date,
        "investable_status": investable_status,
        "profile_activation_timestamp": transaction_timestamp
    }
    
    # Update data store
    entry = data_store.get_by_client_id(client_id)
    if entry:
        data_store.update_entry(entry['transaction_id'], {
            'calculated_risk_profile': calculated_risk_profile,
            'risk_profile_expiry_date': risk_profile_expiry_date,
            'investable_status': investable_status
        })
    
    return result


def RecommendSuitableProducts(client_id: str, calculated_risk_profile: str, 
                              investable_status: str) -> Dict[str, Any]:
    """
    Queries product master database applying hierarchical suitability filters based on 
    client calculated_risk_profile to return only suitable investment products.
    """
    # Check investable status
    if investable_status != "Active":
        return {
            "client_id": client_id,
            "calculated_risk_profile": calculated_risk_profile,
            "investable_status": investable_status,
            "recommendation_status": "Blocked_Inactive_Status",
            "suitable_products": []
        }
    
    # Define product suitability matrix
    # Conservative: Conservative, Moderate
    # Balanced: Conservative, Moderate, Balanced
    # Aggressive: All (Conservative, Moderate, Balanced, Growth, Aggressive)
    
    suitable_risk_ratings = []
    if calculated_risk_profile == "Conservative":
        suitable_risk_ratings = ["Conservative", "Moderate"]
    elif calculated_risk_profile == "Balanced":
        suitable_risk_ratings = ["Conservative", "Moderate", "Balanced"]
    elif calculated_risk_profile == "Aggressive":
        suitable_risk_ratings = ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"]
    
    # Simulate product recommendations (in real system, would query product database)
    suitable_products = []
    if "Conservative" in suitable_risk_ratings:
        suitable_products.append({
            "product_code": "MF_BOND_055",
            "product_name": "Conservative Income Fund",
            "product_risk_rating": "Conservative"
        })
    if "Moderate" in suitable_risk_ratings:
        suitable_products.append({
            "product_code": "MF_MOD_033",
            "product_name": "Moderate Allocation Fund",
            "product_risk_rating": "Moderate"
        })
    if "Balanced" in suitable_risk_ratings:
        suitable_products.append({
            "product_code": "MF_BAL_045",
            "product_name": "Balanced Growth and Income Fund",
            "product_risk_rating": "Balanced"
        })
    if "Growth" in suitable_risk_ratings:
        suitable_products.append({
            "product_code": "MF_TECH_001",
            "product_name": "Global Tech Growth Fund",
            "product_risk_rating": "Growth"
        })
    if "Aggressive" in suitable_risk_ratings:
        suitable_products.append({
            "product_code": "MF_AGGR_122",
            "product_name": "Emerging Markets Equity Fund",
            "product_risk_rating": "Aggressive"
        })
    
    result = {
        "client_id": client_id,
        "calculated_risk_profile": calculated_risk_profile,
        "investable_status": investable_status,
        "recommendation_status": "Products_Available",
        "suitable_products": suitable_products,
        "number_of_suitable_products": len(suitable_products)
    }
    
    return result


def CreateInvestmentOrder(client_id: str, relationship_manager_id: str, product_code: str,
                         subscription_amount: float, settlement_account_number: str,
                         transaction_timestamp: str, branch_code: str, channel: str,
                         region: str) -> Dict[str, Any]:
    """
    Creates comprehensive new transaction record with system-generated transaction_id.
    Captures all order parameters and retrieves product master data.
    """
    # Get existing entry for this client
    entry = data_store.get_by_client_id(client_id)
    
    if not entry:
        # Generate new transaction ID
        transaction_id = f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        transaction_id = entry.get('transaction_id', f"INV_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    
    # Retrieve product data from entry if exists
    product_name = entry.get('product_name', '') if entry else ''
    product_risk_rating = entry.get('product_risk_rating', '') if entry else ''
    product_category = entry.get('product_category', '') if entry else ''
    current_nav = float(entry.get('current_nav', 0)) if entry and entry.get('current_nav') else 0.0
    front_end_load_percentage = float(entry.get('front_end_load_percentage', 0)) if entry and entry.get('front_end_load_percentage') else 0.0
    annual_management_fee_percentage = float(entry.get('annual_management_fee_percentage', 0)) if entry and entry.get('annual_management_fee_percentage') else 0.0
    compliance_officer_id = entry.get('compliance_officer_id', '') if entry else ''
    
    result = {
        "transaction_id": transaction_id,
        "client_id": client_id,
        "relationship_manager_id": relationship_manager_id,
        "product_code": product_code,
        "product_name": product_name,
        "product_risk_rating": product_risk_rating,
        "product_category": product_category,
        "subscription_amount": subscription_amount,
        "settlement_account_number": settlement_account_number,
        "transaction_timestamp": transaction_timestamp,
        "branch_code": branch_code,
        "channel": channel,
        "region": region,
        "current_nav": current_nav,
        "front_end_load_percentage": front_end_load_percentage,
        "annual_management_fee_percentage": annual_management_fee_percentage,
        "compliance_officer_id": compliance_officer_id,
        "order_creation_status": "Order_Created"
    }
    
    return result


def ValidateProductSuitability(transaction_id: str, calculated_risk_profile: str,
                               product_risk_rating: str) -> Dict[str, Any]:
    """
    Executes automated real-time suitability check by comparing product_risk_rating 
    against client calculated_risk_profile using hierarchical matching matrix.
    """
    # Hierarchical suitability matrix
    suitability_matrix = {
        "Conservative": ["Conservative", "Moderate"],
        "Balanced": ["Conservative", "Moderate", "Balanced"],
        "Aggressive": ["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"]
    }
    
    suitable_ratings = suitability_matrix.get(calculated_risk_profile, [])
    
    # Check suitability
    if product_risk_rating in suitable_ratings:
        suitability_check_passed = True
        suitability_mismatch_flag = False
        suitability_assessment = "Suitable"
    else:
        suitability_check_passed = False
        suitability_mismatch_flag = True
        suitability_assessment = "Unsuitable_Requires_Waiver"
    
    result = {
        "transaction_id": transaction_id,
        "calculated_risk_profile": calculated_risk_profile,
        "product_risk_rating": product_risk_rating,
        "suitability_check_passed": suitability_check_passed,
        "suitability_mismatch_flag": suitability_mismatch_flag,
        "suitability_assessment": suitability_assessment
    }
    
    # Update data store
    data_store.update_entry(transaction_id, {
        'suitability_check_passed': str(suitability_check_passed),
        'suitability_mismatch_flag': str(suitability_mismatch_flag)
    })
    
    return result


def ExecuteAMLSanctionsScreening(client_id: str, client_name: str, 
                                 transaction_id: str) -> Dict[str, Any]:
    """
    Interfaces with enterprise AML monitoring platform to perform real-time sanctions 
    list screening of client_id and client_name against all relevant sanctions databases.
    """
    # Get entry data
    entry = data_store.get_by_transaction_id(transaction_id)
    
    # Retrieve existing AML status if available
    if entry:
        aml_status = entry.get('aml_sanctions_screening_status', 'Clear')
    else:
        aml_status = 'Clear'
    
    # Parse status
    if aml_status == "Pending":
        screening_status = "Pending"
        screening_result = "Screening_In_Progress"
    elif aml_status == "Flagged":
        screening_status = "Flagged"
        screening_result = "Potential_Match_Manual_Review_Required"
    else:
        screening_status = "Clear"
        screening_result = "No_Matches_Found"
    
    result = {
        "transaction_id": transaction_id,
        "client_id": client_id,
        "client_name": client_name,
        "aml_sanctions_screening_status": screening_status,
        "screening_result": screening_result,
        "screening_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    
    return result


def ValidateThresholdAndSOW(subscription_amount: float, monitoring_threshold_amount: float,
                            source_of_wealth_documented: bool) -> Dict[str, Any]:
    """
    Compares subscription_amount against configured monitoring_threshold_amount to determine 
    if transaction exceeds regulatory large-transaction threshold requiring enhanced due diligence.
    """
    # Check if threshold exceeded (inclusive rule: >= is treated as exceeding)
    transaction_threshold_exceeded = subscription_amount >= monitoring_threshold_amount
    
    # Validate SOW documentation
    if transaction_threshold_exceeded:
        if source_of_wealth_documented:
            threshold_validation_status = "Passed"
            sow_validation_result = "SOW_Documentation_Valid"
        else:
            threshold_validation_status = "Failed"
            sow_validation_result = "SOW_Documentation_Missing"
    else:
        threshold_validation_status = "Passed"
        sow_validation_result = "Below_Threshold_SOW_Not_Required"
    
    result = {
        "subscription_amount": subscription_amount,
        "monitoring_threshold_amount": monitoring_threshold_amount,
        "transaction_threshold_exceeded": transaction_threshold_exceeded,
        "source_of_wealth_documented": source_of_wealth_documented,
        "threshold_validation_status": threshold_validation_status,
        "sow_validation_result": sow_validation_result
    }
    
    return result


def VerifyFundsAvailability(settlement_account_number: str, 
                           subscription_amount: float) -> Dict[str, Any]:
    """
    Queries Core Banking System to retrieve real-time available_balance for specified 
    settlement_account_number and compares against subscription_amount.
    """
    # Get available balance from data store
    available_balance = 0.0
    for entry in data_store.data:
        if entry.get('settlement_account_number') == settlement_account_number:
            available_balance = float(entry.get('available_balance', 0))
            break
    
    # Compare balance (inclusive rule: equal amounts pass)
    funds_availability_verified = available_balance >= subscription_amount
    
    if funds_availability_verified:
        verification_result = "Sufficient_Funds_Available"
    else:
        verification_result = "Insufficient_Funds"
    
    result = {
        "settlement_account_number": settlement_account_number,
        "subscription_amount": subscription_amount,
        "available_balance": available_balance,
        "funds_availability_verified": funds_availability_verified,
        "verification_result": verification_result
    }
    
    return result


def AggregateComplianceChecks(suitability_check_passed: bool, aml_sanctions_screening_status: str,
                              transaction_threshold_exceeded: bool, source_of_wealth_documented: bool,
                              funds_availability_verified: bool) -> Dict[str, Any]:
    """
    Aggregates validation results from all parallel pre-trade compliance checks.
    Applies boolean AND logic across all validation outcomes.
    """
    # Check AML status
    aml_passed = aml_sanctions_screening_status == "Clear"
    
    # Check threshold/SOW
    if transaction_threshold_exceeded:
        threshold_sow_passed = source_of_wealth_documented
    else:
        threshold_sow_passed = True
    
    # Apply boolean AND logic
    pre_trade_compliance_passed = (
        suitability_check_passed and 
        aml_passed and 
        threshold_sow_passed and 
        funds_availability_verified
    )
    
    # Build compliance summary
    compliance_checks_summary = {
        "suitability_check": "Passed" if suitability_check_passed else "Failed",
        "aml_sanctions_screening": "Passed" if aml_passed else "Failed",
        "threshold_sow_validation": "Passed" if threshold_sow_passed else "Failed",
        "funds_availability": "Passed" if funds_availability_verified else "Failed"
    }
    
    if pre_trade_compliance_passed:
        overall_compliance_status = "All_Checks_Passed"
    else:
        overall_compliance_status = "One_Or_More_Checks_Failed"
    
    result = {
        "suitability_check_passed": suitability_check_passed,
        "aml_sanctions_screening_status": aml_sanctions_screening_status,
        "transaction_threshold_exceeded": transaction_threshold_exceeded,
        "source_of_wealth_documented": source_of_wealth_documented,
        "funds_availability_verified": funds_availability_verified,
        "pre_trade_compliance_passed": pre_trade_compliance_passed,
        "overall_compliance_status": overall_compliance_status,
        "compliance_checks_summary": compliance_checks_summary
    }
    
    return result


def CalculateTransactionFinancials(subscription_amount: float, front_end_load_percentage: float,
                                  annual_management_fee_percentage: float, current_nav: float,
                                  product_category: str, pre_trade_compliance_passed: bool) -> Dict[str, Any]:
    """
    Computes comprehensive transaction financial metrics with regulatory precision.
    Only executes when pre_trade_compliance_passed = TRUE.
    """
    if not pre_trade_compliance_passed:
        return {
            "subscription_amount": subscription_amount,
            "calculation_status": "Blocked_Compliance_Not_Passed",
            "front_end_load_amount": None,
            "net_investment_amount": None,
            "estimated_units": None,
            "annual_management_fee_amount": None
        }
    
    # Calculate front end load amount
    front_end_load_amount = round(subscription_amount * front_end_load_percentage / 100, 2)
    
    # Calculate net investment amount
    net_investment_amount = round(subscription_amount - front_end_load_amount, 2)
    
    # Calculate estimated units (only for Mutual_Fund category with NAV)
    if product_category == "Mutual_Fund" and current_nav > 0:
        estimated_units = round(net_investment_amount / current_nav, 4)
    else:
        estimated_units = None
    
    # Calculate annual management fee amount
    annual_management_fee_amount = round(net_investment_amount * annual_management_fee_percentage / 100, 2)
    
    result = {
        "subscription_amount": subscription_amount,
        "front_end_load_percentage": front_end_load_percentage,
        "front_end_load_amount": front_end_load_amount,
        "net_investment_amount": net_investment_amount,
        "current_nav": current_nav,
        "estimated_units": estimated_units,
        "annual_management_fee_percentage": annual_management_fee_percentage,
        "annual_management_fee_amount": annual_management_fee_amount,
        "calculation_status": "Completed"
    }
    
    return result


def ManageCoolingOffPeriod(transaction_id: str, product_category: str, 
                          cooling_off_period_required: bool, cooling_off_period_hours: int,
                          client_authorization_timestamp: str) -> Dict[str, Any]:
    """
    Evaluates product_category against cooling-off period requirements to determine applicability.
    When applicable, calculates cooling-off expiry timestamp.
    """
    # Categories requiring cooling-off
    cooling_off_categories = ["Structured_Note", "Dual_Currency_Investment", "Principal_Protected_Note"]
    
    # Check if cooling-off applies
    cooling_off_applicable = product_category in cooling_off_categories and cooling_off_period_required
    
    if cooling_off_applicable:
        # Parse authorization timestamp
        auth_dt = datetime.fromisoformat(client_authorization_timestamp.replace('Z', '+00:00'))
        
        # Calculate expiry timestamp
        expiry_dt = auth_dt + timedelta(hours=cooling_off_period_hours)
        cooling_off_expiry_timestamp = expiry_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Check if cooling-off period has elapsed
        current_dt = datetime.now()
        if current_dt >= expiry_dt:
            transaction_status = "Ready_for_Execution"
            cooling_off_status = "Cooling_Off_Period_Elapsed"
        else:
            transaction_status = "Pending_Cooling_Off"
            cooling_off_status = "Cooling_Off_Period_Active"
    else:
        transaction_status = "Ready_for_Execution"
        cooling_off_status = "Cooling_Off_Not_Required"
        cooling_off_expiry_timestamp = None
    
    result = {
        "transaction_id": transaction_id,
        "product_category": product_category,
        "cooling_off_period_required": cooling_off_applicable,
        "cooling_off_period_hours": cooling_off_period_hours if cooling_off_applicable else 0,
        "client_authorization_timestamp": client_authorization_timestamp,
        "cooling_off_expiry_timestamp": cooling_off_expiry_timestamp,
        "transaction_status": transaction_status,
        "cooling_off_status": cooling_off_status
    }
    
    # Update data store
    data_store.update_entry(transaction_id, {
        'transaction_status': transaction_status
    })
    
    return result


def ExecuteInvestmentTransaction(transaction_id: str, product_category: str, product_code: str,
                                 net_investment_amount: float, settlement_account_number: str,
                                 client_id: str, transaction_status: str, 
                                 subscription_amount: float) -> Dict[str, Any]:
    """
    Final execution step consummating investment transaction through appropriate channel routing.
    Only executes when transaction_status = 'Ready_for_Execution'.
    """
    if transaction_status != "Ready_for_Execution":
        return {
            "transaction_id": transaction_id,
            "execution_status": "Blocked_Not_Ready_For_Execution",
            "transaction_status": transaction_status,
            "execution_timestamp": None
        }
    
    # Generate execution timestamp
    execution_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    execution_dt = datetime.now()
    
    # Route based on product category
    if product_category == "Mutual_Fund":
        # Route to fund house gateway
        trade_reference_number = f"TRF_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        contract_note_number = None
        execution_channel = "Fund_House_Gateway"
        settlement_cycle = "T+1"
        settlement_date = (execution_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Route to Treasury/Global Markets desk
        trade_reference_number = None
        contract_note_number = f"CN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        execution_channel = "Treasury_Desk"
        settlement_cycle = "T+2"
        settlement_date = (execution_dt + timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Update transaction status
    new_transaction_status = "Executed"
    
    result = {
        "transaction_id": transaction_id,
        "product_category": product_category,
        "product_code": product_code,
        "client_id": client_id,
        "net_investment_amount": net_investment_amount,
        "subscription_amount": subscription_amount,
        "settlement_account_number": settlement_account_number,
        "execution_timestamp": execution_timestamp,
        "execution_channel": execution_channel,
        "trade_reference_number": trade_reference_number,
        "contract_note_number": contract_note_number,
        "settlement_cycle": settlement_cycle,
        "settlement_date": settlement_date,
        "transaction_status": new_transaction_status,
        "execution_status": "Successfully_Executed"
    }
    
    # Update data store
    data_store.update_entry(transaction_id, {
        'transaction_status': new_transaction_status,
        'execution_timestamp': execution_timestamp,
        'settlement_date': settlement_date,
        'trade_reference_number': trade_reference_number if trade_reference_number else '',
        'contract_note_number': contract_note_number if contract_note_number else ''
    })
    
    return result


def CalculateSTPFlag(transaction_id: str, suitability_mismatch_flag: bool,
                    supervisor_approval_required: bool, aml_sanctions_screening_status: str) -> Dict[str, Any]:
    """
    Evaluates entire transaction lifecycle to determine if straight-through processing was achieved.
    Sets straight_through_processing_flag to TRUE only for zero-touch transactions.
    """
    # Check for any manual intervention indicators
    has_suitability_mismatch = suitability_mismatch_flag
    has_supervisor_approval = supervisor_approval_required
    has_aml_flag = aml_sanctions_screening_status != "Clear"
    
    # STP achieved only if no manual interventions
    straight_through_processing_flag = not (has_suitability_mismatch or has_supervisor_approval or has_aml_flag)
    
    if straight_through_processing_flag:
        stp_assessment = "Zero_Touch_STP_Achieved"
    else:
        stp_assessment = "Manual_Intervention_Required"
    
    # Build intervention summary
    manual_interventions = []
    if has_suitability_mismatch:
        manual_interventions.append("Suitability_Mismatch_Waiver")
    if has_supervisor_approval:
        manual_interventions.append("Supervisor_Approval")
    if has_aml_flag:
        manual_interventions.append("AML_Screening_Flag")
    
    result = {
        "transaction_id": transaction_id,
        "suitability_mismatch_flag": suitability_mismatch_flag,
        "supervisor_approval_required": supervisor_approval_required,
        "aml_sanctions_screening_status": aml_sanctions_screening_status,
        "straight_through_processing_flag": straight_through_processing_flag,
        "stp_assessment": stp_assessment,
        "manual_interventions": manual_interventions if manual_interventions else None
    }
    
    # Update data store
    data_store.update_entry(transaction_id, {
        'straight_through_processing_flag': str(straight_through_processing_flag)
    })
    
    return result


def ExecutePostTradeComplianceReview(transaction_id: str, audio_video_recording_id: Optional[str],
                                    suitability_mismatch_flag: bool, transaction_threshold_exceeded: bool,
                                    disclosure_documents_presented: List[str]) -> Dict[str, Any]:
    """
    Performs post-execution quality assurance and regulatory compliance verification.
    Only executes after transaction_status = 'Executed'.
    """
    compliance_defects = []
    
    # Check audio/video recording for complex products
    entry = data_store.get_by_transaction_id(transaction_id)
    if entry:
        recording_required = entry.get('audio_video_recording_required', 'False') == 'True'
        if recording_required:
            if not audio_video_recording_id or audio_video_recording_id == "":
                compliance_defects.append("Missing_Audio_Video_Recording")
    
    # Check suitability mismatch documentation
    if suitability_mismatch_flag:
        if entry:
            waiver_doc = entry.get('mismatch_waiver_document_id', '')
            if not waiver_doc or waiver_doc == "":
                compliance_defects.append("Missing_Suitability_Waiver_Documentation")
    
    # Check threshold SOW documentation
    if transaction_threshold_exceeded:
        if entry:
            sow_doc = entry.get('source_of_wealth_documented', 'False')
            if sow_doc != 'True':
                compliance_defects.append("Missing_SOW_Documentation")
    
    # Check disclosure documents completeness
    if entry:
        product_category = entry.get('product_category', '')
        required_docs = []
        
        if product_category == "Mutual_Fund":
            required_docs = ["Fund_Fact_Sheet", "Prospectus", "KID"]
        elif product_category in ["Structured_Note", "Dual_Currency_Investment", "Principal_Protected_Note"]:
            required_docs = ["Risk_Disclosure_Statement", "KID"]
        
        # Parse disclosure_documents_presented if string
        if isinstance(disclosure_documents_presented, str):
            try:
                disclosure_documents_presented = json.loads(disclosure_documents_presented)
            except:
                disclosure_documents_presented = []
        
        for req_doc in required_docs:
            doc_found = False
            for presented_doc in disclosure_documents_presented:
                if req_doc in presented_doc or presented_doc in req_doc:
                    doc_found = True
                    break
            if not doc_found:
                compliance_defects.append(f"Missing_{req_doc}")
    
    # Set compliance defect flag
    compliance_defect_flag = len(compliance_defects) > 0
    
    if compliance_defect_flag:
        review_conclusion = "Deficiencies_Identified"
    else:
        review_conclusion = "No_Deficiencies_Found"
    
    compliance_review_notes = "; ".join(compliance_defects) if compliance_defects else "All compliance requirements met"
    
    result = {
        "transaction_id": transaction_id,
        "compliance_defect_flag": compliance_defect_flag,
        "compliance_defects_identified": compliance_defects if compliance_defects else None,
        "review_conclusion": review_conclusion,
        "compliance_review_notes": compliance_review_notes,
        "review_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    
    # Update data store
    data_store.update_entry(transaction_id, {
        'compliance_defect_flag': str(compliance_defect_flag),
        'compliance_review_notes': compliance_review_notes
    })
    
    return result


def CalculateProcessingMetrics(transaction_timestamp: str, execution_timestamp: str) -> Dict[str, Any]:
    """
    Computes end-to-end transaction processing efficiency metrics by calculating 
    total_processing_time_minutes as elapsed time between transaction initiation and completion.
    """
    # Parse timestamps
    transaction_dt = datetime.fromisoformat(transaction_timestamp.replace('Z', '+00:00'))
    execution_dt = datetime.fromisoformat(execution_timestamp.replace('Z', '+00:00'))
    
    # Calculate elapsed time in minutes
    elapsed_time = execution_dt - transaction_dt
    total_processing_time_minutes = round(elapsed_time.total_seconds() / 60, 2)
    
    # Calculate processing efficiency metrics
    if total_processing_time_minutes <= 30:
        processing_efficiency_rating = "Excellent"
    elif total_processing_time_minutes <= 60:
        processing_efficiency_rating = "Good"
    elif total_processing_time_minutes <= 120:
        processing_efficiency_rating = "Average"
    else:
        processing_efficiency_rating = "Below_Target"
    
    result = {
        "transaction_timestamp": transaction_timestamp,
        "execution_timestamp": execution_timestamp,
        "total_processing_time_minutes": total_processing_time_minutes,
        "processing_efficiency_rating": processing_efficiency_rating
    }
    
    return result

# ==================== 无效API函数 ====================

def DocumentImageResolutionValidation(application_id: str, submitted_documents: List[str]) -> Dict[str, Any]:
    """Document image resolution validation API"""
    return {"application_id": application_id, "resolution_check_passed": len(submitted_documents) > 0}

def MaterialIntegrityScoreCalculation(application_id: str, submitted_documents: List[str], 
                                      image_clarity_scores: Dict[str, float], 
                                      text_recognition_accuracy: Dict[str, float],
                                      loan_type: str, customer_occupation_type: str) -> Dict[str, Any]:
    """Material integrity score calculation API"""
    return {"application_id": application_id, "integrity_score": len(submitted_documents) * 10.0}

def IdentityAuthenticationAndFaceMatching(application_id: str, customer_id: str,
                                         id_card_number: str, id_card_name: str,
                                         id_card_images: Dict[str, str], 
                                         face_photo: str) -> Dict[str, Any]:
    """Identity authentication and face matching API"""
    return {"application_id": application_id, "authentication_passed": len(id_card_number) == 18}

def FraudPreventionValidation(application_id: str, customer_id: str, application_source: str,
                              identity_authentication_result: bool, face_matching_passed: bool,
                              device_id: Optional[str] = None, ip_location: Optional[str] = None,
                              gps_location: Optional[str] = None) -> Dict[str, Any]:
    """Fraud prevention validation API"""
    return {"application_id": application_id, "fraud_check_passed": identity_authentication_result and face_matching_passed}

def ApplicationDataConsistencyCheck(application_id: str, material_integrity_score: float,
                                   requested_amount: float, requested_term: int,
                                   loan_type: str, customer_age_years: int, 
                                   customer_occupation_type: str, customer_monthly_salary: float,
                                   purpose_receipt_amount: Optional[float] = None) -> Dict[str, Any]:
    """Application data consistency check API"""
    return {"application_id": application_id, "consistency_passed": material_integrity_score > 60.0}

def CreditReportRetrievalAndAnalysis(application_id: str, customer_id: str,
                                    identity_authentication_result: bool, fraud_prevention_passed: bool,
                                    credit_authorization_confirmed: bool,
                                    customer_id_number: str, customer_full_name: str) -> Dict[str, Any]:
    """Credit report retrieval and analysis API"""
    return {"application_id": application_id, "credit_report_retrieved": identity_authentication_result}

def InternalExternalDataCrossValidation(application_id: str, customer_id: str,
                                       customer_occupation_type: str, customer_monthly_salary: float,
                                       customer_current_debt: float, internal_account_balance: float,
                                       internal_history_months: int,
                                       total_loan_outstanding: float,
                                       total_creditcard_balance: float) -> Dict[str, Any]:
    """Internal and external data cross-validation API"""
    return {"application_id": application_id, "validation_passed": customer_monthly_salary > customer_current_debt}

def DebtIncomeRatioPreciseCalculation(application_id: str, verified_monthly_salary: float,
                                     verified_current_debt: float, total_loan_outstanding: float,
                                     total_creditcard_balance: float, requested_amount: float,
                                     requested_term: int, 
                                     estimated_rate: Optional[float] = None) -> Dict[str, Any]:
    """Debt-to-income ratio precise calculation API"""
    total_debt = verified_current_debt + total_loan_outstanding + total_creditcard_balance
    return {"application_id": application_id, "debt_income_ratio": total_debt / verified_monthly_salary if verified_monthly_salary > 0 else 999.0}

def CreditScoreAndRiskGradeAssessment(application_id: str, customer_age_years: int,
                                     customer_occupation_type: str, credit_history_length: int,
                                     past_overdue_count: int, active_credit_accounts: int,
                                     credit_query_count_6m: int, debt_income_ratio: float,
                                     internal_account_balance: float, 
                                     internal_history_months: int,
                                     has_negative_records: bool, overdue_status: str,
                                     policy_version: str) -> Dict[str, Any]:
    """Credit score and risk grade assessment API"""
    base_score = 700 - past_overdue_count * 20
    return {"application_id": application_id, "credit_score": max(300, base_score), "risk_grade": "LOW"}

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
    return {"application_id": application_id, "verification_recorded": verification_result}

def ReviewReportDocumentGeneration(application_id: str, assigned_reviewer_id: str,
                                  material_integrity_score: float, identity_authentication_result: bool,
                                  fraud_prevention_passed: bool, credit_score_value: int, risk_grade: str,
                                  debt_income_ratio: float, salary_verification_result: bool,
                                  verification_result: bool, reviewer_suggestion: str,
                                  suggested_amount: Optional[float] = None,
                                  suggestion_rationale: Optional[str] = None) -> Dict[str, Any]:
    """Review report document generation API"""
    return {"application_id": application_id, "report_generated": True, "report_id": f"RPT{application_id}"}

def ApprovalWorkflowRouting(application_id: str, requested_amount: float, credit_score_value: int,
                           risk_grade: str, debt_income_ratio: float, past_overdue_count: int,
                           fraud_prevention_passed: bool, reviewer_suggestion: str) -> Dict[str, Any]:
    """Approval workflow routing API"""
    queue = "auto_approval" if credit_score_value > 650 else "manual_approval"
    return {"application_id": application_id, "routing_queue": queue}

def RiskOrientedPricingCalculation(application_id: str, credit_score_value: int, risk_grade: str,
                                  internal_account_balance: float, internal_history_months: int,
                                  current_benchmark_rate: float, requested_term: int) -> Dict[str, Any]:
    """Risk-oriented pricing calculation API"""
    rate_adjustment = 0.02 if credit_score_value < 600 else 0.0
    return {"application_id": application_id, "suggested_rate": current_benchmark_rate + rate_adjustment}

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
    return {"application_id": application_id, "decision": decision, "approved_amount": approved_amount or requested_amount}

def MonthlyPaymentAmountCalculation(application_id: str, approved_amount: float,
                                   approved_rate: float, approved_term: int,
                                   payment_method: str) -> Dict[str, Any]:
    """Monthly payment amount calculation API"""
    monthly_payment = approved_amount / approved_term
    return {"application_id": application_id, "monthly_payment": monthly_payment}

def DigitalContractCreation(application_id: str, customer_id: str, approved_amount: float,
                           approved_rate: float, approved_term: int,
                           payment_method: str, monthly_payment: float,
                           payment_schedule: List[Dict], loan_type: str,
                           customer_full_name: str, customer_id_number: str,
                           customer_residence: str, customer_phone: str) -> Dict[str, Any]:
    """Digital contract creation API"""
    return {"application_id": application_id, "contract_id": f"CTR{application_id}", "contract_generated": True}

def ContractSignatureValidation(application_id: str, contract_id: str,
                                signing_timestamp: str, signing_method: str,
                                contract_hash: str, digital_certificate: Optional[str] = None,
                                customer_signature: Optional[str] = None,
                                video_recording_id: Optional[str] = None) -> Dict[str, Any]:
    """Contract signature validation API"""
    return {"application_id": application_id, "signature_valid": len(contract_hash) > 0}

def PreDisbursementFinalCheck(application_id: str, customer_id: str, contract_id: str,
                             signature_verified: bool, approved_amount: float,
                             target_account: str, disbursement_type: Optional[str] = None,
                             payee_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Pre-disbursement final check API"""
    return {"application_id": application_id, "final_check_passed": signature_verified}

def FundTransferExecution(application_id: str, contract_id: str, compliance_verified: bool,
                         approved_amount: float, target_account: str,
                         transfer_timestamp: str, transfer_method: str,
                         operator_user_id: str) -> Dict[str, Any]:
    """Fund transfer execution API"""
    return {"application_id": application_id, "transfer_id": f"TXN{application_id}", "transfer_successful": compliance_verified}

def ApplicationStatusUpdateAndAlert(application_id: str, submission_timestamp: str,
                                   decision: str, decision_timestamp: str,
                                   review_begin_timestamp: str,
                                   transfer_timestamp: Optional[str] = None,
                                   transfer_status: Optional[str] = None) -> Dict[str, Any]:
    """Application status update and alert API"""
    return {"application_id": application_id, "status_updated": True, "notification_sent": True}

def adopt_a_pet(location: str, animal_type: str) -> dict:
    return {"pets": [{"name": f"{animal_type}_1", "location": location, "available": True}]}

def get_axolotl_info(fact_id: int) -> dict:
    return {"fact_id": fact_id, "fact": "Axolotls can regenerate limbs", "image_url": f"https://axolotl.com/{fact_id}.jpg"}

def get_cat_fact() -> dict:
    return {"fact": "Cats sleep 70% of their lives"}

def get_cat_image(tag: Optional[str] = None) -> dict:
    return {"url": f"https://cataas.com/cat/{tag or 'random'}", "tags": [tag or "random"]}

def search_anime(query: str) -> dict:
    return {"results": [{"title": query, "episodes": 12, "score": 8.5}]}

def get_anime_info(anime_id: int) -> dict:
    return {"id": anime_id, "title": f"Anime_{anime_id}", "episodes": 24}

def get_anime_list(status: str) -> dict:
    return {"anime_list": [{"id": 1, "title": "Anime 1", "status": status}]}

def get_anime_quote() -> dict:
    return {"quote": "Believe in yourself!", "character": "Naruto", "anime": "Naruto"}

def check_ip_reputation(ip_address: str) -> dict:
    return {"ip": ip_address, "abuse_confidence_score": 15, "is_malicious": False}

def check_threat_intelligence(indicator: str) -> dict:
    return {"indicator": indicator, "threat_level": "low", "pulses": 3}

def analyze_malware(file_hash: str) -> dict:
    return {"hash": file_hash, "malware_detected": False, "analysis_complete": True}

def check_safe_browsing(url: str) -> dict:
    return {"url": url, "is_safe": True, "threats": []}

def generate_discord_image(user_id: str, effect: str) -> dict:
    return {"image_url": f"https://api.amethyste.com/{effect}/{user_id}.png"}

def get_artwork(artwork_id: int) -> dict:
    return {"id": artwork_id, "title": f"Artwork_{artwork_id}", "artist": "Unknown"}

def generate_color_scheme() -> dict:
    return {"colors": ["#4A90E2", "#50E3C2", "#F5A623", "#D0021B"]}

def get_color_palette(palette_id: int) -> dict:
    return {"id": palette_id, "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"]}

def authenticate_user(username: str, password: str) -> dict:
    return {"user_id": f"user_{username}", "token": f"token_{username}", "authenticated": True}

def generate_otp(phone_number: str) -> dict:
    return {"phone": phone_number, "otp": "123456", "expires_in": 300}

def create_user(username: str, email: str) -> dict:
    return {"user_id": f"usr_{username}", "username": username, "email": email}

def passwordless_login(email: str) -> dict:
    return {"email": email, "magic_link": f"https://auth.com/login/{email}", "sent": True}

def query_blockchain(chain: str, query: str) -> dict:
    return {"chain": chain, "data": {"transactions": 100, "query": query}}

def get_chainlink_data(feed_id: str) -> dict:
    return {"feed_id": feed_id, "price": 1850.50, "timestamp": 1234567890}

def anchor_to_blockchain(data: str) -> dict:
    return {"data_hash": f"hash_{data}", "anchored": True, "block_height": 750000}

def get_blockchain_data(chain: str, address: str) -> dict:
    return {"chain": chain, "address": address, "balance": 1.5, "transactions": 42}

def get_bhagavad_gita_verse(chapter: int, verse: int) -> dict:
    return {"chapter": chapter, "verse": verse, "text": "Sample verse text", "translation": "Sample translation"}

def get_bhagavad_gita_telugu(chapter: int, verse: int) -> dict:
    return {"chapter": chapter, "verse": verse, "text_telugu": "తెలుగు వచనం"}

def get_bible_verse(reference: str) -> dict:
    return {"reference": reference, "text": "In the beginning...", "translation": "KJV"}

def get_persian_poetry(poet: str) -> dict:
    return {"poet": poet, "poem": "Sample Persian poetry", "recitation_url": "https://ganjoor.com/audio.mp3"}

def manage_dashboard(dashboard_id: int, action: str) -> dict:
    return {"dashboard_id": dashboard_id, "action": action, "status": "success"}

def search_charity(name: str) -> dict:
    return {"charities": [{"name": name, "ein": "12-3456789", "category": "Education"}]}

def get_company_logo(domain: str) -> dict:
    return {"domain": domain, "logo_url": f"https://logo.clearbit.com/{domain}"}

def search_domains(keyword: str) -> dict:
    return {"domains": [f"{keyword}.com", f"{keyword}.net", f"{keyword}.org"]}

def get_public_holidays(country: str, year: int) -> dict:
    return {"country": country, "year": year, "holidays": [{"name": "New Year", "date": f"{year}-01-01"}]}

def get_holidays_calendarific(country: str, year: int) -> dict:
    return {"country": country, "year": year, "holidays": [{"name": "Christmas", "date": f"{year}-12-25"}]}

def get_national_holiday(date: str) -> dict:
    return {"date": date, "holidays": [{"name": "Independence Day", "type": "National"}]}

def get_church_calendar(date: str) -> dict:
    return {"date": date, "event": "Easter Sunday", "liturgical_color": "white"}

def upload_file_anon(file_content: str) -> dict:
    return {"file_id": f"anon_{hash(file_content)}", "url": f"https://anonfiles.com/file_{hash(file_content)}"}

def upload_to_box(file_name: str, file_content: str) -> dict:
    return {"file_id": f"box_{file_name}", "shared_link": f"https://box.com/s/{file_name}"}

def upload_to_fileio(file_content: str) -> dict:
    return {"link": f"https://file.io/{hash(file_content)}", "expires": "14 days"}

def check_azure_health(resource_id: str) -> dict:
    return {"resource_id": resource_id, "status": "healthy", "issues": []}

def trigger_bitrise_build(app_id: str, workflow: str) -> dict:
    return {"app_id": app_id, "workflow": workflow, "build_number": 123, "status": "started"}

def trigger_buddy_pipeline(pipeline_id: int) -> dict:
    return {"pipeline_id": pipeline_id, "execution_id": f"exec_{pipeline_id}", "status": "running"}

def trigger_circleci_build(project: str, branch: str) -> dict:
    return {"project": project, "branch": branch, "build_num": 456, "status": "queued"}

def get_crypto_rate(crypto: str, fiat: str) -> dict:
    return {"crypto": crypto, "fiat": fiat, "rate": 45000.50}

def get_binance_price(symbol: str) -> dict:
    return {"symbol": symbol, "price": 1.2345, "volume": 1000000}

def get_bitcambio_assets() -> dict:
    return {"assets": ["BTC", "ETH", "USDT", "BNB"]}

def get_bitcoin_average(symbol: str) -> dict:
    return {"symbol": symbol, "last": 45123.45, "bid": 45120.00, "ask": 45125.00}

def convert_currency(from_currency: str, to_currency: str, amount: float) -> dict:
    return {"from": from_currency, "to": to_currency, "amount": amount, "result": amount * 1.2}

def get_exchange_rate(base: str, target: str) -> dict:
    return {"base": base, "target": target, "rate": 1.18}

def get_currency_rate(base: str, target: str) -> dict:
    return {"base": base, "target": target, "rate": 0.85, "date": "2024-01-01"}

def get_currency_freaks_rate(base: str) -> dict:
    return {"base": base, "rates": {"USD": 1.0, "EUR": 0.85, "GBP": 0.73}}

def postman_echo_test(method: str, data: dict) -> dict:
    return {"method": method, "data": data, "success": True}

def check_profanity(text: str) -> dict:
    return {"text": text, "is_clean": True, "censored": text}

def autocomplete_address(partial: str) -> dict:
    return {"suggestions": [f"{partial} Street, City, State", f"{partial} Avenue, Town, State"]}

def extract_address(text: str) -> dict:
    return {"addresses": ["123 Main St, City, State 12345"]}

def get_pull_requests(year: int) -> dict:
    return {"year": year, "total_prs": 24, "projects": 10}

def take_screenshot(url: str) -> dict:
    return {"url": url, "screenshot_url": f"https://screenshot.com/{hash(url)}.png"}

def estimate_age(name: str) -> dict:
    return {"name": name, "age": 35, "count": 12345}

def get_api_gratis(service: str) -> dict:
    return {"service": service, "data": "Sample public API data", "available": True}

def get_chinese_character(character: str) -> dict:
    return {"character": character, "pinyin": "zhong", "definition": "middle"}

def search_chinese_text(query: str) -> dict:
    return {"query": query, "results": [{"text": "古文样本", "source": "论语"}]}

def lookup_collins(word: str, lang_from: str, lang_to: str) -> dict:
    return {"word": word, "translation": f"{word}_translated", "examples": []}

def get_free_dictionary(word: str) -> dict:
    return {"word": word, "phonetic": "/wɜːrd/", "meanings": [{"partOfSpeech": "noun", "definitions": []}]}

def manage_asana_task(task_id: str, action: str) -> dict:
    return {"task_id": task_id, "action": action, "status": "completed"}

def manage_clickup_task(task_id: str, action: str) -> dict:
    return {"task_id": task_id, "action": action, "updated": True}

def get_clockify_time_entries(user_id: str) -> dict:
    return {"user_id": user_id, "entries": [{"duration": 3600, "description": "Work"}]}

def convert_file(input_format: str, output_format: str, file_content: str) -> dict:
    return {"input": input_format, "output": output_format, "converted_url": f"https://cloudconvert.com/{hash(file_content)}"}

def validate_email_mailboxlayer(email: str) -> dict:
    return {"email": email, "valid": True, "smtp_check": True}

def validate_email_address(email: str) -> dict:
    return {"email": email, "is_valid": True, "disposable": False}

def validate_cloudmersive(value: str, validation_type: str) -> dict:
    return {"value": value, "type": validation_type, "is_valid": True}

def check_disposable_email(email: str) -> dict:
    return {"email": email, "is_disposable": False}

def get_chuck_norris_joke() -> dict:
    return {"joke": "Chuck Norris can divide by zero", "category": "dev"}

def get_corporate_buzzword() -> dict:
    return {"buzzword": "synergy", "definition": "cooperative interaction"}

def get_excuse(category: str) -> dict:
    return {"excuse": f"Sample {category} excuse", "category": category}

def get_fun_fact() -> dict:
    return {"fact": "Honey never spoils", "category": "science"}

def get_pollen_forecast(lat: float, lon: float) -> dict:
    return {"lat": lat, "lon": lon, "tree_pollen": "low", "grass_pollen": "medium"}

def calculate_carbon_emissions(activity: str, amount: float) -> dict:
    return {"activity": activity, "amount": amount, "co2_kg": amount * 0.5}

def get_emission_factor(activity: str) -> dict:
    return {"activity": activity, "co2_per_unit": 2.3, "unit": "kg"}

def offset_carbon(kg_co2: float) -> dict:
    return {"kg_co2": kg_co2, "offset_cost_usd": kg_co2 * 0.02, "projects": ["reforestation"]}

def search_events_eventbrite(query: str, location: str) -> dict:
    return {"events": [{"name": query, "location": location, "date": "2024-06-01"}]}

def search_events_seatgeek(query: str) -> dict:
    return {"events": [{"title": query, "venue": "Arena", "datetime": "2024-07-15T19:00:00"}]}

def search_events_ticketmaster(keyword: str, city: str) -> dict:
    return {"events": [{"name": keyword, "city": city, "date": "2024-08-20"}]}

def validate_vat(vat_number: str, country_code: str) -> dict:
    return {"vat_number": vat_number, "country": country_code, "valid": True}

def get_insider_trading(ticker: str) -> dict:
    return {"ticker": ticker, "trades": [{"exec": "John Doe", "shares": 1000, "price": 150.0}]}

def get_alpaca_market_data(symbol: str) -> dict:
    return {"symbol": symbol, "price": 175.50, "volume": 5000000}

def get_alpha_vantage_stock(symbol: str) -> dict:
    return {"symbol": symbol, "price": 150.25, "change": 2.5}

def get_marketstack_data(symbol: str) -> dict:
    return {"symbol": symbol, "open": 100.0, "close": 102.5, "volume": 1500000}

def get_banco_brasil_transactions(account_id: str) -> dict:
    return {"account_id": account_id, "transactions": [{"amount": 100.0, "type": "debit"}]}

def validate_iban(iban: str) -> dict:
    return {"iban": iban, "valid": True, "bank": "Sample Bank"}

def create_billplz_bill(amount: float, description: str) -> dict:
    return {"bill_id": f"bill_{int(amount)}", "amount": amount, "url": "https://billplz.com/pay"}

def get_bin_info(bin_number: str) -> dict:
    return {"bin": bin_number, "bank": "Sample Bank", "type": "debit", "country": "US"}

def generate_boleto(amount: float, payer_name: str) -> dict:
    return {"boleto_id": f"bol_{payer_name}", "amount": amount, "barcode": "12345678901234567890"}

def get_citi_account(account_id: str) -> dict:
    return {"account_id": account_id, "balance": 5000.0, "currency": "USD"}

def get_econdb_data(dataset: str, country: str) -> dict:
    return {"dataset": dataset, "country": country, "value": 3.5, "date": "2024-01"}

def get_fed_treasury_data(series: str) -> dict:
    return {"series": series, "value": 4.5, "date": "2024-01-15"}

def get_finage_stock(symbol: str) -> dict:
    return {"symbol": symbol, "price": 200.0, "timestamp": 1234567890}

def get_financial_modeling_prep(symbol: str) -> dict:
    return {"symbol": symbol, "price": 155.0, "pe_ratio": 25.5}

def get_finnhub_quote(symbol: str) -> dict:
    return {"symbol": symbol, "current_price": 180.0, "high": 185.0, "low": 178.0}

def get_fred_data(series_id: str) -> dict:
    return {"series_id": series_id, "value": 2.5, "date": "2024-01-01"}

def front_accounting_api(operation: str) -> dict:
    return {"operation": operation, "status": "success", "data": {}}

def get_hotstoks_data(ticker: str) -> dict:
    return {"ticker": ticker, "price": 95.0, "volume": 2000000}

def get_iex_cloud_quote(symbol: str) -> dict:
    return {"symbol": symbol, "latest_price": 145.0, "market_cap": 2000000000}

def get_ig_market_data(epic: str) -> dict:
    return {"epic": epic, "bid": 100.5, "offer": 101.0}

def get_indian_mutual_fund(scheme_code: str) -> dict:
    return {"scheme_code": scheme_code, "nav": 150.0, "date": "2024-01-15"}

def get_intrinio_data(ticker: str, data_point: str) -> dict:
    return {"ticker": ticker, "data_point": data_point, "value": 75.5}

def create_klarna_session(amount: int, currency: str) -> dict:
    return {"session_id": f"klarna_{amount}", "amount": amount, "currency": currency}

def create_mercadopago_payment(amount: float, description: str) -> dict:
    return {"payment_id": f"mp_{int(amount)}", "amount": amount, "status": "pending"}

def connect_mono_account(account_id: str) -> dict:
    return {"account_id": account_id, "connected": True, "balance": 10000.0}

def create_moov_transfer(amount: float, destination: str) -> dict:
    return {"transfer_id": f"moov_{destination}", "amount": amount, "status": "pending"}

def connect_nordigen_bank(bank_id: str) -> dict:
    return {"bank_id": bank_id, "connected": True, "accounts": ["acc1", "acc2"]}

def get_openfigi(id_value: str, id_type: str) -> dict:
    return {"figi": f"BBG{id_value}", "name": "Sample Security", "exchange": "NYSE"}

def connect_plaid_account(public_token: str) -> dict:
    return {"access_token": f"plaid_{public_token}", "item_id": "item123"}

def get_polygon_data(ticker: str, date: str) -> dict:
    return {"ticker": ticker, "date": date, "close": 150.0, "volume": 1000000}

def optimize_portfolio(tickers: list, weights: list) -> dict:
    return {"tickers": tickers, "optimal_weights": weights, "expected_return": 8.5}

def get_ifsc_details(ifsc_code: str) -> dict:
    return {"ifsc": ifsc_code, "bank": "State Bank", "branch": "Main Branch"}

def get_realtime_finance(symbol: str) -> dict:
    return {"symbol": symbol, "price": 165.0, "timestamp": 1234567890}

def get_sec_edgar(cik: str, filing_type: str) -> dict:
    return {"cik": cik, "filing_type": filing_type, "filings": [{"date": "2024-01-01", "url": "https://sec.gov"}]}

def smartapi_trade(symbol: str, action: str, quantity: int) -> dict:
    return {"symbol": symbol, "action": action, "quantity": quantity, "order_id": f"ord_{symbol}"}

def get_stockdata(ticker: str) -> dict:
    return {"ticker": ticker, "price": 140.0, "sentiment": "positive"}

def get_styvio_data(symbol: str) -> dict:
    return {"symbol": symbol, "price": 132.0, "sentiment_score": 0.75}

def validate_tax_number(tax_number: str, country: str) -> dict:
    return {"tax_number": tax_number, "country": country, "valid": True}

def get_tradier_quote(symbol: str) -> dict:
    return {"symbol": symbol, "last": 155.0, "bid": 154.5, "ask": 155.5}

def get_twelve_data(symbol: str, interval: str) -> dict:
    return {"symbol": symbol, "interval": interval, "price": 148.0}

def get_wallstreetbets_sentiment(ticker: str) -> dict:
    return {"ticker": ticker, "sentiment": "bullish", "comment_count": 150}

def get_yahoo_finance(symbol: str) -> dict:
    return {"symbol": symbol, "price": 175.0, "change_percent": 1.5}

def get_ynab_budget(budget_id: str) -> dict:
    return {"budget_id": budget_id, "name": "Monthly Budget", "balance": 1500.0}

def get_zoho_books_invoice(invoice_id: str) -> dict:
    return {"invoice_id": invoice_id, "total": 500.0, "status": "paid"}

def get_bacon_image(width: int, height: int) -> dict:
    return {"url": f"https://baconmockup.com/{width}/{height}", "width": width, "height": height}

def get_food_product(barcode: str) -> dict:
    return {"barcode": barcode, "name": "Sample Product", "calories": 200}

def get_brewery(brewery_id: int) -> dict:
    return {"id": brewery_id, "name": f"Brewery_{brewery_id}", "city": "Portland"}

def get_food_facts(barcode: str) -> dict:
    return {"barcode": barcode, "product_name": "Sample Food", "nutrition_grade": "a"}

def get_aoe2_resource(resource_id: int) -> dict:
    return {"id": resource_id, "name": "Villager", "cost": {"food": 50}}

def get_amiibo(amiibo_id: str) -> dict:
    return {"id": amiibo_id, "name": "Mario", "series": "Super Mario"}

def get_autochess_data(player_id: str) -> dict:
    return {"player_id": player_id, "rank": 1500, "matches": 100}

def get_barter_game(game_id: int) -> dict:
    return {"id": game_id, "title": "Sample Game", "platform": "PC"}

def geolocate_ip(ip_address: str) -> dict:
    return {"ip": ip_address, "country": "United States", "city": "New York", "lat": 40.7128, "lon": -74.0060}

def process_grass_gis(operation: str, parameters: dict) -> dict:
    return {"operation": operation, "parameters": parameters, "status": "completed"}

def get_administrative_division(country: str, level: int) -> dict:
    return {"country": country, "level": level, "divisions": ["Division1", "Division2"]}

def geocode_address_france(address: str) -> dict:
    return {"address": address, "lat": 48.8566, "lon": 2.3522}

def get_bc_law(law_id: str) -> dict:
    return {"law_id": law_id, "title": "Sample Law", "content": "Law content..."}

def get_brazil_data(dataset: str) -> dict:
    return {"dataset": dataset, "data": {"population": 213000000}}

def get_brazil_company(cnpj: str) -> dict:
    return {"cnpj": cnpj, "name": "Sample Company", "status": "active"}

def get_federal_code(agency: str) -> dict:
    return {"agency": agency, "projects": 50, "languages": ["Python", "JavaScript"]}

def get_cms_data(dataset: str) -> dict:
    return {"dataset": dataset, "records": 1000, "updated": "2024-01-01"}

def get_covid_data(country: str) -> dict:
    return {"country": country, "cases": 1000000, "deaths": 50000}

def get_covid_datenhub(region: str) -> dict:
    return {"region": region, "datasets": 10, "last_updated": "2024-01-15"}

def get_government_response(country: str) -> dict:
    return {"country": country, "stringency_index": 65.5, "measures": 15}

def search_job_adzuna(query: str, location: str) -> dict:
    return {"jobs": [{"title": query, "location": location, "salary": 50000}]}

def search_job_arbeitnow(query: str) -> dict:
    return {"jobs": [{"title": query, "company": "Tech Corp", "remote": True}]}

def search_job_arbeitsamt(occupation: str, region: str) -> dict:
    return {"jobs": [{"title": occupation, "region": region, "openings": 5}]}

def search_job_careerjet(query: str, location: str) -> dict:
    return {"jobs": [{"title": query, "location": location, "url": "https://careerjet.com/job"}]}

def process_thai_ai(text: str, operation: str) -> dict:
    return {"text": text, "operation": operation, "result": "processed"}

def analyze_image_cloudmersive(image_url: str, analysis_type: str) -> dict:
    return {"image_url": image_url, "type": analysis_type, "nsfw_score": 0.05}

def process_dialogflow(text: str, session_id: str) -> dict:
    return {"text": text, "intent": "greeting", "confidence": 0.95}

def filter_text_exude(text: str, operation: str) -> dict:
    return {"original": text, "filtered": text.replace("stop", ""), "operation": operation}

def get_7digital_track(track_id: int) -> dict:
    return {"id": track_id, "title": f"Track_{track_id}", "artist": "Artist"}

def master_audio(audio_url: str) -> dict:
    return {"input_url": audio_url, "mastered_url": f"https://master.com/{hash(audio_url)}.mp3"}

def get_audiomack_track(track_id: str) -> dict:
    return {"id": track_id, "title": "Sample Track", "artist": "Artist", "plays": 10000}

def search_bandcamp(query: str) -> dict:
    return {"results": [{"title": query, "artist": "Indie Artist", "url": "https://bandcamp.com"}]}

def get_news_mediastack(category: str, country: str) -> dict:
    return {"articles": [{"title": "News Title", "category": category, "country": country}]}

def search_news_ap(query: str) -> dict:
    return {"articles": [{"headline": query, "date": "2024-01-15", "url": "https://ap.org"}]}

def search_historical_newspapers(query: str, year: int) -> dict:
    return {"results": [{"title": query, "year": year, "page_url": "https://chronicling.loc.gov"}]}

def get_currents_news(category: str, language: str) -> dict:
    return {"news": [{"title": "Breaking News", "category": category, "language": language}]}

def get_api_setu_service(service: str) -> dict:
    return {"service": service, "available": True, "description": "Government API service"}

def get_black_history_fact(fact_id: int) -> dict:
    return {"id": fact_id, "fact": "Sample black history fact", "date": "1963-08-28"}

def get_telegram_bot(bot_username: str) -> dict:
    return {"username": bot_username, "description": "Sample bot", "rating": 4.5}

def get_carto_prediction(lat: float, lon: float, data_type: str) -> dict:
    return {"lat": lat, "lon": lon, "prediction": 75.5, "type": data_type}

def get_countly_analytics(app_id: str, metric: str) -> dict:
    return {"app_id": app_id, "metric": metric, "value": 1500}

def search_creative_commons(query: str, license_type: str) -> dict:
    return {"results": [{"title": query, "license": license_type, "url": "https://creativecommons.org"}]}

def search_datamuse(word: str, relation: str) -> dict:
    return {"word": word, "results": ["synonym1", "synonym2"], "relation": relation}

def query_metabase(query_id: int) -> dict:
    return {"query_id": query_id, "results": [{"col1": "data1", "col2": "data2"}]}

def search_epo_patent(query: str) -> dict:
    return {"patents": [{"title": query, "patent_number": "EP1234567", "date": "2020-01-01"}]}

def search_patentsview(query: str) -> dict:
    return {"patents": [{"title": query, "inventor": "John Doe", "year": 2019}]}

def search_tipo_patent(query: str) -> dict:
    return {"patents": [{"title": query, "number": "TW123456", "date": "2021-05-01"}]}

def search_uspto_patent(query: str) -> dict:
    return {"patents": [{"title": query, "patent_number": "US10123456", "grant_date": "2022-03-15"}]}

def get_advice() -> dict:
    return {"slip": {"id": 123, "advice": "Always be yourself"}}

def get_biriyani_image() -> dict:
    return {"url": "https://biriyani.anoram.com/random.jpg", "type": "Hyderabadi"}

def get_dev_article(article_id: int) -> dict:
    return {"id": article_id, "title": "Sample Article", "author": "Dev User", "tags": ["python"]}

def get_dictum_quote(quote_id: int) -> dict:
    return {"id": quote_id, "quote": "Be the change you wish to see", "author": "Gandhi"}

def validate_phone_number(phone: str, country_code: str) -> dict:
    return {"phone": phone, "country": country_code, "valid": True, "type": "mobile"}

def validate_numverify(phone: str) -> dict:
    return {"phone": phone, "valid": True, "country_code": "US", "carrier": "Verizon"}

def validate_phone_cloudmersive(phone: str) -> dict:
    return {"phone": phone, "is_valid": True, "phone_type": "Mobile"}

def get_phone_specs(brand: str, model: str) -> dict:
    return {"brand": brand, "model": model, "specs": {"screen": "6.1 inch", "ram": "8GB"}}

def generate_image_bruzu(text: str, template: str) -> dict:
    return {"text": text, "template": template, "image_url": f"https://bruzu.com/{hash(text)}.png"}

def optimize_image_cheetaho(image_url: str, width: int, height: int) -> dict:
    return {"original": image_url, "optimized": f"https://cheetaho.com/opt/{width}x{height}.jpg"}

def manipulate_image_dagpi(image_url: str, effect: str) -> dict:
    return {"original": image_url, "effect": effect, "result_url": f"https://dagpi.com/{effect}.png"}

def generate_image_duply(template_id: str, data: dict) -> dict:
    return {"template": template_id, "data": data, "url": f"https://duply.co/img/{template_id}.png"}

def get_codeforces_user(handle: str) -> dict:
    return {"handle": handle, "rating": 1800, "rank": "Expert"}

def compile_code_hackerearth(code: str, language: str) -> dict:
    return {"code": code, "language": language, "output": "Hello World", "status": "success"}

def get_coding_contests() -> dict:
    return {"contests": [{"name": "Codeforces Round", "start_time": "2024-06-01T14:00:00"}]}

def generate_docs_mintlify(code: str, language: str) -> dict:
    return {"code": code, "language": language, "documentation": "Function does X"}

def get_astronomy_data(object_name: str) -> dict:
    return {"name": object_name, "type": "galaxy", "distance_ly": 2500000}

def search_arxiv(query: str, category: str) -> dict:
    return {"papers": [{"title": query, "category": category, "authors": ["Author 1"]}]}

def search_core(query: str) -> dict:
    return {"papers": [{"title": query, "year": 2023, "url": "https://core.ac.uk"}]}

def get_gbif_species(species_name: str) -> dict:
    return {"name": species_name, "scientific_name": "Species scientificus", "kingdom": "Animalia"}

def scan_binaryedge(target: str) -> dict:
    return {"target": target, "open_ports": [80, 443], "services": ["http", "https"]}

def get_bitwarden_vault(user_id: str) -> dict:
    return {"user_id": user_id, "items": 15, "folders": 3}

def detect_bot(user_agent: str, ip: str) -> dict:
    return {"user_agent": user_agent, "ip": ip, "is_bot": False, "bot_type": None}

def get_bugcrowd_submission(submission_id: str) -> dict:
    return {"id": submission_id, "severity": "medium", "status": "triaged"}

def search_bestbuy_product(query: str) -> dict:
    return {"products": [{"name": query, "price": 299.99, "sku": "12345"}]}

def search_digikey_part(part_number: str) -> dict:
    return {"part_number": part_number, "price": 1.50, "stock": 1000}

def get_dummy_product(product_id: int) -> dict:
    return {"id": product_id, "name": f"Product_{product_id}", "price": 29.99}

def search_ebay(query: str, category: str) -> dict:
    return {"items": [{"title": query, "price": 49.99, "category": category}]}

def get_4chan_board(board: str) -> dict:
    return {"board": board, "threads": 100, "posts_per_hour": 500}

def post_to_social_ayrshare(message: str, platforms: list) -> dict:
    return {"message": message, "platforms": platforms, "status": "posted", "post_ids": ["123", "456"]}

def get_horoscope_aztro(sign: str, day: str) -> dict:
    return {"sign": sign, "day": day, "horoscope": "Today is a good day", "mood": "happy"}

def get_blogger_posts(blog_id: str) -> dict:
    return {"blog_id": blog_id, "posts": [{"title": "Post 1", "content": "Content..."}]}

def get_football_info(league: str, season: int) -> dict:
    return {"league": league, "season": season, "teams": 20, "matches": 380}

def check_symptoms(symptoms: list, age: int, gender: str) -> dict:
    return {"symptoms": symptoms, "diagnosis": [{"issue": "Common Cold", "accuracy": 75}]}

def get_nba_stats(player: str) -> dict:
    return {"player": player, "ppg": 25.5, "rpg": 7.2, "apg": 5.0}

def get_city_bikes(city: str) -> dict:
    return {"city": city, "stations": 50, "available_bikes": 300}

def generate_bacon_ipsum(paragraphs: int) -> dict:
    return {"paragraphs": paragraphs, "text": "Bacon ipsum dolor amet..."}

def generate_avatar_dicebear(seed: str, style: str) -> dict:
    return {"seed": seed, "style": style, "url": f"https://avatars.dicebear.com/api/{style}/{seed}.svg"}

def get_random_word(count: int) -> dict:
    return {"words": [f"word{i}" for i in range(count)], "pronunciation": ["wɜːrd"]}

def generate_fake_data(schema: dict) -> dict:
    return {"data": [{"name": "John Doe", "email": "john@example.com"}], "schema": schema}

def detect_code_language(code: str) -> dict:
    return {"code": code, "language": "python", "confidence": 0.95}

def detect_language_apilayer(text: str) -> dict:
    return {"text": text, "language": "en", "confidence": 0.98}

def analyze_text_aylien(text: str, analysis_type: str) -> dict:
    return {"text": text, "type": analysis_type, "sentiment": "positive"}

def detect_language_api(text: str) -> dict:
    return {"text": text, "language": "English", "iso_code": "en"}

def track_shipment_aftership(tracking_number: str) -> dict:
    return {"tracking": tracking_number, "status": "in_transit", "location": "New York"}

def track_correios(tracking_code: str) -> dict:
    return {"code": tracking_code, "status": "Em trânsito", "location": "São Paulo"}

def track_habit_pixela(user: str, graph_id: str, date: str) -> dict:
    return {"user": user, "graph": graph_id, "date": date, "quantity": 1}

def get_pincode_info(pincode: str) -> dict:
    return {"pincode": pincode, "area": "Mumbai", "state": "Maharashtra", "country": "India"}

def get_adsb_flight(icao24: str) -> dict:
    return {"icao24": icao24, "callsign": "UAL123", "altitude": 35000, "velocity": 450}

def get_airport_info(icao_code: str) -> dict:
    return {"icao": icao_code, "name": "Sample Airport", "url": "https://airport.com"}

def get_ais_vessel(mmsi: str) -> dict:
    return {"mmsi": mmsi, "name": "Ship Name", "lat": 40.7128, "lon": -74.0060}

def search_flights_amadeus(origin: str, destination: str, date: str) -> dict:
    return {"flights": [{"origin": origin, "destination": destination, "price": 350, "date": date}]}

def shorten_url_cleanuri(url: str) -> dict:
    return {"original": url, "shortened": f"https://cleanuri.com/{hash(url)}"}

def track_link_clickmeter(link_id: str) -> dict:
    return {"link_id": link_id, "clicks": 150, "unique_clicks": 100}

def shorten_url_drivet(url: str) -> dict:
    return {"original": url, "short_url": f"https://drvt.it/{hash(url)[:6]}"}

def shorten_url_free(url: str) -> dict:
    return {"original_url": url, "short_url": f"https://free.us/{hash(url)[:8]}"}

def get_helipad(helipad_id: str) -> dict:
    return {"id": helipad_id, "name": "Downtown Helipad", "lat": 34.0522, "lon": -118.2437}

def get_kbb_vehicle(vin: str) -> dict:
    return {"vin": vin, "make": "Toyota", "model": "Camry", "year": 2022, "value": 25000}

def get_mercedes_vehicle(vin: str) -> dict:
    return {"vin": vin, "model": "C-Class", "fuel_level": 75, "location": {"lat": 48.8566, "lon": 2.3522}}

def get_nhtsa_vehicle(vin: str) -> dict:
    return {"vin": vin, "make": "Ford", "model": "F-150", "year": 2021, "recalls": 0}

def get_doctor_who_info(episode_id: int) -> dict:
    return {"id": episode_id, "title": "The Doctor's Episode", "doctor": "13th Doctor"}

def get_next_episode(show: str) -> dict:
    return {"show": show, "next_episode": "S05E10", "air_date": "2024-06-15"}

def get_czech_tv_schedule(channel: str, date: str) -> dict:
    return {"channel": channel, "date": date, "programs": [{"time": "20:00", "title": "News"}]}

def get_dailymotion_video(video_id: str) -> dict:
    return {"id": video_id, "title": "Sample Video", "views": 10000, "duration": 300}

def get_weather_micro(lat: float, lon: float) -> dict:
    return {"lat": lat, "lon": lon, "temp": 22, "condition": "sunny"}

def get_odweather_forecast(location: str) -> dict:
    return {"location": location, "temp": 18, "humidity": 65, "webcam_url": "https://webcam.com"}

def get_oikolab_weather(lat: float, lon: float, date: str) -> dict:
    return {"lat": lat, "lon": lon, "date": date, "temp": 15, "precipitation": 5}

def get_open_meteo_forecast(lat: float, lon: float) -> dict:
    return {"lat": lat, "lon": lon, "temperature": 20, "wind_speed": 10, "forecast": "partly_cloudy"}


# 定义各工具参数 schema
class Tool_get_by_application_id(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the bank application")
class Tool_ValidateExistingRiskProfile(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client whose risk profile validity is being checked")
    transaction_timestamp: str = Field(..., description="Current transaction initiation timestamp in ISO 8601 format")
    previous_risk_profile: Optional[str] = Field(None, description="The client's existing risk profile classification if one exists")
    previous_risk_profile_expiry_date: Optional[str] = Field(None, description="The expiration date of the existing risk profile in ISO 8601 format")

class Tool_AdministerRiskQuestionnaire(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client completing the risk tolerance questionnaire")
    client_age: int = Field(..., description="Client's age in years, used for age bracket scoring")
    client_annual_income: float = Field(..., description="Client's declared annual income in USD")
    client_investment_experience_years: int = Field(..., description="Number of years of investment experience")
    client_liquidity_needs: str = Field(..., description="Client's liquidity requirements classification")
    risk_questionnaire_responses: Dict[str, Any] = Field(..., description="JSON object containing structured questionnaire answers")

class Tool_CalculateRiskProfile(BaseModel):
    risk_questionnaire_total_score: int = Field(..., description="Total numerical score from risk tolerance questionnaire ranging from 0 to 100 points")

class Tool_AuthorizeAndActivateRiskProfile(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client authorizing their calculated risk profile")
    calculated_risk_profile: str = Field(..., description="The system-calculated risk profile classification")
    client_authorization_method: str = Field(..., description="Method used to capture legally binding client consent")
    transaction_timestamp: str = Field(..., description="Current timestamp in ISO 8601 format")

class Tool_RecommendSuitableProducts(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client for whom suitable investment products are being recommended")
    calculated_risk_profile: str = Field(..., description="Client's authorized risk profile classification")
    investable_status: str = Field(..., description="Client's current investable status flag")

class Tool_CreateInvestmentOrder(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client placing the investment order")
    relationship_manager_id: str = Field(..., description="Unique identifier for the Relationship Manager")
    product_code: str = Field(..., description="Unique product identifier code from the product master database")
    subscription_amount: float = Field(..., description="Total investment amount in USD")
    settlement_account_number: str = Field(..., description="Client's bank account number")
    transaction_timestamp: str = Field(..., description="Transaction initiation timestamp in ISO 8601 format")
    branch_code: str = Field(..., description="Code identifying the originating branch location")
    channel: str = Field(..., description="Distribution channel through which the transaction was initiated")
    region: str = Field(..., description="Geographic region code for the transaction")

class Tool_ValidateProductSuitability(BaseModel):
    transaction_id: str = Field(..., description="Unique system-generated transaction identifier")
    calculated_risk_profile: str = Field(..., description="Client's authorized risk profile classification")
    product_risk_rating: str = Field(..., description="Risk rating of the selected investment product")

class Tool_ExecuteAMLSanctionsScreening(BaseModel):
    client_id: str = Field(..., description="Unique client identifier to be screened")
    client_name: str = Field(..., description="Client's full legal name to be screened")
    transaction_id: str = Field(..., description="Unique transaction identifier")

class Tool_ValidateThresholdAndSOW(BaseModel):
    subscription_amount: float = Field(..., description="Total investment subscription amount in USD")
    monitoring_threshold_amount: float = Field(..., description="Configured regulatory monitoring threshold in USD")
    source_of_wealth_documented: bool = Field(..., description="Boolean flag indicating whether Source of Wealth documentation is on file")

class Tool_VerifyFundsAvailability(BaseModel):
    settlement_account_number: str = Field(..., description="Client's bank account number")
    subscription_amount: float = Field(..., description="Total investment subscription amount in USD")

class Tool_AggregateComplianceChecks(BaseModel):
    suitability_check_passed: bool = Field(..., description="Boolean result from ValidateProductSuitability")
    aml_sanctions_screening_status: str = Field(..., description="Status from ExecuteAMLSanctionsScreening")
    transaction_threshold_exceeded: bool = Field(..., description="Boolean from ValidateThresholdAndSOW")
    source_of_wealth_documented: bool = Field(..., description="Boolean from ValidateThresholdAndSOW")
    funds_availability_verified: bool = Field(..., description="Boolean result from VerifyFundsAvailability")

class Tool_CalculateTransactionFinancials(BaseModel):
    subscription_amount: float = Field(..., description="Total investment amount in USD")
    front_end_load_percentage: float = Field(..., description="Sales charge percentage from product master database")
    annual_management_fee_percentage: float = Field(..., description="Ongoing annual management fee percentage")
    current_nav: float = Field(..., description="Current Net Asset Value per unit")
    product_category: str = Field(..., description="Product classification")
    pre_trade_compliance_passed: bool = Field(..., description="Consolidated compliance validation result")

class Tool_ManageCoolingOffPeriod(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    product_category: str = Field(..., description="Product classification")
    cooling_off_period_required: bool = Field(..., description="Boolean flag indicating whether cooling-off period applies")
    cooling_off_period_hours: int = Field(..., description="Duration of cooling-off period in hours")
    client_authorization_timestamp: str = Field(..., description="ISO 8601 timestamp when client authorized the transaction")

class Tool_ExecuteInvestmentTransaction(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    product_category: str = Field(..., description="Product classification")
    product_code: str = Field(..., description="Unique product identifier")
    net_investment_amount: float = Field(..., description="Actual capital deployed into investment product")
    settlement_account_number: str = Field(..., description="Client's bank account")
    client_id: str = Field(..., description="Unique client identifier")
    transaction_status: str = Field(..., description="Current transaction status")
    subscription_amount: float = Field(..., description="Total gross investment amount")

class Tool_CalculateSTPFlag(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    suitability_mismatch_flag: bool = Field(..., description="Boolean from ValidateProductSuitability")
    supervisor_approval_required: bool = Field(..., description="Boolean indicating whether transaction required supervisor approval")
    aml_sanctions_screening_status: str = Field(..., description="AML screening outcome")

class Tool_ExecutePostTradeComplianceReview(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    audio_video_recording_id: Optional[str] = Field(None, description="Unique identifier for audio/video recording")
    suitability_mismatch_flag: bool = Field(..., description="Boolean from ValidateProductSuitability")
    transaction_threshold_exceeded: bool = Field(..., description="Boolean from ValidateThresholdAndSOW")
    disclosure_documents_presented: List[str] = Field(..., description="Array of document types presented to client")

class Tool_CalculateProcessingMetrics(BaseModel):
    transaction_timestamp: str = Field(..., description="Transaction initiation timestamp in ISO 8601 format")
    execution_timestamp: str = Field(..., description="Transaction execution completion timestamp in ISO 8601 format")

#==============冗余API类定义================
class Tool_DocumentImageResolutionValidation(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the loan application.")
    submitted_documents: List[str] = Field(..., description="List of document keys or URLs uploaded by the customer (e.g., ID card, proof of income).")

class Tool_MaterialIntegrityScoreCalculation(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the loan application.")
    submitted_documents: List[str] = Field(..., description="List of document keys or URLs to be evaluated.")
    image_clarity_scores: Dict[str, float] = Field(..., description="Dictionary mapping document key to its image clarity score (0.0 to 1.0).")
    text_recognition_accuracy: Dict[str, float] = Field(..., description="Dictionary mapping document key to its OCR text recognition accuracy score (0.0 to 1.0).")
    loan_type: str = Field(..., description="Type of loan (e.g., 'personal', 'mortgage', 'auto').")
    customer_occupation_type: str = Field(..., description="Category of the customer's occupation (e.g., 'SALARIED', 'SELF_EMPLOYED').")

class Tool_IdentityAuthenticationAndFaceMatching(BaseModel):
    application_id: str = Field(..., description="Unique identifier for the loan application.")
    customer_id: str = Field(..., description="Internal system identifier for the customer profile.")
    id_card_number: str = Field(..., description="Official identification number from the ID document.")
    id_card_name: str = Field(..., description="Full name as printed on the ID document.")
    id_card_images: Dict[str, str] = Field(..., description="Dictionary containing 'front' and 'back' image keys/URLs of the ID card.")
    face_photo: str = Field(..., description="Key or URL of the customer's selfie or live capture photo for face matching.")

# 封装这些服务为工具列表
tools = [
    StructuredTool.from_function(
        func=get_by_application_id,
        name="get_by_application_id",
        description="Retrieve application data by application_id",
        args_schema=Tool_get_by_application_id
    ),    
    StructuredTool.from_function(
        func=ValidateExistingRiskProfile,
        name="ValidateExistingRiskProfile",
        description="Validates whether the client has an existing, unexpired risk profile assessment on file by comparing the previous_risk_profile_expiry_date against the current transaction_timestamp. Determines if a new risk assessment questionnaire administration is required or if the existing profile can be utilized for the investment transaction.",
        args_schema=Tool_ValidateExistingRiskProfile
    ),
    StructuredTool.from_function(
        func=AdministerRiskQuestionnaire,
        name="AdministerRiskQuestionnaire",
        description="Administers the regulatory-mandated Risk Tolerance Questionnaire to the client, collecting structured responses across multiple dimensions. Applies weighted scoring algorithm based on questionnaire responses to generate a numerical risk_questionnaire_total_score ranging from 0 to 100 points.",
        args_schema=Tool_AdministerRiskQuestionnaire
    ),
    StructuredTool.from_function(
        func=CalculateRiskProfile,
        name="CalculateRiskProfile",
        description="Applies deterministic mapping algorithm to convert numerical risk_questionnaire_total_score into categorical risk profile classification using regulatory-defined score ranges: Conservative (0-35 points), Balanced (36-65 points), and Aggressive (66-100 points).",
        args_schema=Tool_CalculateRiskProfile
    ),
    StructuredTool.from_function(
        func=AuthorizeAndActivateRiskProfile,
        name="AuthorizeAndActivateRiskProfile",
        description="Presents the calculated risk profile to the client for review and explicit consent, captures client authorization via configured method, computes risk profile expiry date by adding 730 days to current timestamp, and updates client master record to set investable_status to 'Active'.",
        args_schema=Tool_AuthorizeAndActivateRiskProfile
    ),
    StructuredTool.from_function(
        func=RecommendSuitableProducts,
        name="RecommendSuitableProducts",
        description="Queries product master database applying hierarchical suitability filters based on client calculated_risk_profile to return only investment products whose product_risk_rating aligns with or is lower than client risk tolerance according to regulatory suitability matrix.",
        args_schema=Tool_RecommendSuitableProducts
    ),
    StructuredTool.from_function(
        func=CreateInvestmentOrder,
        name="CreateInvestmentOrder",
        description="Creates comprehensive new transaction record with system-generated transaction_id, captures all order parameters including client identification, relationship manager, product selection, subscription amount, and settlement account.",
        args_schema=Tool_CreateInvestmentOrder
    ),
    StructuredTool.from_function(
        func=ValidateProductSuitability,
        name="ValidateProductSuitability",
        description="Executes automated real-time suitability check by comparing product_risk_rating against client calculated_risk_profile using hierarchical matching matrix enforcing regulatory suitability rules.",
        args_schema=Tool_ValidateProductSuitability
    ),
    StructuredTool.from_function(
        func=ExecuteAMLSanctionsScreening,
        name="ExecuteAMLSanctionsScreening",
        description="Interfaces with enterprise AML monitoring platform to perform real-time sanctions list screening of client_id and client_name against all relevant sanctions databases.",
        args_schema=Tool_ExecuteAMLSanctionsScreening
    ),
    StructuredTool.from_function(
        func=ValidateThresholdAndSOW,
        name="ValidateThresholdAndSOW",
        description="Compares subscription_amount against configured monitoring_threshold_amount to determine if transaction exceeds regulatory large-transaction threshold requiring enhanced due diligence.",
        args_schema=Tool_ValidateThresholdAndSOW
    ),
    StructuredTool.from_function(
        func=VerifyFundsAvailability,
        name="VerifyFundsAvailability",
        description="Queries Core Banking System to retrieve real-time available_balance for specified settlement_account_number and compares retrieved available_balance against subscription_amount to confirm sufficient liquid funds exist.",
        args_schema=Tool_VerifyFundsAvailability
    ),
    StructuredTool.from_function(
        func=AggregateComplianceChecks,
        name="AggregateComplianceChecks",
        description="Aggregates validation results from all parallel pre-trade compliance checks including suitability validation, AML sanctions screening, threshold/SOW validation, and funds availability verification. Applies boolean AND logic across all validation outcomes.",
        args_schema=Tool_AggregateComplianceChecks
    ),
    StructuredTool.from_function(
        func=CalculateTransactionFinancials,
        name="CalculateTransactionFinancials",
        description="Computes comprehensive transaction financial metrics with regulatory precision including front-end load amount, net investment amount, estimated units, and annual management fee amount. Only executes when pre_trade_compliance_passed = TRUE.",
        args_schema=Tool_CalculateTransactionFinancials
    ),
    StructuredTool.from_function(
        func=ManageCoolingOffPeriod,
        name="ManageCoolingOffPeriod",
        description="Evaluates product_category against cooling-off period requirements to determine applicability. When applicable, calculates cooling-off expiry timestamp by adding cooling_off_period_hours to client_authorization_timestamp.",
        args_schema=Tool_ManageCoolingOffPeriod
    ),
    StructuredTool.from_function(
        func=ExecuteInvestmentTransaction,
        name="ExecuteInvestmentTransaction",
        description="Final execution step consummating investment transaction through appropriate channel routing. For Mutual_Fund products, transmits electronic purchase order to fund house gateway. For structured products, routes order to Treasury/Global Markets desk.",
        args_schema=Tool_ExecuteInvestmentTransaction
    ),
    StructuredTool.from_function(
        func=CalculateSTPFlag,
        name="CalculateSTPFlag",
        description="Evaluates entire transaction lifecycle to determine if straight-through processing was achieved by checking for any manual intervention indicators. Sets straight_through_processing_flag to TRUE only for zero-touch transactions.",
        args_schema=Tool_CalculateSTPFlag
    ),
    StructuredTool.from_function(
        func=ExecutePostTradeComplianceReview,
        name="ExecutePostTradeComplianceReview",
        description="Performs post-execution quality assurance and regulatory compliance verification by reviewing transaction against standardized compliance checklist. Documents findings and sets compliance_defect_flag if violations or deficiencies identified.",
        args_schema=Tool_ExecutePostTradeComplianceReview
    ),
    StructuredTool.from_function(
        func=CalculateProcessingMetrics,
        name="CalculateProcessingMetrics",
        description="Computes end-to-end transaction processing efficiency metrics by calculating total_processing_time_minutes as elapsed time between transaction_timestamp (initiation) and execution_timestamp (completion).",
        args_schema=Tool_CalculateProcessingMetrics
    ),
#==============冗余API工具列表封装================    
    StructuredTool.from_function(
        func=DocumentImageResolutionValidation,
        name="DocumentImageResolutionValidation",
        description="Validates the image clarity and resolution of documents uploaded for a loan application. Ensures uploaded images are clear, legible, and suitable for automated text recognition and manual review. This step is crucial for downstream document integrity assessment.",
        args_schema=Tool_DocumentImageResolutionValidation
    ),
    StructuredTool.from_function(
        func=MaterialIntegrityScoreCalculation,
        name="MaterialIntegrityScoreCalculation",
        description="Calculates a comprehensive integrity score for all application documents based on image clarity and text recognition accuracy. The score reflects the overall quality and trustworthiness of the submitted evidence, forming a key input for risk assessment.",
        args_schema=Tool_MaterialIntegrityScoreCalculation
    ),
    StructuredTool.from_function(
        func=IdentityAuthenticationAndFaceMatching,
        name="IdentityAuthenticationAndFaceMatching",
        description="Performs identity proofing by verifying government ID details and conducting a facial comparison between the ID photo and a live selfie. This is a primary step in Know Your Customer (KYC) and fraud prevention.",
        args_schema=Tool_IdentityAuthenticationAndFaceMatching
    ),
]

# 3. 初始化cluade4.5模型
chat_model = ChatOpenAI(
    api_key="",  # 请替换为你的 API Key
    base_url="https://",  # API 中转地址
    model="deepseek-chat",
    temperature=0.1
)




class ExecutionLogCallback(BaseCallbackHandler):
    """自定义回调函数，捕获所有工具调用和响应"""
    
    def __init__(self):
        self.execution_log = []
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """工具调用开始时"""
        tool_name = serialized.get("name", "unknown_tool")
        # 解析 input_str 为字典
        try:
            import json
            tool_input = json.loads(input_str) if isinstance(input_str, str) else input_str
        except:
            tool_input = input_str
        
        self.execution_log.append(f"Invoking: `{tool_name}` with `{tool_input}`")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """工具调用结束时"""
        self.execution_log.append(f"responded: {output}")
        self.execution_log.append("")
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> Any:
        """工具调用出错时"""
        self.execution_log.append(f"Tool Error: {type(error).__name__}: {str(error)}")
        self.execution_log.append("")
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Agent 决定执行某个 action 时"""
        thought_text = getattr(action, 'log', '')
        if thought_text:
            self.execution_log.append(f"Thought: {thought_text}")

def process_one_record(application_id: str):
    """
    用一条数据跑一次 agent 并把结果保存成以 application_id 命名的 txt 文件。
    通过回调函数捕获所有工具调用，即使发生错误也能保存已执行的步骤。
    """
    # 初始化执行日志和回调
    execution_log = []
    execution_log.append("> Entering new AgentExecutor chain...\n")
    
    # 创建自定义回调
    callback = ExecutionLogCallback()
    
    try:
        # 构造 prompt 文本
        with open(PROMPT_YAML_PATH, "r", encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)
            base_prompt = data_yaml["prompts"]["text"]
        
        user_request = f"Please help me process a business application, the application ID is {application_id}."
        prompt_text = base_prompt.replace("{{SOP}}", sop_text)
        prompt_text = prompt_text.replace("{{USER_REQUEST}}", user_request)

        # 更新 prompt 到 agent 的提示里
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that uses the provided Standard Operating Procedure (SOP) to handle user requests. When you call an API, the input parameters may be obtained by querying a database or by retrieving the outputs of other API calls. Upon receiving a request to process a business application with a specific ID, you must first use the database retrieval function to fetch that entry. Fields that already exist in the database can be used directly for computation and can also be modified via update APIs. Empty fields in the data need to be obtained by calling the appropriate APIs and then updated and written back. Every time you obtain a result from an API, you must write this new result back into that data entry. Additionally, some of the APIs provided to you are invalid; you must call the valid APIs in the correct sequence to obtain the final result."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", prompt_text),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 更新 agent 的 prompt
        agent = create_tool_calling_agent(
            llm=chat_model,
            tools=tools,
            prompt=prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # 关闭verbose，我们用回调捕获
            return_intermediate_steps=True,
            max_iterations=50,
            handle_parsing_errors=True,
            early_stopping_method="generate"
        )

        try:
            # 执行Agent，传入回调函数
            result = agent_executor.invoke(
                {"input": prompt_text},
                config={"callbacks": [callback]}
            )
            
            # 将回调捕获的日志添加到执行日志
            execution_log.extend(callback.execution_log)
            
            # 重构verbose格式的输出（保留原有逻辑作为备份）
            if "intermediate_steps" in result:
                for i, step in enumerate(result["intermediate_steps"], 1):
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        
                        # 重构 Thought 部分（从action中提取）
                        thought_text = getattr(action, 'log', '')
                        if thought_text and f"Thought: {thought_text}" not in execution_log:
                            execution_log.append(f"Thought: {thought_text}")
                        
                        # 重构 Observation 部分
                        obs_text = str(observation)
                        if f"Observation: {obs_text}" not in execution_log:
                            execution_log.append(f"Observation: {obs_text}")
                            execution_log.append("")
            
            # 记录最终输出
            if "output" in result:
                execution_log.append(f"Final Answer: {result['output']}")
            
            execution_log.append("> Finished chain.")
            
        except Exception as agent_error:
            # 即使出错也记录已执行的步骤（包括回调捕获的）
            execution_log.extend(callback.execution_log)
            execution_log.append(f"\n⚠️ Agent execution failed: {type(agent_error).__name__}: {agent_error}")
            execution_log.append("> Chain interrupted.")
            
            # 记录堆栈跟踪
            import traceback
            execution_log.append("\nStack trace:")
            execution_log.append(traceback.format_exc())

        # 保存文件
        filename = f"{application_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(execution_log))
            
        print(f"✅ {application_id} - 执行记录已保存 ({len(execution_log)} 行)")
            
    except Exception as e:
        # 初始化错误的处理
        error_log = [
            "> Entering new AgentExecutor chain...",
            f"❌ Initialization failed: {type(e).__name__}: {str(e)}",
            "> Chain failed to start."
        ]
        
        # 即使初始化失败，也记录回调捕获的任何内容
        if callback.execution_log:
            error_log.extend(callback.execution_log)
        
        with open(f"{application_id}_ERROR.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(error_log))

        print(f"❌ {application_id} - 初始化失败")



def process_multiple_records():
    """
    Read the application_id from each record in MORTGAGE_DATA processed by parse_sample_data(),
    and conduct processing tests for each.
    """
    # Ensure data is loaded
    if not data_store.data:
        print("Warning: data_store is empty, loading data...")
        parse_sample_data()
    
    # Iterate through each data record
    for record in data_store.data:
        application_id = record.get('transaction_id')
        if not application_id:
            print(f"Warning: Skipping a record missing application_id: {record}")
            continue
            
        print(f"Processing application_id: {application_id}...")
        # Call your processing function with the specific application_id
        process_one_record(application_id)

# 调用函数开始处理
process_multiple_records()


# if __name__ == "__main__":
#     main()




