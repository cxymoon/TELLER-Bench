import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():

    # === Input Parameters ===
    system_prompt = "You are a domain-specific SOP context generator. Based on the given domain scenario and template, you must leverage your internal domain knowledge to generate rich, substantial, and authentic domain context. Note that the process you provide should only include the relevant work during business processing and should not include any subsequent business activities. The process you provide should ultimately have several deterministic outputs, preferably numerical values."

    # Initialize the client
    client = OpenAI(
        api_key="",  # Replace with your API Key
        base_url="https://"  # API gateway URL
    )
    # === Read YAML and Fill ===
    user_prompt = '''
        You are an industry scenario analysis expert.
        Using the **Banking Credit Card Issuance Scenario** as the background, output a complete, structured description of multiple business activities within this scenario.
        Please strictly follow the format below for your output:

        1. Business Overview
        - Scenario Name:
        - Industry:
        - Primary Purpose / Value:

        2. Business Types (List all relevant sub-businesses)
        - Sub-business A: Brief Description
        - Sub-business B: Brief Description
        - …

        3. Process Description (One process per sub-business)
        - Sub-business A Process:
            1) Step 1: Description
            2) Step 2: Description
            3) …
        - Sub-business B Process:
            1) Step 1: Description
            2) Step 2: Description
            3) …
        - …

        4. Key Roles and Responsibilities
        - Role 1 (e.g., Customer / Bank Teller / System Administrator): Responsibility Description
        - Role 2: Responsibility Description
        - …

        5. System and Tool Support
        - System / Tool A: Function Description
        - System / Tool B: Function Description
        - …

        6. Risks and Control Points
        - Risk Point 1: Description / Corresponding Control Measures
        - Risk Point 2: Description / Corresponding Control Measures
        - …

        7. Typical Metrics and Measurement Methods
        - Metric 1 (e.g., Conversion Rate, Customer Satisfaction, Processing Time): Description of how to measure
        - Metric 2: Description of how to measure
        - …

        8. Best Practices or Improvement Suggestions
        - Suggestion 1: Specific Description
        - Suggestion 2: Specific Description
        - …

        Please **complete filling in according to the above format**.

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
        model="gemini-3-pro-preview",  # 指定模型
        max_tokens=32768
    )

    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
