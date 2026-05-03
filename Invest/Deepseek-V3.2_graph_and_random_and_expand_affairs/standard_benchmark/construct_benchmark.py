import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():
    # === 输入参数 ===
    system_prompt = '''I will provide you with a detailed SOP process, data schema, data dataset, and defined function APIs. Based on these, you need to generate a piece of Python code that can process each entry in the dataset using the SOP API functions in a calling order that complies with both the actual situation of the data entry and the SOP workflow, and output the result of the last called API. You must execute each API exactly as specified in your defined sequence; modifying or simplifying the original API logic is strictly prohibited. It is acceptable for the output to be lengthy. Your generated code must be able to process all data in the same format as the <schema></schema> and example <data></data>. 
    If an error occurs during the standard execution process for a particular data entry, skip processing the standard result for that data entry and proceed to the next one. The standard result processing code you provide must ultimately yield the two critical fields verification_result and processing_efficiency_rating, along with the final results.
    Your final standard output should have each ID as a separate block, followed by each output field name and its value. For example:
    application_id:MLN-2025-049634
    prequalification_decision:Not Qualified
    max_loan_amount_prequalified:139200.0
    verification_result:xxx
    processing_efficiency_rating:xxx
    ...

    application_id:MLN-2025-049852
    prequalification_decision:Conditionally Qualified
    max_loan_amount_prequalified:1042000.0
    verification_result:xxx
    processing_efficiency_rating:xxx
    ...  
    '''

    # 初始化客户端
    client = OpenAI(
        api_key="",  # 替换为你的API Key
        base_url="https://"  # API中转地址
    )
    # === 读取YAML并填充 ===
    user_prompt = '''
        I will provide you with a detailed SOP process, data schema, data dataset, and defined function APIs. Based on these, you need to generate a piece of Python code that can process each entry in the dataset using the SOP API functions in a calling order that complies with both the actual situation of the data entry and the SOP workflow, and output the result of the last called API. The standard result processing code you provide must ultimately yield the two critical fields verification_result and processing_efficiency_rating, along with the final results.
        ***The SOP process is as follows***:
        <SOP>



        </SOP>

        ***The data schema is as follows***:

        <schema>

        </schema>

        ***The data dataset is as follows***:

        <data>

        </data>

        ***The function APIs are as follows***:



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
        '''


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
    with open("standard_response_output.txt", "w", encoding="utf-8") as file:
        file.write(full_response)

    print("完整回复已保存到 standard_response_output.txt 文件")

if __name__ == "__main__":
    main()
