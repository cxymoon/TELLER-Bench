import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():
    # === 输入参数 ===
    system_prompt = "You are an SOP dataset expansion generator. You need to generate 50 new data entries that are completely identical in format to the given data samples, based on the provided SOP, data schema, and dataset examples. New data IDs should start incrementing from the maximum value in the provided examples to avoid duplication. ***Important: Fields wrapped in <Field_to_be_obtained></Field_to_be_obtained> in the schema should not have values; their values are obtained through subsequent processing. They should remain empty during synthesis, consistent with the examples in <data_samples></data_samples>.***"
    # === 读取YAML并填充 ===
    user_prompt = '''
        I will provide you with a detailed SOP process, a data schema, and a data dataset. Based on these, you need to generate 50 new pieces of data that are completely identical in format to the data dataset examples, enclosed within <data></data>.
        ***Important: Fields wrapped in <Field_to_be_obtained></Field_to_be_obtained> in the schema should not have values; their values are obtained through subsequent processing. They should remain empty during synthesis, consistent with the examples in <data_samples></data_samples>.***
        ***The SOP process is as follows***:
        <SOP>

        </SOP>
        
        ***The data schema is as follows***: 
        <schema>

        </schema>

        ***The data dataset examples are as follows***:   
        <data_samples>  

        </data_samples>
        '''

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
    with open("response_output.txt", "w", encoding="utf-8") as file:
        file.write(full_response)

    print("完整回复已保存到 response_output.txt 文件")

if __name__ == "__main__":
    main()
