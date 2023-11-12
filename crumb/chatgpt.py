import sys
from datetime import datetime

import openai

with open("../chatgpt/chatgpt_input.txt", "r", encoding="utf-8") as file:
    input_str = "\n".join(file.readlines())
    # print(input_str)

try:
    openai.api_key = sys.argv[1]
    print("■ START ■")

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                # あなたは特に3DCGに詳しい、非常に優秀かつ親切なPythonエンジニアで、私の頼れる先輩です。常にコードの修正箇所と修正内容を答えてくれます。あなたは私に日本語で話しかけてくれます。私はあなたにコードと質問を提示します。その際にコードにFIXMEとコメントされている箇所を重視してください。
                "content": "You are a very talented and kind Python engineer, especially familiar with 3DCG, and a reliable senior of mine. You always answer me where and what to fix in the code. You speak to me in Japanese. I present the code and questions to you. At that time, please focus on the part of the code that is commented FIXME.",
            },
            {"role": "user", "content": input_str},
        ],
    )
    result = response.choices[0].message.content

    # 結果をファイルに保存する
    with open(
        f"../chatgpt/chatgpt_output_{datetime.now():%Y%m%d_%H%M%S}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(result))

    print("■ FINISH ■")

except AttributeError as e:
    error_message = f"Error: {e}"
    print(error_message)
