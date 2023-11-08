import sys

import openai

with open("chatgpt_input.txt", "r", encoding="utf-8") as file:
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
                "content": "You are a very talented Python engineer who has been working with 3DCG in Python for over 10 years",
            },
            {"role": "user", "content": input_str},
        ],
    )
    result = response.choices[0].message.content

    # 結果をファイルに保存する
    with open("chatgpt_output.txt", "w", encoding="utf-8") as file:
        file.write(str(result))

    print("■ FINISH ■")

except AttributeError as e:
    error_message = f"Error: {e}"
    print(error_message)
