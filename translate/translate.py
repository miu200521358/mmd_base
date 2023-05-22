import argparse
import re
import traceback

import deepl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--langs", type=str)
    parser.add_argument("--base_dir", type=str)
    args = parser.parse_args()

    re_break = re.compile(r"\n")
    re_end_break = re.compile(r"\\n$")
    re_message = re.compile(r'"(.*)"')

    langs: list[str] = args.langs.lower().split(",")

    messages = []
    with open(f"{args.base_dir}/messages.pot", mode="r", encoding="utf-8") as f:
        messages = f.readlines()

    translator = deepl.Translator(args.api_key)

    for lang in langs:
        file_path = f"{args.base_dir}/{lang}/LC_MESSAGES/messages.po"
        is_ja = "ja" == lang

        try:
            trans_messages = []
            with open(file_path, mode="r", encoding="utf-8") as f:
                trans_messages = f.readlines()

            # 翻訳のペアを辞書にする
            trans_messages_dict = dict([(message_id, message_str[:-1]) for (message_id, message_str) in zip(trans_messages[:-1], trans_messages[1:])])

            msg_id = None
            translated_message = None
            for i, org_msg in enumerate(messages):
                if i < 18:
                    # ヘッダはそのまま
                    continue

                if "msgid" in org_msg:
                    m = re_message.search(org_msg)
                    if m:
                        msg_id = m.group()
                        # 辞書に既にあればそれを採用する
                        translated_message = trans_messages_dict.get(org_msg, None)
                        continue

                if msg_id and "msgstr" in org_msg:
                    if translated_message and '""' not in translated_message:
                        # 既に翻訳済みの場合、記載されてる翻訳情報を転載
                        messages[i] = translated_message
                    else:
                        if is_ja:
                            messages[i] = f"msgstr {msg_id}"
                        else:
                            # 値がないメッセージを翻訳
                            trans_text = translator.translate_text(msg_id, source_lang="JA", target_lang=lang.upper())
                            translated_text = trans_text.text
                            if trans_text.text[-2:] in ['."', '。"']:
                                translated_text = trans_text.text[:-2] + '"'
                            if trans_text.text[-2:] in ['".', '"。']:
                                translated_text = trans_text.text[:-1]
                            messages[i] = f"msgstr {translated_text}"
                            print(f"翻訳: [{lang}][{msg_id}] -> [{translated_text}]")
                        msg_id = None
                        translated_message = None

            for i, message in enumerate(messages):
                message = re_break.sub("\\\\n", message)
                message = re_end_break.sub("\\n", message)
                messages[i] = message

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(messages)
        except Exception:
            print("*** Message Translate ERROR ***\n%s", traceback.format_exc())
