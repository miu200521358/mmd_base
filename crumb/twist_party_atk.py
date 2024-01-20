# # 入力ファイル
# - ツイステ：SSRカード一覧

# ## SSRカード一覧csvのフォーマットと内容
#  - カードキャラクター名 ... カードのキャラクター名
#  - カード種類 ... カードの種類（キャラクター名と種類でカードが一意に決まる）
#  - カード属性 ... カードの属性（アタック、ディフェンス、バランス）
#  - 魔法1属性 ... 魔法1の属性（火、水、木、無）
#  - 魔法2属性 ... 魔法2の属性（火、水、木、無）
#  - HP ... カードのHP
#  - ATK ... カードのATK
#  - デュオ魔法相手 ... デュオ魔法を発生させられるキャラクター名（カード種類は問わない）
#  - バディスキル1相手 ... バディスキル1を発生させられるキャラクター名（カード種類は問わない）
#  - バディスキル1 ... バディスキル1
#  - バディスキル2相手 ... バディスキル2を発生させられるキャラクター名（カード種類は問わない）
#  - バディスキル2 ... バディスキル2
#  - バディスキル3相手 ... バディスキル3を発生させられるキャラクター名（カード種類は問わない）
#  - バディスキル3 ... バディスキル3

# # 出力条件
# - 編成全体のATK合計値が27000以上であること
# - デュオ魔法が4組以上使えること
# - 編成全体でHP UP(中)が2個以上含まれていること
# - 編成全体でHP UP(小)が3個以上含まれていること

# # 編成ルール
# - 編成は5キャラクターで構成する
# - ひとつの編成に一人のキャラクターの複数種類のカードは含まない（キャラクターは重複させない）
# - バディスキル、デュオ魔法の相手は、キャラクター名は指定されているが、そのカード種類は不問である
# - バディスキル、デュオ魔法は編成に加えられているカード同士でのみ発生する

# # 編成リストソート順
# 1. デュオ魔法使用可能数降順
# 2. 編成全体のATK合計値降順
# 3. 編成全体のHP UP(中)個数降順
# 4. 編成全体のHP UP(小)個数降順

# # 出力フォーマット(ヘッダ ... 出力内容)
# 1. カード1キャラ ... 1枚目のカードキャラ
# 2. カード1種類 ... 1枚目のカード種類
# 3. カード2キャラ ... 2枚目のカードキャラ
# 4. カード2種類 ... 2枚目のカード種類
# 5. カード3キャラ ... 3枚目のカードキャラ
# 6. カード3種類 ... 3枚目のカード種類
# 7. カード4キャラ ... 4枚目のカードキャラ
# 8. カード1種類 ... 4枚目のカード種類
# 9. カード5キャラ ... 5枚目のカードキャラ
# 10. カード5種類 ... 5枚目のカード種類
# 11. ATK合計 ... 編成全体のATK合計値
# 12. デュオ数 ... 編成全体のデュオ魔法の組み合わせ数
# 12. デュオ1 ... 編成内で使えるデュオ魔法の組み合わせ1(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 13. デュオ2 ... 編成内で使えるデュオ魔法の組み合わせ2(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 14. デュオ3 ... 編成内で使えるデュオ魔法の組み合わせ3(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 15. デュオ4 ... 編成内で使えるデュオ魔法の組み合わせ4(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 16. デュオ5 ... 編成内で使えるデュオ魔法の組み合わせ5(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 17. デュオ6 ... 編成内で使えるデュオ魔法の組み合わせ6(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 18. デュオ7 ... 編成内で使えるデュオ魔法の組み合わせ7(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 19. デュオ8 ... 編成内で使えるデュオ魔法の組み合わせ8(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 20. デュオ9 ... 編成内で使えるデュオ魔法の組み合わせ9(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 21. HP UP(中)数 ... 編成全体のHP UP(中)の組み合わせ数
# 22. HP UP(中)1 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ1(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 23. HP UP(中)2 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ2(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 24. HP UP(中)3 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ3(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 25. HP UP(中)4 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ4(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 26. HP UP(中)5 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ5(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 27. HP UP(中)6 ... 編成内のHP UP(中)のバディスキルが生まれる組み合わせ6(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 28. HP UP(小)数 ... 編成全体のHP UP(小)の組み合わせ数
# 29. HP UP(小)1 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ1(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 30. HP UP(小)2 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ2(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 31. HP UP(小)3 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ3(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 32. HP UP(小)4 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ4(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 33. HP UP(小)5 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ5(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)
# 34. HP UP(小)6 ... 編成内のHP UP(小)のバディスキルが生まれる組み合わせ6(カードキャラクター名-カード種類&バディカードキャラクター名-カード種類。存在しない場合空欄)

# ------------------------------------------------------
import os
from argparse import ArgumentParser
from itertools import combinations, product
from winsound import SND_ALIAS, PlaySound

import pandas as pd
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--character", type=str)
parser.add_argument("--card_type", type=str)

args, argv = parser.parse_known_args()

main_card = {"カードキャラクター名": args.character, "カード種類": args.card_type}

# ファイルを読み込む
ssr_card_file_path = "ツイステ：SSRカード一覧.csv"
ssr_card_df = pd.read_csv(ssr_card_file_path)

party_file_path = "ツイステ：編成組み合わせ.csv"
party_df = pd.read_csv(party_file_path)

# # csvファイルの先頭数行を出力
# print(ssr_card_df.head())

# 出力用のDataFrameを初期化
output_columns = [
    "カード1",
    "カード2",
    "カード3",
    "カード4",
    "カード5",
    "HP合計",
    "ATK合計",
    "火属性数",
    "水属性数",
    "木属性数",
    "無属性数",
    "デュオ数",
    "HP UP(中)数",
    "HP UP(小)数",
    "ATK UP(中)数",
    "ATK UP(小)数",
    "デュオ1",
    "デュオ2",
    "デュオ3",
    "デュオ4",
    "デュオ5",
    "HP UP(中)1",
    "HP UP(中)2",
    "HP UP(中)3",
    "HP UP(中)4",
    "HP UP(中)5",
    "HP UP(中)6",
    "HP UP(小)1",
    "HP UP(小)2",
    "HP UP(小)3",
    "HP UP(小)4",
    "HP UP(小)5",
    "HP UP(小)6",
    "ATK UP(中)1",
    "ATK UP(中)2",
    "ATK UP(中)3",
    "ATK UP(中)4",
    "ATK UP(中)5",
    "ATK UP(中)6",
    "ATK UP(小)1",
    "ATK UP(小)2",
    "ATK UP(小)3",
    "ATK UP(小)4",
    "ATK UP(小)5",
    "ATK UP(小)6",
]
output_df = pd.DataFrame(columns=output_columns)


# 出力条件を満たすかチェックする関数
def check_conditions(cards):
    # キャラクター名が重複していないかチェック
    if len(set(cards["カードキャラクター名"])) != 5:
        return False

    # ATK合計のチェック
    if cards["ATK"].sum() < 27000:
        return False

    # 該当する編成におけるデュオ魔法、ATK UP(中)、ATK UP(小)の数を計算
    counts = {"デュオ魔法": 0, "ATK UP(中)": 0, "ATK UP(小)": 0}

    # デュオ魔法の数を計算
    counts["デュオ魔法"] = sum(cards["デュオ魔法相手"].isin(cards["カードキャラクター名"]))

    if counts["デュオ魔法"] < 4:
        return False

    # ATK UP(中)の数を計算
    # ATK UP(小)の数を計算
    for k in range(1, 4):
        buddy_skill = f"バディスキル{k}"
        buddy_skill_partner = f"バディスキル{k}相手"
        counts["ATK UP(中)"] += sum(
            (cards[buddy_skill] == "ATK UP(中)")
            & (cards[buddy_skill_partner].isin(cards["カードキャラクター名"]))
        )
        counts["ATK UP(小)"] += sum(
            (cards[buddy_skill] == "ATK UP(小)")
            & (cards[buddy_skill_partner].isin(cards["カードキャラクター名"]))
        )

    # 条件を満たしているかチェック
    if (counts["ATK UP(中)"] + (counts["ATK UP(小)"] / 2)) >= 3:
        return True

    return False


# 条件を満たしている組み合わせを出力用DataFrameに追加する関数
def add_to_output_df(selected_cards, output_df):
    # デュオ魔法、バディスキルの組み合わせを探索
    duos = []
    hp_up_medium = []
    hp_up_small = []
    atk_up_medium = []
    atk_up_small = []
    count_fire = 0
    count_water = 0
    count_green = 0
    count_none = 0

    for i in range(5):
        card_i = selected_cards.iloc[i]

        if card_i["魔法1属性"] == "火":
            count_fire += 1
        elif card_i["魔法1属性"] == "水":
            count_water += 1
        elif card_i["魔法1属性"] == "木":
            count_green += 1
        elif card_i["魔法1属性"] == "無":
            count_none += 1

        if card_i["魔法2属性"] == "火":
            count_fire += 1
        elif card_i["魔法2属性"] == "水":
            count_water += 1
        elif card_i["魔法2属性"] == "木":
            count_green += 1
        elif card_i["魔法2属性"] == "無":
            count_none += 1

        for j in range(5):
            if i == j:
                continue  # 同じカードの組み合わせはスキップ

            card_j = selected_cards.iloc[j]

            # デュオ魔法のチェック
            if card_i["デュオ魔法相手"] == card_j["カードキャラクター名"]:
                duos.append(
                    f"{card_i['カードキャラクター名']}-{card_i['カード種類']}&{card_j['カードキャラクター名']}-{card_j['カード種類']}"
                )

            # バディスキルのチェック
            for k in range(1, 4):
                buddy_skill = f"バディスキル{k}"
                buddy_skill_partner = f"バディスキル{k}相手"

                if card_i[buddy_skill_partner] == card_j["カードキャラクター名"]:
                    if card_i[buddy_skill] == "HP UP(中)":
                        hp_up_medium.append(
                            f"{card_i['カードキャラクター名']}-{card_i['カード種類']}&{card_j['カードキャラクター名']}-{card_j['カード種類']}"
                        )
                    elif card_i[buddy_skill] == "HP UP(小)":
                        hp_up_small.append(
                            f"{card_i['カードキャラクター名']}-{card_i['カード種類']}&{card_j['カードキャラクター名']}-{card_j['カード種類']}"
                        )
                    elif card_i[buddy_skill] == "ATK UP(中)":
                        atk_up_medium.append(
                            f"{card_i['カードキャラクター名']}-{card_i['カード種類']}&{card_j['カードキャラクター名']}-{card_j['カード種類']}"
                        )
                    elif card_i[buddy_skill] == "ATK UP(小)":
                        atk_up_small.append(
                            f"{card_i['カードキャラクター名']}-{card_i['カード種類']}&{card_j['カードキャラクター名']}-{card_j['カード種類']}"
                        )

    # 出力用データを作成
    output_data = []
    for card in selected_cards.itertuples():
        output_data.extend([f"{card.カードキャラクター名}-{card.カード種類}"])

    output_data.extend(
        [
            selected_cards["HP"].sum(),  # HP合計
            selected_cards["ATK"].sum(),  # ATK合計
            count_fire,  # 火属性数
            count_water,  # 水属性数
            count_green,  # 木属性数
            count_none,  # 無属性数
            len(duos),  # デュオ数
        ]
    )

    # デュオ魔法の組み合わせを追加（最大5組）
    for i in range(5):
        output_data.append(duos[i] if i < len(duos) else "")

    # HP UP(中)の組み合わせを追加（最大6組）
    output_data.append(len(hp_up_medium))
    for i in range(6):
        output_data.append(hp_up_medium[i] if i < len(hp_up_medium) else "")

    # HP UP(小)の組み合わせを追加（最大6組）
    output_data.append(len(hp_up_small))
    for i in range(6):
        output_data.append(hp_up_small[i] if i < len(hp_up_small) else "")

    # ATK UP(中)の組み合わせを追加（最大6組）
    output_data.append(len(atk_up_medium))
    for i in range(6):
        output_data.append(atk_up_medium[i] if i < len(atk_up_medium) else "")

    # ATK UP(小)の組み合わせを追加（最大6組）
    output_data.append(len(atk_up_small))
    for i in range(6):
        output_data.append(atk_up_small[i] if i < len(atk_up_small) else "")

    # 出力用DataFrameにデータを追加
    output_df.loc[len(output_df)] = output_data


main_card_info = ssr_card_df[
    (ssr_card_df["カードキャラクター名"] == main_card["カードキャラクター名"])
    & (ssr_card_df["カード種類"] == main_card["カード種類"])
]
print(main_card)

# メインカードの情報を取得
main_card_character = main_card["カードキャラクター名"]
main_card_type = main_card["カード種類"]
dir_name = "party_atk"

os.makedirs(dir_name, exist_ok=True)

# キャラ名一覧
character_names = [
    "リドル",
    "エース",
    "デュース",
    "トレイ",
    "ケイト",
    "レオナ",
    "ラギー",
    "ジャック",
    "アズール",
    "ジェイド",
    "フロイド",
    "カリム",
    "ジャミル",
    "ヴィル",
    "ルーク",
    "エペル",
    "イデア",
    "オルト",
    "マレウス",
    "リリア",
    "シルバー",
    "セベク",
]

# メインカードのキャラクター名以外のキャラ名を取得
other_character_names = [
    character_name
    for character_name in character_names
    if character_name != main_card_character
]

# othersから4名のキャラクターを選択する全組み合わせを取得
other_character_combinations = list(combinations(other_character_names, 4))

# others組み合わせを1件ずつ取り出して、4名のキャラクターのカード一覧を個別に取得
for n, other_characters in enumerate(tqdm(other_character_combinations)):
    card_combinations = [main_card_info.values.tolist()]
    for other_character in other_characters:
        # other_characterのカード一覧を取得
        other_character_cards = ssr_card_df[
            (ssr_card_df["カードキャラクター名"] == other_character)
        ]
        card_combinations.append(other_character_cards.values.tolist())

    # 5名のカード一覧を組み合わせる
    character_products = product(
        card_combinations[0],
        card_combinations[1],
        card_combinations[2],
        card_combinations[3],
        card_combinations[4],
    )

    for i, character_product in enumerate(character_products):
        cards = pd.DataFrame(
            character_product,
            columns=[
                "カードキャラクター名",
                "カード種類",
                "カード属性",
                "魔法1属性",
                "魔法2属性",
                "HP",
                "ATK",
                "デュオ魔法相手",
                "バディスキル1相手",
                "バディスキル1",
                "バディスキル2相手",
                "バディスキル2",
                "バディスキル3相手",
                "バディスキル3",
            ],
        )

        if check_conditions(cards):
            add_to_output_df(cards, output_df)

    # if n > 10:
    #     break

# Sort output_df based on priority order
output_df = output_df.sort_values(
    by=["デュオ数", "ATK合計", "ATK UP(中)数", "ATK UP(小)数"],
    ascending=[False, False, False, False],
)

# 追加した出力用DataFrameのデータをcsvに出力
output_df.to_csv(f"{dir_name}/{main_card_character}-{main_card_type}.csv", index=False)

PlaySound("SystemAsterisk", SND_ALIAS)


# # SSR一覧からメインカードの情報を抽出
# main_card_info = ssr_card_df[(ssr_card_df['カードキャラクター名'] == main_card_character) & (ssr_card_df['カード種類'] == main_card_type)]

# # SSR一覧からメインカード以外のカードを抽出
# other_cards_info = ssr_card_df[(ssr_card_df['カードキャラクター名'] != main_card_character)]

# card_combinations = []
# for i in range(len(other_cards_info)):
#     combination = [main_card_info.iloc[0]]  # メインカードを組み合わせの先頭に追加
#     combination.extend(other_cards_info.iloc[i:i+4].values.tolist())  # メインカード以外のカードを追加
#     characters = set()
#     duplicate = False
#     for card in combination:
#         character = card[0]  # カードキャラクター名
#         if character in characters:
#             duplicate = True
#             break
#         characters.add(character)
#     if not duplicate:
#         card_combinations.append(combination)


# # カードの組み合わせを生成
# card_combinations = []
# for i in range(len(other_cards_info)):
#     combination = [main_card_info.iloc[0]]  # メインカードを組み合わせの先頭に追加
#     combination.extend(other_cards_info.iloc[i:i+4].values.tolist())  # メインカード以外のカードを追加
#     card_combinations.append(combination)

# valid_combinations = []
# for combination in card_combinations:
#     characters = set()
#     duplicate = False
#     for card in combination:
#         character = card[0]  # カードキャラクター名
#         if character in characters:
#             duplicate = True
#             break
#         characters.add(character)
#     if not duplicate:
#         valid_combinations.append(combination)

# # CSVファイルに出力する
# output_file_path = 'valid_combinations.csv'
# output_df = pd.DataFrame(valid_combinations)
# output_df.to_csv(output_file_path, index=False)


# top_10_rows = ssr_card_df.head(10)

# # トップ10行に含まれるキャラクター名のリストを作成
# top_10_characters = set(top_10_rows['カードキャラクター名'])

# # 絞り込み
# for i in range(7, len(party_df)):
# # for i in range(0, 5):
#     print(f"{i} / {len(party_df)}")

#     party = party_df.loc[i]
#     selected_characters = [party['キャラ1'], party['キャラ2'], party['キャラ3'], party['キャラ4'], party['キャラ5']]
#     print("selected_characters: ", selected_characters)

#     if len(top_10_rows[top_10_rows['カードキャラクター名'].isin(selected_characters)]):
#         # トップ10に含まれるキャラクターが選択された場合、そのカード種類をトップ10のものに限定
#         filter_conditions = []
#         for character in selected_characters:
#             if character in top_10_characters:
#                 # トップ10に含まれるキャラクターのカード種類を限定
#                 top_10_card_types = top_10_rows[top_10_rows['カードキャラクター名'] == character]['カード種類']
#                 condition = (ssr_card_df['カードキャラクター名'] == character) & (ssr_card_df['カード種類'].isin(top_10_card_types))
#             else:
#                 # トップ10に含まれないキャラクターはすべてのカード種類を対象
#                 condition = ssr_card_df['カードキャラクター名'] == character

#             filter_conditions.append(condition)

#         # フィルタリング処理
#         df_filtered = ssr_card_df[filter_conditions[0] | filter_conditions[1] | filter_conditions[2] | filter_conditions[3] | filter_conditions[4]]

#         # 組み合わせを生成
#         combos = list(combinations(df_filtered.index, 5))

#         # 組み合わせの件数を表示
#         print("組み合わせ件数: ", len(combos))

#         for combo in tqdm(combos):
#             cards = ssr_card_df.loc[list(combo)]

#             if len(cards[(cards['カードキャラクター名'].isin(top_10_rows['カードキャラクター名'])) & (cards['カード種類'].isin(top_10_rows['カード種類']))]) and check_conditions(cards):
#                 add_to_output_df(cards, output_df)

#                 # 生成したDataFrameの件数を表示
#                 print("出力件数: ", len(output_df))

#                 if len(output_df) > 0:
#                     # 追加した出力用DataFrameのデータをcsvに出力
#                     output_df.to_csv(f'party/高火力編成_{"-".join(selected_characters)}.csv', index=False)
