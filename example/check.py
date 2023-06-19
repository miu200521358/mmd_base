import glob
import os

import numpy as np

# ffmpeg  -f image2 -r 30 -i %08d.png sizing.mp4

folder_path = "D:/MMD/MikuMikuDance_v926x64/UserFile/Motion/ダンス_1人/好き雪本気マジック_モーション hino/capture/"  # 対象のフォルダのパス
png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))  # フォルダ内のPNGファイルのリストを取得し、ソートする

num_files = len(png_files)  # ファイルの数を取得する
total_times: list[float] = []

for i in range(num_files):
    file_path = png_files[i]
    file_time = os.path.getmtime(file_path)  # ファイルの最終更新時間を取得する
    total_times.append(file_time)

np_total_times = np.array(total_times)
print(np.diff(np_total_times)[:3])
# avg_time_per_file = () / (num_files - 1)  # 最初のファイルは生成された時刻が取得されるため、除外する

print(f"平均一枚あたりの時間は{np.mean(np.diff(np_total_times)):.5f}秒です。")
