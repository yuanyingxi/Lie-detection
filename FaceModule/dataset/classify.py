import os
import shutil
import pandas as pd

# 设置路径（假设所有内容在同一个父目录下）
video_dir = 'SEUMLD_Video'  # 需要分类的视频文件夹
csv_path = 'Coarse-grained-labels.csv'  # 视频名称与视频标签
deceptive_dir = 'Deceptive'
truthful_dir = 'Truthful'

# 读取标签文件（假设有两列：filename 和 label）
df = pd.read_csv(csv_path)

# 遍历标签并复制文件
for _, row in df.iterrows():
    filename = str(row['name']) + '.mp4'  # 使用列名访问，并转换为字符串
    label = int(row['label'])  # 确保label是整数

    src_path = os.path.join(video_dir, filename)

    if not os.path.exists(src_path):
        print(f'文件未找到: {filename}')
        continue

    if label == 1:
        dst_path = os.path.join(deceptive_dir, filename)
    else:
        dst_path = os.path.join(truthful_dir, filename)

    shutil.copy2(src_path, dst_path)  # 保留元数据复制文件
    print(f'已复制 {filename} 到 {"Deceptive" if label == 1 else "Truthful"} 文件夹')
