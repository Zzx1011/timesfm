import pandas as pd

# 读取CSV文件，假设文件名为input.csv，你需要根据实际文件名修改
df = pd.read_csv('/home/zzx/projects/rrg-timsbc/zzx/News dataset/Facebook_Obama_transpose_final.csv')

# 使用第一列作为unique_id
df['unique_id'] = df.iloc[:, 0]
# 使用第二列作为target_value（可修改为其他列）
df['target_value'] = df.iloc[:, 1]

# 生成时间戳序列，间隔20分钟，起始时间为0:00
start_time = pd.Timestamp('00:00:00')
time_intervals = pd.date_range(start=start_time, periods=len(df), freq='20T')
df['ds'] = time_intervals

# 选取unique_id和ds两列，并保存为新的CSV文件，文件名为output.csv，你可按需修改
result_df = df[['unique_id', 'ds','target_value']]
result_df.to_csv('/home/zzx/projects/rrg-timsbc/zzx/News dataset/Facebook_Obama_transpose_timesfm.csv', index=False)