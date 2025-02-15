import json
import time
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import torch

def get_time_series_data(key):
    """
    读取时间序列数据。

    参数:
    key (str): 要提取的时间序列数据的列名，可传入的列名示例有 'Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
               'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
               'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
               'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
               'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'CO2 (ppm)' 等。

    返回:
    numpy.ndarray: 包含指定列的时间序列数据的数组
    """
    # 读取Parquet文件
    table = pq.read_table('/home/zzx/projects/rrg-timsbc/zzx/Weather-Captioned/time_series/weather_large.parquet')
    df = table.to_pandas()
    df = df[:1500]
    # print(df[:5])
    # 根据传入的 key 提取相应列的数据
    time_series_data = df[key].values
    # 获取时间戳列表
    timestamps = df['Date Time'].tolist()
    
    # 获取所有时间戳对应的新闻文本列表
    news_texts_list = []
    for timestamp in timestamps:
        # 将 parquet_time 转换为 datetime 对象
        parquet_datetime = datetime.strptime(timestamp, "%d.%m.%Y %H:%M:%S")
        # 将 datetime 对象转换为 get_news_text_by_timestamp 所需的格式
        converted_time = parquet_datetime.strftime("%Y%m%d%H%M")
        # print(converted_time)
        news_texts = get_news_text_by_timestamp(converted_time)
        news_texts_list.append(news_texts)
    return time_series_data, news_texts_list


# 定义一个函数，根据时间戳获取对应的新闻文本列表
def get_news_text_by_timestamp(timestamp):
    """
    根据时间戳获取对应的新闻文本列表。

    参数:
    timestamp (str): 时间戳

    返回:
    list: 包含新闻文本的列表，如果没有找到对应的时间戳或哈希键，则返回空列表
    """
        # 读取 hash2text 哈希表文件
    try:
        with open('/home/zzx/projects/rrg-timsbc/zzx/Weather-Captioned/hash2text/hashtable_large.json', 'r') as hash2text_file:
            hash2text = json.load(hash2text_file)
    except FileNotFoundError:
        print("hash2text/hashtable_large.json 文件未找到，请检查文件路径。")
        hash2text = {}

    # 读取 date2hash 哈希表文件
    try:
        with open('/home/zzx/projects/rrg-timsbc/zzx/Weather-Captioned/date2hash/wm_messages_large_v1.json', 'r') as date2hash_file:
            date2hash = json.load(date2hash_file)
    except FileNotFoundError:
        print("date2hash/wm_messages_large_v1.json 文件未找到，请检查文件路径。")
        date2hash = {}
    # 首先从 date2hash 中获取对应的哈希键列表
    hash_keys = date2hash.get(timestamp, [])
    # print(hash_keys)
    news_texts = []
    text2hash = {v: k for k, v in hash2text.items()}
    # 遍历哈希键列表，从 hash2text 中获取对应的新闻文本
    for hash_key in hash_keys:
        news_text = text2hash.get(hash_key, None)
        if news_text:
            news_texts.append(news_text)
    if not news_texts:
        return None
    return news_texts

# 示例使用
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print((device))
    # a=get_time_series_data('T (degC)')
    # print(a)
    # 假设一个时间戳
    example_timestamp = "201401010010"
    news_texts = get_news_text_by_timestamp(example_timestamp)
    if news_texts:
        print(f"时间戳 {example_timestamp} 对应的新闻文本列表:")
        for text in news_texts:
            print(text)
    else:
        print((news_texts == None))
        print(f"未找到时间戳 {example_timestamp} 对应的新闻文本。")