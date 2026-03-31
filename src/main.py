"""
author:CBJ
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_fwi_grade

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'demo_data.csv'

    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        return None

    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            logger.info(f"成功使用 {enc} 编码")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        logger.error("无法读取文件")
        return None

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def main():
    logger.info("=" * 60)
    logger.info("FWI-MSNet 数据统计")
    logger.info("=" * 60)

    df = load_data()
    if df is None:
        return

    if 'FWI' not in df.columns:
        logger.error("数据中没有FWI列")
        return

    print("\n" + "=" * 60)
    print("📊 数据概览")
    print("=" * 60)
    print(f"  数据行数: {len(df)}")
    print(f"  数据列数: {len(df.columns)}")
    if 'date' in df.columns:
        print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")

    print("\n" + "=" * 60)
    print("📈 FWI统计")
    print("=" * 60)
    print(f"  最小值: {df['FWI'].min():.2f}")
    print(f"  最大值: {df['FWI'].max():.2f}")
    print(f"  平均值: {df['FWI'].mean():.2f}")
    print(f"  中位数: {df['FWI'].median():.2f}")
    print(f"  标准差: {df['FWI'].std():.2f}")

    print("\n" + "=" * 60)
    print("🔥 FWI危险等级分布")
    print("=" * 60)
    df['grade'] = df['FWI'].apply(get_fwi_grade)
    grade_counts = df['grade'].value_counts()
    for grade in ['低危险', '中危险', '高危险', '很高危险', '极高危险']:
        if grade in grade_counts.index:
            count = grade_counts[grade]
            print(f"  {grade}: {count} 条 ({count / len(df) * 100:.1f}%)")

    # 按月统计
    if 'date' in df.columns:
        print("\n" + "=" * 60)
        print("📅 各月平均FWI")
        print("=" * 60)
        df['month'] = df['date'].dt.month
        monthly_avg = df.groupby('month')['FWI'].mean()
        for month in range(1, 13):
            if month in [6, 7, 8]:
                season = "夏"
            elif month in [3, 4, 5]:
                season = "春"
            elif month in [9, 10, 11]:
                season = "秋"
            else:
                season = "冬"
            print(f"  {month:2d}月({season}): {monthly_avg[month]:.2f}")

    print("\n" + "=" * 60)
    print("📋 前10行数据")
    print("=" * 60)
    print(df[['date' if 'date' in df.columns else 'index', 'FWI', 'grade']].head(10).to_string())

    print("\n" + "=" * 60)
    print("📋 后10行数据")
    print("=" * 60)
    print(df[['date' if 'date' in df.columns else 'index', 'FWI', 'grade']].tail(10).to_string())

    print("\n✅ 数据展示完成！")


if __name__ == "__main__":
    main()