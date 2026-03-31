#!/usr/bin/env python
"""
author:CBJ
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Add the project root directory to the system path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data():
    """Load Raw Data"""
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'demo_data.csv'

    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        return None

    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            logger.info(f"成功使用 {enc} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        logger.error("无法读取文件")
        return None

    # Ensure the date column format is correct
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")

    return df


def split_features_by_physics(df):

    # Grouping of 21 Features
    short_term_features = [
        'ISI', 'FFMC', '温度变化率', '温度距平', '月尺度_地表温度(°C)',
        '月尺度_净辐射(MJ/m2)', '月尺度_总辐射(MJ/m2)', '月尺度_紫外辐射(MJ/m2)',
        '月尺度_光合有效辐射(MJ/m2)', '月尺度_反射辐射(MJ/m2)'
    ]

    mid_term_features = [
        '树干径流_树干径流量', '蒸散量_该日土层储水量', 'DMC', 'BUI', '雨水水质_硫酸根'
    ]

    long_term_features = [
        'DC', '月尺度_5厘米土壤温度(°C)', '月尺度_10厘米土壤温度(°C)',
        '月尺度_60厘米土壤温度(°C)', '月尺度_气温(°C)', '月尺度_气压(hPa)'
    ]

    short_term = [f for f in short_term_features if f in df.columns]
    mid_term = [f for f in mid_term_features if f in df.columns]
    long_term = [f for f in long_term_features if f in df.columns]

    logger.info(f"短期特征({len(short_term)}): {short_term}")
    logger.info(f"中期特征({len(mid_term)}): {mid_term}")
    logger.info(f"长期特征({len(long_term)}): {long_term}")

    return short_term, mid_term, long_term


def prepare_sequences_augmented(df, short_term, mid_term, long_term, base_seq_length=6):
    """
    Enhanced Sequence Preparation: Combining Sliding Window, Multi-Scale, and Noise Augmentation

    """
    X_short = df[short_term].values
    X_mid = df[mid_term].values
    X_long = df[long_term].values
    y = df['FWI'].values

    # 1. Basic Sliding Window Sampling (Fixed Length base_seq_length)
    X_short_list, X_mid_list, X_long_list, y_list = [], [], [], []

    for i in range(0, len(df) - base_seq_length):
        X_short_list.append(X_short[i:i + base_seq_length])
        X_mid_list.append(X_mid[i:i + base_seq_length])
        X_long_list.append(X_long[i:i + base_seq_length])
        y_list.append(y[i + base_seq_length])

    X_short_base = np.array(X_short_list, dtype=np.float32)
    X_mid_base = np.array(X_mid_list, dtype=np.float32)
    X_long_base = np.array(X_long_list, dtype=np.float32)
    y_base = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    logger.info(f"基础滑动窗口样本数: {len(X_short_base)}")

    # 2. Multi-Scale Sampling - Padding to the Same Length
    seq_lengths = [3, 4, 5, 7, 8]  # Different Sequence Lengths
    X_short_multi, X_mid_multi, X_long_multi, y_multi = [], [], [], []

    for seq_len in seq_lengths:
        for i in range(0, len(df) - seq_len, 2):
            # Extract Sequences
            seq_short = X_short[i:i + seq_len]
            seq_mid = X_mid[i:i + seq_len]
            seq_long = X_long[i:i + seq_len]

            # Pad to base_seq_length
            if seq_len < base_seq_length:
                pad_len = base_seq_length - seq_len
                pad_short = np.zeros((pad_len, seq_short.shape[1]))
                pad_mid = np.zeros((pad_len, seq_mid.shape[1]))
                pad_long = np.zeros((pad_len, seq_long.shape[1]))

                seq_short = np.vstack([seq_short, pad_short])
                seq_mid = np.vstack([seq_mid, pad_mid])
                seq_long = np.vstack([seq_long, pad_long])
            elif seq_len > base_seq_length:
                # If the sequence is longer, take the last base_seq_length time steps
                seq_short = seq_short[-base_seq_length:]
                seq_mid = seq_mid[-base_seq_length:]
                seq_long = seq_long[-base_seq_length:]

            X_short_multi.append(seq_short)
            X_mid_multi.append(seq_mid)
            X_long_multi.append(seq_long)
            y_multi.append(y[i + seq_len])

    if X_short_multi:
        X_short_multi_arr = np.array(X_short_multi, dtype=np.float32)
        X_mid_multi_arr = np.array(X_mid_multi, dtype=np.float32)
        X_long_multi_arr = np.array(X_long_multi, dtype=np.float32)
        y_multi_arr = np.array(y_multi, dtype=np.float32).reshape(-1, 1)

        logger.info(f"多尺度采样样本数: {len(X_short_multi_arr)}")

        # Merge Base Samples and Multi-Scale Samples
        X_short_all = np.concatenate([X_short_base, X_short_multi_arr], axis=0)
        X_mid_all = np.concatenate([X_mid_base, X_mid_multi_arr], axis=0)
        X_long_all = np.concatenate([X_long_base, X_long_multi_arr], axis=0)
        y_all = np.concatenate([y_base, y_multi_arr], axis=0)
    else:
        X_short_all = X_short_base
        X_mid_all = X_mid_base
        X_long_all = X_long_base
        y_all = y_base

    logger.info(f"合并后总样本数: {len(X_short_all)}")

    # 3. Data Augmentation
    noise_levels = [0.01, 0.02]
    X_short_aug = [X_short_all]
    X_mid_aug = [X_mid_all]
    X_long_aug = [X_long_all]
    y_aug = [y_all]

    for noise in noise_levels:
        noise_short = np.random.normal(0, noise, X_short_all.shape)
        noise_mid = np.random.normal(0, noise, X_mid_all.shape)
        noise_long = np.random.normal(0, noise, X_long_all.shape)

        X_short_aug.append(X_short_all + noise_short)
        X_mid_aug.append(X_mid_all + noise_mid)
        X_long_aug.append(X_long_all + noise_long)
        y_aug.append(y_all)

    X_short_final = np.concatenate(X_short_aug, axis=0)
    X_mid_final = np.concatenate(X_mid_aug, axis=0)
    X_long_final = np.concatenate(X_long_aug, axis=0)
    y_final = np.concatenate(y_aug, axis=0)

    logger.info(f"数据增强后总样本数: {len(X_short_final)}")

    # Shuffle Data
    indices = np.random.permutation(len(X_short_final))
    X_short_final = X_short_final[indices]
    X_mid_final = X_mid_final[indices]
    X_long_final = X_long_final[indices]
    y_final = y_final[indices]

    return X_short_final, X_mid_final, X_long_final, y_final


def normalize_and_augment():
    """Main Function for Data Preprocessing: Standardization + Data Augmentation + Saving"""
    logger.info("=" * 60)
    logger.info("开始数据预处理")
    logger.info("=" * 60)

    # 1. Load Raw Data
    df = load_raw_data()
    if df is None:
        return

    # 2. Feature Grouping
    short_term, mid_term, long_term = split_features_by_physics(df)

    # All Feature Columns
    all_features = short_term + mid_term + long_term

    # Check if FWI column exists
    if 'FWI' not in df.columns:
        logger.error("数据中没有FWI列")
        return

    logger.info(f"特征数量: {len(all_features)}")

    # 3. Create Scaler
    scaler_short = StandardScaler()
    scaler_mid = StandardScaler()
    scaler_long = StandardScaler()
    scaler_y = StandardScaler()

    # Extract Data
    X_short = df[short_term].values
    X_mid = df[mid_term].values
    X_long = df[long_term].values
    y = df['FWI'].values.reshape(-1, 1)

    # 4. Standardization
    X_short_scaled = scaler_short.fit_transform(X_short)
    X_mid_scaled = scaler_mid.fit_transform(X_mid)
    X_long_scaled = scaler_long.fit_transform(X_long)
    y_scaled = scaler_y.fit_transform(y)

    # 5. Create Standardized DataFrame
    df_scaled = df.copy()

    # Replace with Standardized Feature Values
    for i, col in enumerate(short_term):
        df_scaled[col] = X_short_scaled[:, i]

    for i, col in enumerate(mid_term):
        df_scaled[col] = X_mid_scaled[:, i]

    for i, col in enumerate(long_term):
        df_scaled[col] = X_long_scaled[:, i]

    df_scaled['FWI'] = y_scaled.flatten()

    # 6. Save Standardized Data
    normalized_path = PROJECT_ROOT / 'data' / 'processed' / 'demo_data_normalized.csv'
    df_scaled.to_csv(normalized_path, index=False, encoding='utf-8-sig')
    logger.info(f"标准化数据已保存: {normalized_path}")

    # 7. Save Scaler
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler_short, results_dir / 'scaler_short.pkl')
    joblib.dump(scaler_mid, results_dir / 'scaler_mid.pkl')
    joblib.dump(scaler_long, results_dir / 'scaler_long.pkl')
    joblib.dump(scaler_y, results_dir / 'scaler_y.pkl')
    logger.info(f"标准化器已保存: {results_dir}")

    # 8. Print Standardized Statistics
    print("\n" + "=" * 60)
    print("📊 标准化后数据统计")
    print("=" * 60)
    print(f"数据形状: {df_scaled.shape}")
    print(f"特征均值: {df_scaled[all_features].mean().mean():.6f}")
    print(f"特征标准差: {df_scaled[all_features].std().mean():.6f}")
    print(f"FWI均值: {df_scaled['FWI'].mean():.6f}")
    print(f"FWI标准差: {df_scaled['FWI'].std():.6f}")

    print("\n前5行标准化数据:")
    print(df_scaled[['date'] + all_features[:3] + ['FWI']].head())

    # 9. Data Augmentation: Generate Augmented Sequences
    logger.info("\n" + "=" * 60)
    logger.info("开始数据增强")
    logger.info("=" * 60)

    base_seq_length = 6
    X_short_aug, X_mid_aug, X_long_aug, y_aug = prepare_sequences_augmented(
        df_scaled, short_term, mid_term, long_term, base_seq_length=base_seq_length
    )

    # 10. Save Augmented Sequence Data
    np.save(results_dir / 'X_short_aug.npy', X_short_aug)
    np.save(results_dir / 'X_mid_aug.npy', X_mid_aug)
    np.save(results_dir / 'X_long_aug.npy', X_long_aug)
    np.save(results_dir / 'y_aug.npy', y_aug)

    logger.info(f"增强序列数据已保存: {results_dir}")
    logger.info(f"X_short_aug形状: {X_short_aug.shape}")
    logger.info(f"X_mid_aug形状: {X_mid_aug.shape}")
    logger.info(f"X_long_aug形状: {X_long_aug.shape}")
    logger.info(f"y_aug形状: {y_aug.shape}")

    # 11. Print Augmented Statistics
    print("\n" + "=" * 60)
    print("📊 数据增强后统计")
    print("=" * 60)
    print(f"增强后样本数: {len(X_short_aug)}")
    print(f"序列长度: {base_seq_length}")
    print(f"短期特征维度: {X_short_aug.shape[2]}")
    print(f"中期特征维度: {X_mid_aug.shape[2]}")
    print(f"长期特征维度: {X_long_aug.shape[2]}")
    print(f"目标值范围: [{y_aug.min():.4f}, {y_aug.max():.4f}]")

    print("\n✅ 数据预处理完成！")
    print(f"   - 标准化数据: {normalized_path}")
    print(f"   - 增强序列数据: {results_dir}/*_aug.npy")
    print(f"   - 标准化器: {results_dir}/scaler_*.pkl")

    return df_scaled, (X_short_aug, X_mid_aug, X_long_aug, y_aug), (scaler_short, scaler_mid, scaler_long, scaler_y)


def load_preprocessed_data():
    """Load Preprocessed Data for Training"""
    results_dir = PROJECT_ROOT / 'results'

    # Check if Augmented Data Exists
    if (results_dir / 'X_short_aug.npy').exists():
        logger.info("加载预处理后的增强数据...")
        X_short = np.load(results_dir / 'X_short_aug.npy')
        X_mid = np.load(results_dir / 'X_mid_aug.npy')
        X_long = np.load(results_dir / 'X_long_aug.npy')
        y = np.load(results_dir / 'y_aug.npy')

        # Load Scaler
        scaler_short = joblib.load(results_dir / 'scaler_short.pkl')
        scaler_mid = joblib.load(results_dir / 'scaler_mid.pkl')
        scaler_long = joblib.load(results_dir / 'scaler_long.pkl')
        scaler_y = joblib.load(results_dir / 'scaler_y.pkl')

        logger.info(f"加载增强数据: X_short{X_short.shape}, X_mid{X_mid.shape}, y{y.shape}")

        return X_short, X_mid, X_long, y, (scaler_short, scaler_mid, scaler_long, scaler_y)
    else:
        logger.warning("预处理数据不存在，请先运行: python src/preprocess_data.py")
        return None, None, None, None, None


if __name__ == "__main__":
    normalize_and_augment()