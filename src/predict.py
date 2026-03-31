"""
author:CBJ
Number of Prediction Samples: 30个
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import FWIMSNet
from src.preprocess_data import load_preprocessed_data
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict(n_samples=30):

    logger.info("=" * 60)
    logger.info(f"预测模式 - FWIMSNet- 预测{n_samples}个样本")
    logger.info("=" * 60)

    set_seed(42)

    # load_data
    X_short, X_mid, X_long, y, scalers = load_preprocessed_data()

    if X_short is None:
        logger.error("预处理数据不存在，请先运行: python src/preprocess_data.py")
        return

    logger.info(f"加载数据完成:")
    logger.info(f"  X_short形状: {X_short.shape}")
    logger.info(f"  X_mid形状: {X_mid.shape}")
    logger.info(f"  X_long形状: {X_long.shape}")
    logger.info(f"  y形状: {y.shape}")
    logger.info(f"  总样本数: {len(X_short)}")

    # Obtain Feature Dimensions
    short_term_dim = X_short.shape[2]
    mid_term_dim = X_mid.shape[2]
    long_term_dim = X_long.shape[2]

    logger.info(f"特征维度: 短期={short_term_dim}, 中期={mid_term_dim}, 长期={long_term_dim}")

    # Split Training Set and Test Set
    split = int(0.8 * len(X_short))
    X_short_test = X_short[split:]
    X_mid_test = X_mid[split:]
    X_long_test = X_long[split:]
    y_test = y[split:]

    logger.info(f"测试集总样本数: {len(X_short_test)}")

    # Take the first n_samples for prediction
    n = min(n_samples, len(X_short_test))
    X_short_test = X_short_test[:n]
    X_mid_test = X_mid_test[:n]
    X_long_test = X_long_test[:n]
    y_test = y_test[:n]

    logger.info(f"实际预测样本数: {len(X_short_test)}")

    # FWIMSNet
    model = FWIMSNet(
        short_term_dim=short_term_dim,
        mid_term_dim=mid_term_dim,
        long_term_dim=long_term_dim,
        kernel_sizes=[7, 30, 90],
        cnn_out_channels=32,
        gru_hidden_dim=128,
        transformer_nhead=8,
        transformer_layers=2,
        dropout=0.1
    )

    # Load the trained model
    model_path = PROJECT_ROOT / 'results' / 'FWIMSNet_model.pth'
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        logger.info("请先运行训练: python src/train.py")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prediction
    X_short_test_t = torch.FloatTensor(X_short_test)
    X_mid_test_t = torch.FloatTensor(X_mid_test)
    X_long_test_t = torch.FloatTensor(X_long_test)

    with torch.no_grad():
        y_pred = model(X_short_test_t, X_mid_test_t, X_long_test_t).numpy()

    print("\n" + "=" * 60)
    print(f"📊 预测结果（前{n}个样本）")
    print("=" * 60)
    print(f"{'序号':<6} {'真实值':<12} {'预测值':<12} {'绝对误差':<12} {'相对误差(%)':<12}")
    print("-" * 65)

    errors = []
    rel_errors = []
    for i in range(len(y_pred)):
        true_val = y_test[i][0]
        pred_val = y_pred[i][0]
        error = abs(true_val - pred_val)
        errors.append(error)

        # Compute the Error
        if abs(true_val) > 1e-8:
            rel_error = error / abs(true_val) * 100
        else:
            rel_error = 0
        rel_errors.append(rel_error)

        print(f"{i + 1:<6} {true_val:<12.4f} {pred_val:<12.4f} {error:<12.4f} {rel_error:<12.2f}")

    print(f"\n平均绝对误差: {np.mean(errors):.4f}")
    print(f"平均相对误差: {np.mean(rel_errors):.2f}%")
    print(f"最大绝对误差: {np.max(errors):.4f}")
    print(f"最小绝对误差: {np.min(errors):.4f}")

    # Compute Aggregate Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    y_test_nonzero = y_test_flat.copy()
    y_test_nonzero[y_test_nonzero == 0] = 1e-8
    mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_nonzero)) * 100

    print("\n" + "=" * 60)
    print("📊 总体评估结果")
    print("=" * 60)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Save Prediction Results
    results_df = pd.DataFrame({
        'index': range(1, len(y_pred) + 1),
        'true_value': y_test.flatten(),
        'predicted_value': y_pred.flatten(),
        'absolute_error': np.abs(y_test.flatten() - y_pred.flatten()),
        'relative_error_percent': rel_errors
    })
    save_path = PROJECT_ROOT / 'results' / 'predictions.csv'
    results_df.to_csv(save_path, index=False)
    logger.info(f"预测结果已保存: {save_path}")

    # 显示预测结果统计
    print("\n" + "=" * 60)
    print("📊 预测结果统计")
    print("=" * 60)
    print(f"  预测样本数: {len(y_pred)}")
    print(f"  真实值范围: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"  预测值范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  误差范围: [{np.min(errors):.4f}, {np.max(errors):.4f}]")

    # 显示误差分布
    error_bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    error_labels = ['0-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '0.20-0.30', '0.30-0.50', '>0.50']
    error_counts = np.histogram(errors, bins=error_bins)[0]

    print("\n  误差分布:")
    for label, count in zip(error_labels, error_counts):
        pct = count / len(errors) * 100
        print(f"    {label}: {count} 个样本 ({pct:.1f}%)")

    print("\n✅ 预测完成！")


if __name__ == "__main__":
    import numpy as np

    predict(n_samples=30)