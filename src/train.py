#!/usr/bin/env python
"""
author:CBJ
"""

import sys
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import FWIMSNet, count_parameters
from src.preprocess_data import load_preprocessed_data
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train():
    """Main Training Function"""
    logger.info("=" * 60)
    logger.info("开始训练FWIMSNet模型")
    logger.info("=" * 60)

    set_seed(42)

    # Load Preprocessed Augmented Data
    X_short, X_mid, X_long, y, scalers = load_preprocessed_data()

    if X_short is None:
        logger.error("预处理数据不存在，请先运行: python src/preprocess_data.py")
        return

    logger.info(f"加载增强数据完成:")
    logger.info(f"  X_short形状: {X_short.shape}")
    logger.info(f"  X_mid形状: {X_mid.shape}")
    logger.info(f"  X_long形状: {X_long.shape}")
    logger.info(f"  y形状: {y.shape}")
    logger.info(f"  总样本数: {len(X_short)}")

    # Split Training Set and Test Set (80% Training, 20% Testing)
    split = int(0.8 * len(X_short))
    X_short_train, X_short_test = X_short[:split], X_short[split:]
    X_mid_train, X_mid_test = X_mid[:split], X_mid[split:]
    X_long_train, X_long_test = X_long[:split], X_long[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info(f"训练集: {X_short_train.shape}, 测试集: {X_short_test.shape}")
    logger.info(f"训练样本数: {len(X_short_train)}, 测试样本数: {len(X_short_test)}")

    # Get Feature Dimensions
    short_term_dim = X_short.shape[2]
    mid_term_dim = X_mid.shape[2]
    long_term_dim = X_long.shape[2]

    logger.info(f"特征维度: 短期={short_term_dim}, 中期={mid_term_dim}, 长期={long_term_dim}")

    # Create FWIMSNet
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

    logger.info(f"模型参数量: {count_parameters(model):,}")

    #Tensor
    X_short_train_t = torch.FloatTensor(X_short_train)
    X_mid_train_t = torch.FloatTensor(X_mid_train)
    X_long_train_t = torch.FloatTensor(X_long_train)
    y_train_t = torch.FloatTensor(y_train)

    X_short_test_t = torch.FloatTensor(X_short_test)
    X_mid_test_t = torch.FloatTensor(X_mid_test)
    X_long_test_t = torch.FloatTensor(X_long_test)
    y_test_t = torch.FloatTensor(y_test)

    # Data Loader
    train_dataset = TensorDataset(X_short_train_t, X_mid_train_t, X_long_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training Settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    # Training
    logger.info("开始训练...")
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 50

    train_losses = []
    val_losses = []

    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for batch_short, batch_mid, batch_long, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_short, batch_mid, batch_long)
            loss = criterion(output, batch_y)
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_short_test_t, X_mid_test_t, X_long_test_t)
            val_loss = criterion(val_output, y_test_t).item()
        val_losses.append(val_loss)

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            model_path = PROJECT_ROOT / 'results' / 'FWIMSNet_model.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Print Progress
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/200 | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best Loss: {best_loss:.4f} (Epoch {best_epoch}) | "
                f"LR: {current_lr:.6f}"
            )

        # Early Stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load Best Model
    model.load_state_dict(torch.load(PROJECT_ROOT / 'results' / 'FWIMSNet_model.pth'))
    model.eval()

    # Test Set Evaluation
    with torch.no_grad():
        y_pred = model(X_short_test_t, X_mid_test_t, X_long_test_t).numpy()

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
    print("📊 评估结果")
    print("=" * 60)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  最佳轮数: {best_epoch}")
    print(f"  最佳验证损失: {best_loss:.4f}")

    # Save Results
    results = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'best_loss': float(best_loss),
        'best_epoch': best_epoch,
        'train_samples': len(X_short_train),
        'test_samples': len(X_short_test),
        'total_samples': len(X_short),
        'short_term_dim': short_term_dim,
        'mid_term_dim': mid_term_dim,
        'long_term_dim': long_term_dim,
        'model_type': 'FWIMSNet'
    }
    with open(PROJECT_ROOT / 'results' / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save Loss History
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(PROJECT_ROOT / 'results' / 'loss_history.json', 'w') as f:
        json.dump(loss_history, f, indent=2)

    logger.info(f"模型已保存: {PROJECT_ROOT / 'results' / 'FWIMSNet_model.pth'}")
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    train()