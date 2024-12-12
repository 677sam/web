import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from arch import arch_model

def prepare_data(file_path, features):
    """
    预处理数据：加载 Excel 文件，计算 Change_Rate 列，填充缺失值并归一化。
    """
    data = pd.read_excel(file_path)

    # 确保 USD_CNY 列存在
    if 'USD_CNY' not in data.columns:
        raise ValueError("Input file is missing the 'USD_CNY' column, which is required.")

    # 计算 Change_Rate 列
    if 'Change_Rate' not in data.columns:
        data['Change_Rate'] = data['USD_CNY'].pct_change().fillna(0)

    # 填充缺失值
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # 数据归一化
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    return data, scaler

def calculate_garch_volatility(data, steps=7):
    """
    计算 GARCH 模型的未来波动率。
    """
    rescaled_change_rate = data['Change_Rate'] * 1000  # 放大涨跌幅
    model = arch_model(rescaled_change_rate, vol='Garch', p=1, q=1, mean='AR', lags=5)
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=steps)
    predicted_volatility = forecasts.variance[-1:]
    return np.sqrt(predicted_volatility.values.flatten()) / 1000

def load_model_and_predict(data, scaler, features):
    """
    加载保存的模型并进行未来 7 天的预测。
    """
    custom_objects = {'mse': MeanSquaredError()}
    model = load_model('D:/download_browser/project/lstm_model.h5', custom_objects=custom_objects)

    # 确保输入数据与训练时的特征完全一致
    recent_data = data[features].values[-7:]
    recent_data = recent_data.reshape(1, 7, -1)

    # 计算 GARCH 波动率
    garch_volatility = calculate_garch_volatility(data)

    # 滚动预测
    predictions = []
    alpha = 0.5  # 平滑系数
    for i in range(7):
        prediction = model.predict(recent_data)[0, 0]

        # 波动性调整
        direction = np.sign(np.mean(recent_data[0, :, 0]))  # 根据历史均值确定方向
        prediction_adjusted = prediction + alpha * garch_volatility[i] * direction
        prediction_adjusted = np.clip(prediction_adjusted, prediction - 0.02, prediction + 0.02)

        predictions.append(prediction_adjusted)

        # 滚动窗口更新
        recent_data = np.roll(recent_data, -1, axis=1)
        recent_data[0, -1, 0] = prediction_adjusted

    # 反归一化还原预测值
    other_features_mean = data[features[1:]].mean().values
    predictions_2d = np.array(predictions).reshape(-1, 1)
    predictions_with_means = np.concatenate(
        (predictions_2d, np.tile(other_features_mean, (7, 1))), axis=1
    )
    predictions_actual = scaler.inverse_transform(predictions_with_means)[:, 0]

    return predictions_actual
