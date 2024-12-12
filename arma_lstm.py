
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from arch import arch_model

# 加载数据
file_path = 'D:\download_browser\project\output.xlsx'
data = pd.read_excel(file_path)

# 计算涨跌幅
data['Change_Rate'] = data['USD_CNY'].pct_change()
data['Change_Rate'].fillna(0, inplace=True)

# 填充缺失值
data['SHIBOR_Rate'].fillna(method='ffill', inplace=True)
data['SHIBOR_Rate'].fillna(method='bfill', inplace=True)
data['Dow_Jones_Index'].fillna(method='ffill', inplace=True)
data['Dow_Jones_Index'].fillna(method='bfill', inplace=True)
data['Gold_Futures'].fillna(method='ffill', inplace=True)
data['Gold_Futures'].fillna(method='bfill', inplace=True)

# 特征列
features = ['USD_CNY', 'Change_Rate', 'UST_10Y_Yield', 'China_10Y_Yield', 'Yield_Spread',
            'VIX_Index', 'SHIBOR_Rate', 'WTI_Crude_Oil', 'Dow_Jones_Index',
            'Gold_Futures', 'USD_Index', 'MSCI_Index']

# 归一化
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 构造特征和目标变量
X = data[features].values
y = data['USD_CNY'].values

# 按时间顺序分割数据集（90%训练集，10%测试集）
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构造模型所需的三维输入
time_steps = 7

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train, y_train, time_steps)
X_test, y_test = create_sequences(X_test, y_test, time_steps)

# 构建LSTM模型
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# 预测
y_pred = model.predict(X_test)

# 反归一化还原预测值和实际值
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, :, 1:].mean(axis=1)), axis=1))[:, 0]
y_pred_actual = scaler.inverse_transform(np.concatenate((y_pred, X_test[:, :, 1:].mean(axis=1)), axis=1))[:, 0]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual USD/CNY')
plt.plot(y_pred_actual, label='Predicted USD/CNY', linestyle='--')
plt.title('LSTM: Actual vs Predicted USD/CNY Exchange Rate')
plt.xlabel('Sample Index')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()

# 损失曲线
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import numpy as np

# 计算 MSE 和 RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

from sklearn.metrics import r2_score

# 计算 R²
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

model.save_weights('lstm_model_weights.weights.h5')

# ARMA-GARCH波动性预测
def arma_garch_prediction(data, steps=7):
    rescaled_change_rate = data['Change_Rate'] * 1000  # 对涨跌幅进行放大
    model = arch_model(rescaled_change_rate, vol='Garch', p=1, q=1, mean='AR', lags=5)
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=steps)
    predicted_volatility = forecasts.variance[-1:]  # 提取波动率预测
    return np.sqrt(predicted_volatility.values.flatten()) / 1000  # 缩放回原始范围

garch_volatility = arma_garch_prediction(data, steps=7)
print(f"GARCH模型预测的未来波动率: {garch_volatility}")

# 连续预测未来7天（改进版：平滑波动性调整）
last_sequence = X_test[-1]
future_predictions_combined = []

# 平滑波动权重系数
alpha = 0.5  # 控制波动性对预测的影响

for i in range(7):
    # LSTM预测
    next_lstm_pred = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
    
    # 平滑波动性调整
    direction = np.sign(np.mean(last_sequence[:, 0]))  # 根据历史均值确定调整方向
    next_pred = next_lstm_pred + alpha * garch_volatility[i] * direction
    
    # 限制波动范围
    next_pred = np.clip(next_pred, next_lstm_pred - 0.02, next_lstm_pred + 0.02)
    future_predictions_combined.append(next_pred)
    
    # 动态更新特征
    last_sequence = np.roll(last_sequence, -1, axis=0)
    last_sequence[-1, 0] = next_pred
    last_sequence[-1, 1:] = np.mean(last_sequence[:, 1:], axis=0)

# 反归一化预测结果
future_predictions_actual_combined = scaler.inverse_transform(
    np.concatenate((np.array(future_predictions_combined).reshape(-1, 1), 
                    np.tile(X_test[:, :, 1:].mean(axis=1).mean(axis=0), (7, 1))), axis=1)
)[:, 0]

# 输出预测结果
print("结合改进版ARMA-GARCH波动性和LSTM预测的未来7天USD/CNY汇率:")
for i, rate in enumerate(future_predictions_actual_combined, start=1):
    print(f"第{i}天: {rate:.4f}")
   
    # 输出预测结果变量
future_predictions_actual_combined = future_predictions_actual_combined  # 预测的未来7天汇率

model.save('lstm_model.h5')



