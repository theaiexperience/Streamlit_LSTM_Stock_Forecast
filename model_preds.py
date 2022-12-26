import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf

# Data preparation
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# neural networks 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# Graphics library
import matplotlib.pyplot as plt
import pandas as pd 
"""# SETTINGS """
from datetime import date
from datetime import timedelta

# Window size or the sequence length, 7 (1 week)
N_STEPS = 7

# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]


# Stock ticker
def stock_sele(stock):
    today = date.today()
    date_now = tm.strftime('%Y-%m-%d')
    date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

    init_df = yf.get_data(stock, 
        start_date=date_3_years_back, 
        end_date=today + dt.timedelta(days=2), 
        interval='1d')
# init_df = stock_sele()    
# remove columns which our neural network will not use
    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    # create the column 'date' based on index column
    init_df['date'] = init_df.index

    scaler = MinMaxScaler()
    init_df["scaled_close"] = scaler.fit_transform(np.expand_dims(init_df["close"].values,axis=1))
    
    return init_df,scaler
# init_df["close"][:5],init_df["close"].shape

# np.expand_dims(init_df['close'].values, axis=1)[:5]

# def PrepareData(days):
#   df = init_df.copy()
#   df['future'] = df['scaled_close'].shift(-days)
#   last_sequence = np.array(df[['scaled_close']].tail(days))
#   df.dropna(inplace=True)
#   sequence_data = []
#   sequences = deque(maxlen=N_STEPS)

#   for entry, target in zip(df[['scaled_close'] + ['date']].values, df['future'].values):
#       sequences.append(entry)
#       if len(sequences) == N_STEPS:
#           sequence_data.append([np.array(sequences), target])

#   last_sequence = list([s[:len(['scaled_close'])] for s in sequences]) + list(last_sequence)
#   last_sequence = np.array(last_sequence).astype(np.float32)

#   # construct the X's and Y's
#   X, Y = [], []
#   for seq, target in sequence_data:
#       X.append(seq)
#       Y.append(target)

#   # convert to numpy arrays
#   X = np.array(X)
#   Y = np.array(Y)

#   return df, last_sequence, X, Y

# PrepareData(3)

# def GetTrainedModel(x_train, y_train):
#   model = Sequential()
#   model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['scaled_close']))))
#   model.add(Dropout(0.3))
#   model.add(LSTM(120, return_sequences=False))
#   model.add(Dropout(0.3))
#   model.add(Dense(20))
#   model.add(Dense(1))

#   BATCH_SIZE = 20
#   EPOCHS = 200

#   model.compile(loss='mean_squared_error', optimizer='adam')

#   model.fit(x_train, y_train,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             verbose=1)

#   model.summary()

#   return model

# # model.save("stock_model.h5")
# def data_ready_rev():
#     # GET PREDICTIONS
#     predictions = []

#     for step in LOOKUP_STEPS:
#       df, last_sequence, x_train, y_train = PrepareData(step)
#       x_train = x_train[:, :, :len(['scaled_close'])].astype(np.float32)
      
#       # model = GetTrainedModel(tf.constant(x_train), tf.constant(y_train))
#       model = tf.keras.models.load_model("stock_model.h5")
#       last_sequence = last_sequence[-N_STEPS:]
#       last_sequence = np.expand_dims(last_sequence, axis=0)
#       prediction = model.predict(tf.constant(last_sequence))
#       predicted_price = scaler.inverse_transform(prediction)[0][0]

#       predictions.append(round(float(predicted_price), 2))
      
#     return predictions,model,x_train,y_train
# predictions,model,x_train,y_train = data_ready_rev()      

def next_3_pred(predictions,STOCK):
 if bool(predictions) == True and len(predictions) > 0:
      predictions_list = [str(d)+'$' for d in predictions]
      predictions_str = ', '.join(predictions_list)
      message = f'{STOCK} prediction for upcoming 3 days ({predictions_str})'
      return message
# next_3_pred()      
def user_df_view(model,x_train,y_train,init_df,scaler): 
    copy_df = init_df.copy()
    y_predicted = model.predict(x_train)
    y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
    first_seq = scaler.inverse_transform(np.expand_dims(y_train[:8], axis=1))
    last_seq = scaler.inverse_transform(np.expand_dims(y_train[-4:], axis=1))
    y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
    y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
    copy_df[f'predicted_close'] = y_predicted_transformed
    return copy_df
# copy_df = user_df_view()    
# date_now = dt.date.today()
# date_tomorrow = dt.date.today() + dt.timedelta(days=1)
# date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)
# copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0, 0]
# copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0, 0]
# copy_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0, 0]

# Result chart
# plt.style.use(style='ggplot')
# plt.figure(figsize=(16,10))
# plt.plot(copy_df['close'][-150:].head(147))
# plt.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
# plt.plot(copy_df['close'][-150:].tail(4))
# plt.xlabel('days')
# plt.ylabel('price')
# plt.legend([f'Actual price for {STOCK}', 
#             f'Predicted price for {STOCK}',
#             f'Predicted price for future 3 days'])
# plt.save("predictions_3_days")

