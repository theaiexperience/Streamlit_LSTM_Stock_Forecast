import streamlit as st
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
import model_preds
import pandas as pd


def PrepareData(days,data):
  df = data.copy()
  df['future'] = df['scaled_close'].shift(-days)
  last_sequence = np.array(df[['scaled_close']].tail(days))
  df.dropna(inplace=True)
  sequence_data = []
  sequences = deque(maxlen=N_STEPS)

  for entry, target in zip(df[['scaled_close'] + ['date']].values, df['future'].values):
      sequences.append(entry)
      if len(sequences) == N_STEPS:
          sequence_data.append([np.array(sequences), target])

  last_sequence = list([s[:len(['scaled_close'])] for s in sequences]) + list(last_sequence)
  last_sequence = np.array(last_sequence).astype(np.float32)

  # construct the X's and Y's
  X, Y = [], []
  for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

  # convert to numpy arrays
  X = np.array(X)
  Y = np.array(Y)

  return df, last_sequence, X, Y

def data_ready_rev(df,scaler):
    # GET PREDICTIONS
    predictions = []

    for step in LOOKUP_STEPS:
      df, last_sequence, x_train, y_train = PrepareData(step,df)
      x_train = x_train[:, :, :len(['scaled_close'])].astype(np.float64)
      
      # model = GetTrainedModel(tf.constant(x_train), tf.constant(y_train))
      model = tf.keras.models.load_model("stock_model.h5")
      last_sequence = last_sequence[-N_STEPS:]
      last_sequence = np.expand_dims(last_sequence, axis=0)
      prediction = model.predict(tf.constant(last_sequence))
      predicted_price = scaler.inverse_transform(prediction)[0][0]

      predictions.append(round(float(predicted_price), 2))
      
    return predictions,model,x_train,y_train


N_STEPS = 7

LOOKUP_STEPS = [1, 2, 3]


st.write(""" # STOCK PRICE FORECAST
    ##### by CyberManithan""")
option = st.selectbox(
    'Select which company u want:',
    ('AAPL', 'TSLA', 'MSFT'))
st.write('You selected:', option)
STOCK = option
init_df,scaler_i = model_preds.stock_sele(STOCK)  
predictions,model,x_train,y_train = data_ready_rev(init_df,scaler_i)  
model_preds_3 = model_preds.next_3_pred(predictions,STOCK)
copy_df = model_preds.user_df_view(model=model,x_train=x_train,y_train=y_train,init_df=init_df,scaler=scaler_i)  

#Text

st.write(f"#### {model_preds_3}")
# copy_df['predicted_close']
#plot
plt.style.use(style='ggplot')
fig,ax = plt.subplots(figsize=(16,10))
ax.plot(copy_df['close'][-150:].head(147))
ax.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
ax.plot(copy_df['close'][-150:].tail(4))
ax.set_xlabel('days')
ax.set_ylabel('price')
ax.legend([f'Actual price for {STOCK}', 
                    f'Predicted price for {STOCK}',
                    f'Predicted price for future 3 days'])
st.pyplot(fig)
