from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow_hub as hub  
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
import pickle
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
#from tensorflow import keras
#from keras import layers
#from keras import mixed_precision

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)
#importing both models using keras load_model
model1 = tf.keras.models.load_model('./nse_stock_prediction_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
model2 = tf.keras.models.load_model('./nasdaq_stock_prediction_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

#load pickled files
nse_p = open('nse_pickled.pickle', 'rb')
x_test_nya = pickle.load(nse_p)

ixic_p = open('ixic_pickled.pickle', 'rb')
x_test_gsp = pickle.load(ixic_p)

#defining graph plots
def plot_nya(i, j):
    print("Click and drag on the plot to zoom in, you can reset using the top right option")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = i.index, y = i.Close,
                        mode='lines',
                        name='Close',
                        marker_color = '#1F77B4'))
    fig.add_trace(go.Scatter(x = j.index, y = j.Close,
                        mode='lines',
                        name='Val',
                        marker_color = '#FF7F0E'))
    fig.add_trace(go.Scatter(x = j.index, y = j.Predictions,
                        mode='lines',
                        name='Predictions',
                        marker_color = '#2CA02C'))

    fig.update_layout(
        title=' Predictions for NSE Stock Exchange - INDIA',
        titlefont_size = 28,
        hovermode = 'x',
        xaxis = dict(
            title='Date',
            titlefont_size=16,
            tickfont_size=14),
        
        height = 800,
        
        yaxis=dict(
            title='Close price in Rupees (₹)',
            titlefont_size=16,
            tickfont_size=14),
        legend=dict(
            y=0,
            x=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'))

    #graph = fig.show()
    #return graph
    graph = fig.show()
    return graph

def plot_gsp(i, j):
    print("Click and drag on the plot to zoom in, you can reset using the top right option")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = i.index, y = i.Close,
                        mode='lines',
                        name='Close',
                        marker_color = '#1F77B4'))
    fig.add_trace(go.Scatter(x = j.index, y = j.Close,
                        mode='lines',
                        name='Val',
                        marker_color = '#FF7F0E'))
    fig.add_trace(go.Scatter(x = j.index, y = j.Predictions,
                        mode='lines',
                        name='Predictions',
                        marker_color = '#2CA02C'))

    fig.update_layout(
        title=' Predictions for Nasdaq Stock Exchange',
        titlefont_size = 28,
        hovermode = 'x',
        xaxis = dict(
            title='Date',
            titlefont_size=16,
            tickfont_size=14),
        
        height = 800,
        
        yaxis=dict(
            title='Close price in USD (USD$)',
            titlefont_size=16,
            tickfont_size=14),
        legend=dict(
            y=0,
            x=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'))

    graph = fig.show()
    return graph


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        index_1 = request.form["index"]
        data = pd.read_csv(r'indexData.csv')
        data = data.set_index('Date')
        index = data.groupby(data.Index)
        if index_1 in ['nse', 'nsei', 'NSE', 'NSEI']:
            NYA = index.get_group("NSEI")
            data_nya = NYA.filter(['Close'])
            dataset_nya = data_nya.values
            training_data_nya_len = math.ceil(len(dataset_nya) * .8)
            scaler_nya = MinMaxScaler(feature_range=(0,1))
            scaled_data_nya = scaler_nya.fit_transform(dataset_nya)

            prediction = model1.predict(x_test_nya)   #prediction matches to predictions_nya in ipynb
            prediction = scaler_nya.inverse_transform(prediction)

            train_nya = data_nya[:training_data_nya_len]
            valid_nya = data_nya[training_data_nya_len:-7]
            # valid_nya['Predictions'] = np.zeros((669, 1));
            # valid_nya['Predictions'][670:] = np.reshape(prediction, ())
            valid_nya['Predictions'] = prediction
            #print(plot_nya(train_nya, valid_nya))
            return render_template('index.html', prediction_text=f'This is the predicted chart of {index_1}', plot_graph=plot_nya(train_nya, valid_nya))

        elif index_1 in ['ixic', 'IXIC']:
            GSPTSE = index.get_group("IXIC") 
            data_gsp = GSPTSE.filter(['Close'])
            dataset_gsp = data_gsp.values
            training_data_gsp_len = math.ceil(len(dataset_gsp) * .8)
            scaler_gsp = MinMaxScaler(feature_range=(0,1))
            scaled_data_gsp = scaler_gsp.fit_transform(dataset_gsp)

            prediction = model2.predict(x_test_gsp)
            prediction = scaler_gsp.inverse_transform(prediction)

            train_gsp = data_gsp[:training_data_gsp_len]
            valid_gsp = data_gsp[training_data_gsp_len:]
            valid_gsp['Predictions'] = prediction

            #plot_gsp(train_gsp, valid_gsp)
            return render_template('index.html', prediction_text=f'This is the predicted chart of {index_1}', plot_graph=plot_gsp(train_gsp, valid_gsp))
    
if __name__ == "__main__":
    app.run(debug=True)