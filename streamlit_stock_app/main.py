import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as web
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
st.set_option('deprecation.showPyplotGlobalUse', False)
st. set_page_config(layout="wide")
from gnews import GNews
import scipy.stats as stats

from prophet import Prophet, serialize
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import sys
import json


import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.sidebar.write("Select from below options")
side = st.sidebar.selectbox("Selcect one", ["Price Prediction", "Correlation Check", "Stock News", "Fbprophet"])

if side == "Price Prediction":

    mlflow.tensorflow.autolog()
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    st.title('Stock Price Prediction')
    company = st.text_input("Enter Stock/Index Ticker in Capitals")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    submit = st.button("Submit")

    if submit:
        # get stock data from yahoo finance
        data = web.DataReader(company, 'yahoo', start, end)
        #print(data.shape)
        st.dataframe(data.tail())
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        prediction_days = 100

        x_train = []
        y_train = []

        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Creating the ML model
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=15, batch_size=32)

        #Load test data
        test_start = end
        test_end = dt.datetime.now()
        test_data = web.DataReader(company, 'yahoo', test_start, test_end)

        original_price = test_data['Close'].values

        combined_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

        model_input = combined_dataset[len(combined_dataset) - len(test_data) - prediction_days:].values
        model_input = model_input.reshape(-1, 1)
        model_input = scaler.transform(model_input)

        #Prediction on test data
        x_test = []

        for i in range(prediction_days, len(model_input)):
            x_test.append(model_input[i-prediction_days:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        #Plot the predictions
        plt.figure(figsize=(18,8))
        plt.plot(original_price, color='black', label=f"Actual {company} price")
        plt.plot(predicted_prices, color='green', label=f"Predicted {company} price")
        plt.title(f"{company} share price")
        plt.xlabel('Time')
        plt.ylabel(f'{company} Share price')
        plt.legend()
        graph = plt.show()
        st.pyplot(graph)

        #prediction for next n days
        x_input = original_price[len(original_price)-100:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        lst_output = []
        n_steps = 100
        i = 0
        while(i<30):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
        
        #day_new=np.arange(1,101)
        #day_pred=np.arange(101,131)
        df3 = model_input.tolist()
        df3.extend(lst_output)
        df3 = scaler.inverse_transform(df3).tolist()
        plt.plot(df3)
        fig3 = plt.show()
        st.pyplot(fig3)


if side == "Correlation Check":

    st.title("Correlation Check")
    company = st.text_input("Stock Ticker in Capitals")
    index = st.text_input("Enter Index to correlate with")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    button = st.button("Submit")
    if button:
        
        data_stock = web.DataReader(company, 'yahoo', start, end)
        data_index = web.DataReader(index, 'yahoo', start, end)

        if data_stock.shape[0] > data_index.shape[0]:
            diff = data_stock.shape[0] - data_index.shape[0]
            data_stock = data_stock.iloc[:-diff]

        elif data_stock.shape[0] < data_index.shape[0]:
            diff = data_index.shape[0] - data_stock.shape[0]
            data_index = data_index.iloc[:-diff]

        c , p = stats.pearsonr(data_stock.dropna()['Close'], data_index.dropna()['Close'])
        output = "{} vs {} correlation is: {}".format(company, index, c)
        #st.write("{} vs {} Correlation is: ", c)
        st.write(output)


if side == "Stock News":
    st.title('Stock News')
    st.markdown("""---""")
    user_input = st.text_input("Enter Stock name")
    state = st.button("Get News!")
    if state:
        news = GNews().get_news(user_input)
        if news:
            for i in news:
                st.markdown(f"**{i['title']}**")
                st.write(f"Published Date - {i['published date']}")
                st.write(i["description"])
                st.markdown(f"[Article Link]({i['url']})")
                st.markdown("""---""")

        else:
            st.write("No news for this stock")


if side == "Fbprophet":

    st.title('Fbprophet')
    st.markdown("""---""")
    company = st.text_input("Enter Stock/Index Ticker in Capitals", value = 'TSLA')
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    period = st.number_input("Number of days want to predict", step=1, value = 365)
    submit = st.button("Submit")
    if submit:
        #get data from yahoo
        df = web.DataReader(company, 'yahoo', start, end)

        ARTIFACT_PATH = "model"
        np.random.seed(12345)

        def extract_params(pr_model):
            return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

        # data preprocessing
        df = df.reset_index()
        new_df = df[['Date', 'Close']]
        new_df = new_df.rename(columns={'Date':'ds', 'Close':'y'})

        # initialize prophet model with mlflow run

        with mlflow.start_run():

            fp = Prophet(daily_seasonality=True)
            fp.fit(new_df)
            params = extract_params(fp)

            metric_keys = ["mse", "rmse", "mae", "mape", "coverage"]
            metrics_raw = cross_validation(
                model=fp,
                horizon="365 days",
                period="180 days",
                initial="710 days",
                parallel="threads",
                disable_tqdm=True,
            )

            cv_metrics = performance_metrics(metrics_raw)
            metrics = {k: cv_metrics[k].mean() for k in metric_keys}

            print(f"Logged Metrics: \n{json.dumps(metrics, indent=2)}")
            print(f"Logged Params: \n{json.dumps(params, indent=2)}")
            
            mlflow.prophet.log_model(fp, artifact_path=ARTIFACT_PATH)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
            print(f"Model artifact logged to: {model_uri}")


        #make future predictions

        loaded_model = mlflow.prophet.load_model(model_uri)
        forecast = loaded_model.predict(loaded_model.make_future_dataframe(periods=period))

        # future = fp.make_future_dataframe(periods=period)
        # forecast = fp.predict(future)
        
        #Plot the predictions
        fig = plot_plotly(fp, forecast)
        fig.update_xaxes(title_text = 'Time')
        y_text = '{company_name} Stock price'.format(company_name=company)
        fig.update_yaxes(title_text = y_text)
        fig.update_layout(
            autosize=False,
            width=1500,
            height=800,)

        st.plotly_chart(fig)




