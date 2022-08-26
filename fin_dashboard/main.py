# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import datetime as dt
# import pandas_datareader as web
# from prophet import Prophet
# from prophet.plot import plot_plotly
import streamlit as st
# import yfinance as yf
# import urllib

from stock_recomendation import analyst_recommendation
from plot_charts import get_stock_name, plot_chart, plot_chart_short
from sentiment import sentiment_analysis

# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Stock Analysis", page_icon=":chart_with_upwards_trend:", layout="wide"
)
from gnews import GNews

# import scipy.stats as stats
import requests
# import json

st.sidebar.write("Select from below options")
side = st.sidebar.selectbox(
    "Selcect one", ["Dashboard", "Foreign Markets", "Stock News"]
)

# To get location of the user
ip_req = requests.get("https://get.geojs.io/v1/ip.json")
ip_addr = ip_req.json()["ip"]
url = "https://get.geojs.io/v1/ip/geo/" + ip_addr + ".json"

geo_req = requests.get(url)
location = geo_req.json()["country"]  # stores location of the user

location_index = {
    "India": "^NSEI",
    "USA": "^DJI",
    "Germany": "^GDAXI",
    "UK": "^FTSE",
    "China": "^HSI",
    "Japan": "^N225",
}
top_stocks = {
    "^NSEI": ["INFY.NS", "RELIANCE.NS", "BAJFINANCE.NS", "TCS.NS", "SBIN.NS"],
    "^DJI": ["AAPL", "GOOG", "TSLA", "MSFT", "UNH"],
    "^GDAXI": ["AIR.DE", "VOW3.DE", "ADS.DE", "DTE.DE", "BMW.DE"],
    "^FTSE": ["CCH.L", "VOD.L", "RR.L", "AZN.L", "HSBA.L"],
    "^N225": ["6702.T", "6501.T", "6503.T", "7751.T", "6952.T"],
    "^HSI": ["0992.HK", "9988.HK", "1810.HK", "0386.HK", "1398.HK"],
}


# def get_stock_name(symbol):
#     response = urllib.request.urlopen(
#         f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
#     )
#     content = response.read()
#     data = json.loads(content.decode("utf8"))["quotes"][0]["shortname"]
#     return data


if side == "Dashboard":

    st.sidebar.markdown(
        "### The Dashboard will show the predictions for the top 5 stocks for short/long term. You can also get predictions for any other stock using the next block"
    )
    st.image("stock.png", width=1100)
    text = "{location}".format(location=location)
    st.markdown("### See predictions for top stocks for your country " + text)
    st.markdown(
        "###### *Not your country? Select your country from the dropdown below*"
    )
    location = st.selectbox("Select your country", location_index.keys())

    if location in location_index.keys():
        index = location_index[location]

    if index in top_stocks:
        stocks = top_stocks[index]

    choice = st.selectbox("Select your timeframe", ["Short Term", "Long Term"])

    # Short term predictions
    if choice == "Short Term":

        period = st.number_input("Number of days want to predict", step=1, value=14)
        submit = st.button("Submit")
        if submit:
            st.write("Expand below to view top 5 stock predictions")
            with st.expander("View predictions"):

                for stock in stocks:

                    # start = dt.datetime(2018,1,1)
                    # end = dt.datetime.now()
                    stock_name = get_stock_name(stock)
                    text = "Predictions for {stock}".format(stock=stock_name)
                    st.markdown("## " + text)
                    top, value, fig = plot_chart_short(stock, period)
                    st.metric(label="Predicted price", value=top, delta=value)
                    st.plotly_chart(fig)
                    st.markdown("_______")

    # Long term predictions
    if choice == "Long Term":

        period = st.number_input("Number of days want to predict", step=1, value=365)
        submit = st.button("Submit")
        if submit:
            st.write("Expand below to view top 5 stock predictions")
            with st.expander("View predictions"):

                for stock in stocks:

                    start = dt.datetime(2014, 1, 1)
                    end = dt.datetime.now()
                    stock_name = get_stock_name(stock)
                    text = "Predictions for {stock}".format(stock=stock_name)
                    st.markdown("## " + text)
                    top, value, fig = plot_chart(stock, period)
                    st.metric(label="Predicted price", value=top, delta=value)
                    st.plotly_chart(fig)
                    st.markdown("_______")

    # Predictions for any stock
    st.markdown("## Predict your own stock")

    company = st.text_input("Enter Stock/Index Ticker in Capitals")
    term = st.radio("Select Timeframe", ["Long Term", "Short Term"])

    if term == "Short Term":
        period = st.slider("Number of days want to predict", 1, 30)
        get_pred = st.button("Get Predictions")
        if get_pred:

            stock_name = get_stock_name(company)

            fig, score = sentiment_analysis(company)

            st.markdown("### News based Stock Sentiment")
            st.markdown("***Expand below tab to view***")

            with st.expander("Click to view stock sentiment based on News"):

                st.dataframe(score)
                st.plotly_chart(fig)

            text = "Prediction for {stock}".format(stock=stock_name)
            st.markdown("### " + text)
            top, value, fig = plot_chart_short(company, period)
            # fig = plot_chart_short(company, period)
            st.metric(label="Predicted price", value=top, delta=value)
            st.plotly_chart(fig)

    if term == "Long Term":
        period = st.number_input("Number of days to predict", step=1, value=365)
        submit = st.button("Get Predictions")
        if submit:

            stock_name = get_stock_name(company)
            text = "Prediction for {stock}".format(stock=stock_name)
            st.markdown("### " + text)

            analyst_score = analyst_recommendation(ticker=company)

            start = dt.datetime(2016, 1, 1)
            end = dt.datetime.now()
            
            top, value, fig = plot_chart(company, period)

            col1, col2 = st.columns(2)
            
            if analyst_score == "This ticker does not have Analyst Recommendations":

                col1.markdown("### Model predicted price")
                col1.metric(label="Predicted price", value=top, delta=value)
                st.plotly_chart(fig)

            else:
            
                change = round(analyst_score - top, 2)
                col1.markdown("### Model predicted price")
                col1.metric(label="Predicted price", value=top, delta=value)
                col2.markdown("### Analyst Predicted price after 1 year")
                col2.metric(label="Projected price", value=analyst_score, delta=change)
                col2.markdown("Change from predicted value")

                st.plotly_chart(fig)


if side == "Foreign Markets":
    st.sidebar.write("This will show you the predictions for other countries index's")
    foreign_index = location_index
    del foreign_index[location]

    for country, index in zip(foreign_index, foreign_index.values()):

        text = "{country} Stock Market".format(country=country)
        st.markdown("## " + text)
        start = dt.datetime(2020, 1, 1)
        end = dt.datetime.now()
        top, value, fig = plot_chart(index, period=365)
        st.metric(label="Predicted price", value=top, delta=value)
        st.plotly_chart(fig)


if side == "Stock News":
    st.title("Stock News")
    st.markdown("""---""")
    user_input = st.text_input("Enter Stock name")
    state = st.button("Get News!")
    if state:
        news = GNews().get_news(user_input)
        if news:
            for i in news:
                st.markdown(f"**{i['title']}**")
                st.write(f"Published Date - {i['published date']}")
                # st.write(i["description"])
                st.markdown(f"[Article Link]({i['url']})")
                st.markdown("""---""")

        else:
            st.write("No news for this stock")
