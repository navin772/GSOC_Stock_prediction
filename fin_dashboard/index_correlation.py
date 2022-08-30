import numpy as np
import plotly.express as px
import yfinance as yf

def correlate(stock, index, period):
    tickers = [stock, index]
    df = yf.download(tickers=tickers, period=period)
    df = df['Close']
    log_returns = np.log(df/df.shift())
    correlation = log_returns.corr()
    df = df/df.iloc[0]
    fig = px.line(df)
    fig = fig.update_layout(width=1300, height=600)
    correlation = correlation.reset_index()
    num = correlation[index][0]
    num = (num*100).round(2)

    return fig, num