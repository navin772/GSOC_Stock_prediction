from prophet import Prophet
from prophet.plot import plot_plotly
import yfinance as yf
import urllib
import json

def get_stock_name(symbol):
    response = urllib.request.urlopen(
        f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}"
    )
    content = response.read()
    data = json.loads(content.decode("utf8"))["quotes"][0]["shortname"]
    return data


def plot_chart(company, period):

    df = yf.download(tickers=company, period="7y", interval="1d")
    name = get_stock_name(company)
    # data preprocessing
    df = df.reset_index()
    new_df = df[["Date", "Close"]]
    new_df = new_df.rename(columns={"Date": "ds", "Close": "y"})
    new_df["ds"] = new_df["ds"].dt.tz_localize(None)

    # initialize prophet model
    fp = Prophet(yearly_seasonality=True)
    fp.fit(new_df)

    # make future predictions
    future = fp.make_future_dataframe(periods=period)
    forecast = fp.predict(future)

    # Plot the predictions
    fig = plot_plotly(fp, forecast)
    fig.update_xaxes(title_text="Time")
    y_text = "{company_name} price".format(company_name=name)
    fig.update_yaxes(title_text=y_text)
    fig.update_layout(autosize=False)

    top = forecast["yhat"].iloc[-1]
    top = round(top, 2)
    curr = df["Close"].iloc[-1]
    curr = round(curr, 2)
    change = ((top - curr) / curr) * 100
    change = round(change, 2)
    value = "{change} %".format(change=change)

    return top, value, fig


def plot_chart_short(company, period):

    data = yf.download(tickers=company, period="60d", interval="30m")
    name = get_stock_name(company)

    data = data.reset_index()
    new_df = data.rename(columns={"Datetime": "ds", "Close": "y"})
    new_df = new_df[["ds", "y"]]
    new_df["ds"] = new_df["ds"].dt.tz_localize(None)

    fp = Prophet(daily_seasonality=True)
    fp.fit(new_df)
    future = fp.make_future_dataframe(periods=period)
    forecast = fp.predict(future)

    fig = plot_plotly(fp, forecast)
    fig.update_xaxes(title_text="Time")
    y_text = "{company_name} price".format(company_name=name)
    fig.update_yaxes(title_text=y_text)
    fig.update_layout(autosize=False)

    top = forecast["yhat"].iloc[-1]
    top = round(top, 2)
    curr = data["Close"].iloc[-1]
    curr = round(curr, 2)
    change = ((top - curr) / curr) * 100
    change = round(change, 2)
    value = "{change} %".format(change=change)

    return top, value, fig