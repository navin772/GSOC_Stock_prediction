# Fin-dashboard - Your financial markets dashboard
![stock_image](https://img.freepik.com/premium-photo/financial-market-analytics-graph-world-map-background-scale-pieces-stock-markets_86639-1859.jpg)

Fin-dashboard is a web application that provides a dashboard for financial markets. It is built using [Streamlit](https://streamlit.io/). It leverages the facebook's Prophet ML library to help predict the stock/index prices for next n days/months/years. 

To access the web application visit [Fin-dashboard](https://navin-stock.streamlitapp.com/).

## Fin-dashboard features

Fin-dashboard provides a dashboard with the following features:

1. Automatic detection of the user’s country to show predictions for the top five, country-specific stocks
![dashboard](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/images/edge-analytics_finance_dashboard.png)

2. Ability for users to select long- or short-term predictions and to specify the number of forecasted days

3. Ability for users to choose other stocks and other stock exchanges

4. For long term predictions, a comparison of the model’s predicted price with the analyst’s target price
![analyst](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/images/edge-analytics_finance_analyst.png)

5. For short term predictions, a sentiment analysis of the stock news to enable visualization of its impact on stock price
![news](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/images/edge-analytics_finance_news.png)

6. Correlation of a stock with the index
![correlation](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/images/edge-analytics_finance_correlation.png)

7. Long term predictions for stock exchanges of different countries
![exchanges](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/images/edge-analytics_finance_foreign.png)

To read a more detailed guide on how to use and deploy the application, refer to [Edge Analytics with SUSE Rancher: Finance - Market Predictions](https://documentation.suse.com/trd/kubernetes/single-html/gs_rancher_edge-analytics_finance_stocks/).

To see the presentation and explanation of the Fin-dashboard project, refer to [this video](https://www.youtube.com/watch?v=D2mFfApyS_Q&t=1354s).

## File Structure

Fin-Dashboard is composed of the following Python modules:

1. geo_location.py: Determines the country where the user is located to give country specific information.

2. index_correlation.py: Correlates the stock with its parent index.

3. main.py: Contains all the Streamlit configurations/page setup and uses the other files to create the application.

4. plot_charts.py: Contains the modified and generalized form of the Jupyter Notebook code we created in the previous steps. Contains two python functions - one for long term and other for short term predictions.

5. sentiment.py: Collects stock news from Google News and uses the VADER sentiment analysis tool to determine the stock’s trend.

6. stock_recomendation.py: Extracts the analyst’s price target of a stock from Yahoo Finance.


## Steps to deploy the `fin_dashboard` app locally

1. Install necessary libraries using `pip install -r requirements.txt`
2. Run `streamlit run main.py` and access the application on `localhost:8501`

## Steps to deploy the `fin_dashboard` as a container

1. Pull the image from docker hub `sudo docker pull navin772/fin_dashboard:latest` or build the image yourself from the Dockerfile using `docker build .`
2. Start a new container using the image `sudo docker run -p 7000:8501 <image_id>`
3. Access the streamlit app on `localhost:7000`

## Steps for deploying on k8s cluster

1. Run `kubectl apply -k yaml_files` and kustomize will apply all the yaml files to the cluster currently configured with kubectl.
2. Access the app on `<NodeIP>:<NodePort>`