# GSOC_Stock_prediction
![stock_image](https://www.umpindex.com/images/UMPI-Stock-Market-Projection-Software.png)
A ML model implementation on how to train a model on existing Index data and try to predict the future value of the Index.

This repository contains the following apps:
1. `index_prediction_app` - Made on flask and can be used to predict the future value of IXIC and NSE indices.
2. `streamlit_stock_app` - Made using Streamlit and uses LSTM and Prophet for stock/index predictions.
3. `fin_dashboard` - The latest iteration of the app, contains many useful features for comparisons, predictions and chart visualization.

All 3 apps are deployable as containers using the provided dockerfile in their respective directory and can also orchestrated using kustomize on any k8s cluster. The directory also contains the YAML files for the deployment.

For deployment instructions refer to the documentation inside each app directory.

Visit my Medium account to read detailed blogs for the work done here - [Medium-Navin Chandra](https://medium.com/@navinchandra772)
 ## Mentors
 This project is done during Google Summer of Code 2022 and is mentored by [Bryan Gartner](https://github.com/bwgartner), [Brian Fromme](https://github.com/mrjazzcat) and [Ann Davis](https://github.com/andavissuse).
 
 Organization - [openSUSE](https://www.opensuse.org/)
## License
 `index_prediction_app/indexData.csv` is licensed by Data files © Original Authors. Data sourced from [Kaggle](https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data)