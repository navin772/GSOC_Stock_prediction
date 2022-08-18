# GSOC_Stock_prediction
A ML model implementation on how to train a model on existing Index data and try to predict the future value of the Index.

This repository contains the following apps:
1. `index_prediction_app` - Made on flask and can be used to predict the future value of IXIC and NSE indices.
2. `streamlit_stock_app` - Made using Streamlit and uses LSTM and Prophet for stock/index predictions.
3. `fin_dashboard` - The latest iteration of the app, contains many useful features for comparisons, predictions and chart visualization.

All 3 apps are deployable as containers using the provided dockerfile in their respective directory and can also orchestrated using kustomize on any k8s cluster. The directory also contains the YAML files for the deployment.

For deployment instructions refer to the documentation inside each app directory.

## Steps to run index_prediction_app on local machine

1. cd into `index_prediction_app` folder
2. Install required libraries `pip install -r requirements.txt`
3. Inside the `index_prediction_app` folder run `flask run` and access the app in the browser.
## Steps to run the index_prediction_app inside a container:

1. Pull the image from docker hub `sudo docker pull navin772/preds_dev:4.0` or you can build the image using the Dockerfile.
2. Start a new container using the pulled image `sudo docker run -p 7000:5000 <image_id>`
3. Access the flask app on `localhost:7000`

## Steps for deploying on k8s cluster

1. cd into `k8s_deployment_yaml` folder and run `kubectl apply -k` and kustomize will apply all the yaml files to the k8s cluster.
2. Access the app on `<NodeIP>:<NodePort>`

 ## Mentors
 This project is done during Google Summer of Code 2022 and is mentored by [Bryan Gartner](https://github.com/bwgartner), [Brian Fromme](https://github.com/mrjazzcat) and [Ann Davis](https://github.com/andavissuse).
 
 Organization - openSUSE
## License
 `index_prediction_app/indexData.csv` is licensed by Data files © Original Authors. Data sourced from [Kaggle](https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data)