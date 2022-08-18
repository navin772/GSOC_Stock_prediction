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