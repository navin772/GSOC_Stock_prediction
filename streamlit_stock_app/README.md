## Steps to deploy the `streamlit_stock_app` locally

1. Install necessary libraries using `pip install -r requirements.txt`
2. Run `streamlit run main.py` and access the application on `localhost:8501`

## Steps to deploy the `streamlit_stock_app` as a container

1. Pull the image from docker hub `sudo docker pull navin772/streamlitstockapp:2.0` or build the image yourself from the Dockerfile.
2. Start a new container using the image `sudo docker run -p 8501:8501 <image_id>`
3. Access the streamlit app on `localhost:8501`

## Steps for deploying on k8s cluster

1. Run `kubectl apply -k yaml_files` and kustomize will apply all the yaml files to the cluster currently configured with kubectl.
2. Access the app on `<NodeIP>:<NodePort>`