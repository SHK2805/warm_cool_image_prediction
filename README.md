# Warm Cool Prediction
Application description
## Description
This application is a simple image classifier that predicts whether an image is warm or cool. 
The application uses opencv to predict the image. 
The application is built using OpenCV and Flask framework. 
The flask application runs on http://127.0.0.1:8080
The user can upload an image to the application, and the application will predict whether the image is warm or cool.
The files will be uploaded to the images folder in the application directory.

## Installation
Run the app from the terminal using the following commands:
```bash
# on terminal
# navigate to the project path
cd /path/to/your/project
pip install -r requirements.txt
python app.py
```
Run the app in docker
```bash
# navigate to the project path
cd /path/to/your/project
# When using the docker compose use the below
# since we are using the docker compose we will run the below command
docker-compose up
# stop the container
docker-compose down

# if you are not using docker compose use the below
# build the image
docker build -t warm_cool_image_prediction .
# run the docker
docker run -p 8000:8080 warm_cool_image_prediction

# see logs
docker-compose logs web

# stop the container
docker-compose down
# rebuild the container
docker-compose up --build

# check the container
docker ps
```
