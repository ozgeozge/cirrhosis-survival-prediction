# Problem description

Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption.  This project aims to  predict survival state of patients with liver cirrhosis. 

The project offers a solution by providing a user-friendly endpoint. Users can submit a set of essential parameters, and in response, the system will provide an outcome, indicating the survival state of the pationt. The survival states include 0 = D (death), 1 = C (censored), 2 = CL (censored due to liver transplantation). The supervised machine learning model supports healthcare professionals and researchers to identify high-risk patients swiftly and accurately.

# Getting Started

You can find all the information to run/build the application on your local or Docker in this document.

## Prerequisites

- Python 3.10
- Docker
- [Cirrhosis Patient Survival Prediction dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data) from Kaggle
- Heroku account (optional for cloud deployment)

## Preparation

In order to train the model, you need to download the dataset from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data) and locate to the root directory of the project with name `cirrhosis.csv`. 

## Exploratory Data Analysis

The jupyter notebook file `notebook-capstone-project.ipynb` contains exploratory data analysis. In addition, four machine learning models namely Linear SVC, Decision Tree, Random Forest and XGBoost models have been trained with hyper-parameter tuning. The best model with the highest weighted f-1 score has been selected for deployment.

## Model Training

- To train the model, run `python train.py` command. This will read the `cirrhosis.csv` file and train the model with the dataset which it contains.
- This step creates `model_RF_ADASYN.bin` file which will be used in prediction.

## Build and run locally

- Run `pipenv shell` to create and activate a virtual environment
- `requirements.txt` will be automatically converted to `Pipfile`.
- Run `pipenv install` to install dependencies. It will also create `Pipfile.lock` file.
- Run `python predict.py` command to spin up an API endpoint to return the prediction of the given request.
- To test the endpoint, you can run below cURL command:

```shell
curl  -X POST \
  'http://localhost:5000/predict' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "N_Days": 400,
    "Drug": "D-penicillamine",
    "Age": 21464,
    "Sex": "F",
    "Ascites": "Y",
    "Hepatomegaly": "Y",
    "Spiders": "Y",
    "Edema": "Y",
    "Bilirubin": 14.5,
    "Cholesterol": 261.0,
    "Albumin": 2.6,
    "Copper": 156.0,
    "Alk_Phos": 1718.0,
    "SGOT": 137.95,
    "Tryglicerides": 172.0,
    "Platelets": 190.0,
    "Prothrombin": 12.2,
    "Stage": 4.0
}'
```

- You will get the result below after running the previous command:

```json
{
  "Status": "D"
}
```

## Build and run via Docker

- Run `docker build -t cirrhosis-survival-prediction .` command in the root directory of the project to create a docker image.
- Run `docker run -e PORT=5000 -p 9696:5000 --name=predictionApp cirrhosis-survival-prediction` to run a container which serves an API endpoint which exposes container default port `5000` to local `9696` port. Now, the app which is running inside the docker container is accessible with the address `http://localhost:9696`.
- To test the endpoint, you can run below cURL command:

```shell
curl  -X POST \
  'http://localhost:9696/predict' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "N_Days": 400,
    "Drug": "D-penicillamine",
    "Age": 21464,
    "Sex": "F",
    "Ascites": "Y",
    "Hepatomegaly": "Y",
    "Spiders": "Y",
    "Edema": "Y",
    "Bilirubin": 14.5,
    "Cholesterol": 261.0,
    "Albumin": 2.6,
    "Copper": 156.0,
    "Alk_Phos": 1718.0,
    "SGOT": 137.95,
    "Tryglicerides": 172.0,
    "Platelets": 190.0,
    "Prothrombin": 12.2,
    "Stage": 4.0
}'
```

- You will get the result below after running the previous command:

```json
{
  "Status": "D"
}
```

## Release & Deploy

GitHub action is used to deploy this project to Heroku. You can find the details in [actions file](.github/workflows/main.yaml).

![trigger deployment](/images/trigger_deployment.png "trigger doployment")

![deploy to heroku github actions](/images/deploy_to_heroku_github_actions.png "deploy to heroku github actions")

![send post request to heroku](/images/send_post_request_to_heroku.png "send post request to heroku")

# Contact

If you have any question please [create an issue](https://github.com/ozgeozge/cirrhosis-survival-prediction/issues/new).
