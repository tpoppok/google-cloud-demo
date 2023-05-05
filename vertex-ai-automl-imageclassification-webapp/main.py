import os,requests, json, base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.oauth2 import service_account
from flask import Flask, request, render_template

ENDPOINT_ID = os.environ.get("ENDPOINT_ID")
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")
API_ENDPOINT = os.environ.get("API_ENDPOINT")

credentials = service_account.Credentials.from_service_account_file('/sec/compute-sa')

app = Flask(__name__)

@app.route("/prediction", methods=["POST"])
def prediction(
    project: str = PROJECT_ID,
    endpoint_id: str = ENDPOINT_ID,
    location: str = REGION,
    api_endpoint: str = API_ENDPOINT,
):
    aiplatform.init(project=project, location=location, credentials=credentials)
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    file = request.files['image'].read()
    image_byte = base64.b64encode(file).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=image_byte,
    ).to_value()
    instances = [instance]

    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    output = response.predictions
    result = output[0]
  
    return render_template(
        'result.html',
        model_name=response.model_display_name,
        model_id=response.deployed_model_id,
        model_version=response.model_version_id,
        displayname=result["displayNames"][0],
        confidence=result["confidences"][0]
    )



@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))