import sagemaker
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import boto3
import os
import pandas as pd
from sagemaker.sklearn.estimator import SKLearn



#Preparein Data,by getting it from S3 and then spliting it into train and test data
def prepare_data(bucket_name,key):
    """This function is used to prepare the data for training the model. It gets the data from S3 and then splits it into train and test data.
    Args:
        bucket_name (str): The name of the S3 bucket where the data is stored.
        key (str): The key of the S3 object where the data is stored.
        Download the data from s3,load into pandas dataframe, and return request and actions dataframes."""

     s3=boto3.client('s3')
    local_path='/tmp/error_log.csv'
    s3.download_file(bucket_name,key,local_path)
    df=pd.read_csv(local_path, on_bad_lines='skip', engine='python')
    
    # Handle different column names
    error_col = 'Error' if 'Error' in df.columns else df.columns[0]
    action_col = 'ActionRecommended' if 'ActionRecommended' in df.columns else df.columns[1]
    
    request_df=df[error_col]
    actions_df=df[action_col]
    return request_df,actions_df

#Create training script
def create_training_script(script_path="train_script.py"):  
    """Create a local training script that the SKLearn Estimator can use."""
    script_code='''
import argparse
import pandas as pd
import os
import joblib
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def model_fn(model_dir):
    """Load the model from the model_dir."""
    model_path=os.path.join(model_dir,"model.joblib")
    return joblib.load(model_path)

def input_fn(request_body,content_type):
    """Deserialize the input data to the format expected by the model."""
    if content_type=="application/json":
      requestData=json.loads(request_body)    

      if isinstance(requestData,list) or isinstance(requestData,str):
        return requestData
      else:
        raise ValueError("Invalid input data format. Expected a list or a string.")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data,model):
    """Make a prediction using the model."""
    if isinstance(input_data,str):
        input_data=[input_data]
    return model.predict(input_data)

def output_fn(prediction,accept):
    """Serialize the prediction to the format expected by the client."""
    if accept=="application/json":
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported accept type: {accept}")  

if __name__=="__main__":
    parser=argparse.ArgumentParser()   
    parser.add_argument("--output-data-dir",type=str,default="/opt/ml/output/data")
    parser.add_argument("--train",type=str,default="/opt/ml/input/data/train") 
    parser.add_argument("--model-dir",type=str,default="/opt/ml/model")
    args=parser.parse_args()

    #Load the training data
    train_data=pd.read_csv(os.path.join(args.train,"train.csv"))
    X_train=train_data["Error"].to_list()
    y_train=train_data["ActionRecommended"].to_list()
    #Create a pipeline that combines CountVectorizer and LogisticRegression
    pipeline=Pipeline([
        ("vectorizer",CountVectorizer()),
        ("classifier",LogisticRegression())
    ])
    pipeline.fit(X_train,y_train)
    #Save the modelto the model directory
    joblib.dump(pipeline,os.path.join(args.model_dir,"model.joblib"))
    print("Model training completed and saved to {}".format(args.model_dir))
'''

    with open(script_path, 'w') as f:
        f.write(script_code)

def main():
    session = sagemaker.Session()
    bucket_name = os.environ.get('S3_BUCKET', 'sagemaker-basic-poc')
    data_key = "data/error_logs.csv"
    
    # Download data to local /tmp, parse
    X, y = prepare_data(bucket_name, data_key)
    df = pd.DataFrame({"Error": X, "ActionRecommended": y})
    
    # Save training data locally first
    os.makedirs("poc_data", exist_ok=True)
    training_data_path = "poc_data/train.csv"
    df.to_csv(training_data_path, index=False)
    
    # Upload training data to S3
    s3_train_path = session.upload_data(training_data_path, bucket=bucket_name, key_prefix="training")
    
    # Create the training script
    create_training_script()

    # Create the SKLearn Estimator
    # Get execution role from environment variable or use auto-detection
    execution_role = os.environ.get('SAGEMAKER_EXECUTION_ROLE', None)
    if not execution_role:
        try:
            execution_role = sagemaker.get_execution_role()
        except ValueError:
            raise ValueError("SAGEMAKER_EXECUTION_ROLE environment variable not set and current identity is not a role")
    
    sklearn_estimator = SKLearn(
        entry_point="train_script.py",
        framework_version="1.0-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=execution_role,
        base_job_name="logs-error-model",
        sagemaker_session=session
    )
    
    # # Create the SKLearn Estimator
    # sklearn_estimator = SKLearn(
    #     entry_point="train_script.py",
    #     framework_version="1.0-1",
    #     instance_type="ml.m5.xlarge",
    #     instance_count=1,
    #     role=sagemaker.get_execution_role(),
    #     base_job_name="logs-error-model",
    #     sagemaker_session=session
    # )
    
    # Fit using S3 path instead of local path
    sklearn_estimator.fit({'train': s3_train_path})
    
    # Deploy model with serializer configuration
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="logs-error-endpoint",
        serializer=sagemaker.serializers.JSONSerializer(),
        deserializer=sagemaker.deserializers.JSONDeserializer()
    )
    
    print("Model deployed successfully. Testing endpoint...")
    
    # Test the endpoint with a single prediction
    test_error = ["Error 500: Internal Server Error"]
    try:
        response = predictor.predict(test_error)
        print(f"Test successful!")
        print(f"Input Error: {test_error[0]}")
        print(f"Predicted Action: {response[0]}")
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")

if __name__ == "__main__":
    main()
        