# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
Resnet50 was selected for tuning.

- Completed training jobs
<img src="./images/CompletedTrainingJob.png" alt="CompletedTrainingJob" title="CompletedTrainingJob">

- Logs metrics during the training process
<img src="./images/TestLog.png" alt="Test" title="Test">

- Tune at least two hyperparameters
```
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256]),
}
```
<img src="./images/HyperparameterTuning.png" alt="HyperparameterTuning" title="HyperparameterTuning">

- Retrieve the best best hyperparameters from all your training jobs

<img src="./images/Best_HyperparameterTuningJob.png" alt="BestHyperparameterTuning" title="BestHyperparameterTuning">

## Debugging and Profiling
<a href="./report/profiler-report.html" target="_top">Profiler Report</a>

### Results
#### Success
<img src="./images/Success.png" alt="Success" title="Success Result">

#### Fail
<img src="./images/Fail.png" alt="Fail" title="Fail Result">


## Model Deployment
<img src="./images/ModelDeployment.png" alt="Deployed Endpoint" >

