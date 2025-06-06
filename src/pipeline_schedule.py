from google.cloud import aiplatform

PROJECT_ID = "discai-450508"
LOCATION = "us-central1" 

PIPELINE_JOB_DISPLAY_NAME = "mri-unified-prod-template" 
COMPILED_PIPELINE_PATH = "gs://discai-450508-deployment/unified_pipeline.json" 
PIPELINE_ROOT_PATH = "gs://discai-450508-pipeline/production_scheduled_runs" 
PIPELINE_RUNNER_SERVICE_ACCOUNT = "vertex-ai-user@discai-450508.iam.gserviceaccount.com" 

SCHEDULE_DISPLAY_NAME = "mri-unified-prod-schedule" 
CRON_EXPRESSION = "55 1 * * *"  


MAX_CONCURRENT_RUN_COUNT = 3 
PIPELINE_PARAMETERS = {
    "project": PROJECT_ID,
    "location": LOCATION,
    "pipeline_root": PIPELINE_ROOT_PATH,
    "service_account": PIPELINE_RUNNER_SERVICE_ACCOUNT, 
    "tensorboard_instance": "projects/63404319155/locations/us-central1/tensorboards/4154742180063215616",
    "run_segmentation_pipeline": False,
    
    "seg_base_dir": f"/gcs/{PROJECT_ID}-raw-data/SPIDER/",
    "seg_output_dir": f"/gcs/{PROJECT_ID}-processed-data/segmentation_processed/",
    "seg_model_output_dir": f"gs://{PROJECT_ID}-pipeline/segmentation_model/", 
    "preprocess_seg_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/preprocessing-seg-image:latest",
    "train_seg_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/training-seg-image:latest",
    "seg_model_serving_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/training-seg-image:latest",
    "seg_model_display_name": "mri-segmentation-model",
    "seg_epochs": 5,
    "seg_batch_size": 4,
    "seg_image_height": 512,
    "seg_image_width": 512,
    
    "original_processed_train_dir": f"/gcs/{PROJECT_ID}-processed-data/Full Data/train/",
    "original_processed_val_dir": f"/gcs/{PROJECT_ID}-processed-data/Full Data/val/",
    "original_processed_test_dir": f"/gcs/{PROJECT_ID}-processed-data/Full Data/test/",
    "class_train_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/training-class-image:latest",
    "class_serving_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/training-class-image:latest",
    "corrections_gcs_dir": f"gs://{PROJECT_ID}-deployment/corrections/", 
    "accepted_predictions_gcs_dir": f"gs://{PROJECT_ID}-deployment/accepted_predictions/",
    "original_sequences_gcs_dir": f"gs://{PROJECT_ID}-deployment/input_sequences/", 
    "preprocess_class_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/preprocess-corrections-image:latest",
    "class_model_display_name": "mri-classification-model-retrained",
    "class_epochs": 50,
    "class_batch_size": 32,
    "tuner_epochs": 15,
    "max_tuner_trials": 5, 
    "patch_height": 40,
    "patch_width": 80,    
    "classification_processed_data_gcs_path": f"gs://{PROJECT_ID}-processed-data/classification_preprocessed/", 
    "evaluation_data_gcs_dir": f"gs://{PROJECT_ID}-processed-data/Full Data/test/", 
    "evaluation_image_uri": f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/discai/evaluation-image:latest",
    "production_model_uri": f"gs://{PROJECT_ID}-deployment/models/", 
    "evaluation_metric_name": "accuracy",
    "evaluation_metric_threshold": 0.0,
    'target_url': "https://mri-backend-service-yew6jcdx7a-uc.a.run.app/reload-models",
}


def create_pipeline_schedule():
    """Creates a Vertex AI Pipeline Job Schedule."""

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    
    pipeline_job = aiplatform.PipelineJob(
        display_name=PIPELINE_JOB_DISPLAY_NAME,
        template_path=COMPILED_PIPELINE_PATH,
        pipeline_root=PIPELINE_ROOT_PATH,
        parameter_values=PIPELINE_PARAMETERS,
        enable_caching=None, 
    )

    print(f"PipelineJob object configured to run as: {PIPELINE_RUNNER_SERVICE_ACCOUNT}")
    print("Pipeline Parameters being used:")
    for key, value in PIPELINE_PARAMETERS.items():
        print(f"  {key}: {value}")


    pipeline_job_schedule = pipeline_job.create_schedule(
        display_name=SCHEDULE_DISPLAY_NAME,
        cron=CRON_EXPRESSION,
        max_concurrent_run_count=MAX_CONCURRENT_RUN_COUNT,
        start_time=None,  
        end_time="2025-05-31T21:00:00Z",    
        service_account=PIPELINE_RUNNER_SERVICE_ACCOUNT #
    )

    print(f"Successfully created/updated schedule:")
    print(f"  Display Name: {pipeline_job_schedule.display_name}")
    print(f"  Resource Name: {pipeline_job_schedule.resource_name}")
    print(f"  Cron: {pipeline_job_schedule.cron}")
    print(f"  State: {pipeline_job_schedule.state}")
    print(f"  Pipeline Runner SA (for jobs): {PIPELINE_RUNNER_SERVICE_ACCOUNT}")

if __name__ == "__main__":
    create_pipeline_schedule()
