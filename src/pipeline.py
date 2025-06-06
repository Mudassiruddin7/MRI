from kfp import dsl
from kfp.dsl import pipeline, If as KFP_If, Input, Output, Artifact, Dataset, Model, OutputPath, Metrics
from kfp import compiler
import argparse
import google.cloud.aiplatform as aip
import sys
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp





@dsl.container_component
def preprocess_segmentation_data(
    base_dir: str,
    output_dir: str,
    image_height: int,
    image_width: int,
    preprocess_seg_image_uri: str
    ):
    """Preprocesses raw data for segmentation training."""

    return dsl.ContainerSpec(
        image=str(preprocess_seg_image_uri),
        command=['python', 'preprocessing.py'], 
        args=[
            "--base_dir", base_dir,
            "--output_dir", output_dir,
            "--image_height", str(image_height),
            "--image_width", str(image_width),
        ],
    )


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-pipeline-components"]
)
def create_unmanaged_model(
    model_serving_container_image_uri: str,
    model_artifact_dir: str,
    unmanaged_model: Output[artifact_types.UnmanagedContainerModel],
):
    """Creates an UnmanagedContainerModel artifact for ModelUploadOp."""
    unmanaged_model.uri = model_artifact_dir
    unmanaged_model.metadata["containerSpec"] = {
        "imageUri": str(model_serving_container_image_uri),
    }


@dsl.container_component
def preprocess_corrections_for_classification(
    project: str,
    location: str,
    corrections_gcs_dir: str,
    original_sequences_gcs_dir: str,
    accepted_predictions_gcs_dir: str,
    patch_height: int,
    patch_width: int,
    output_classification_data_gcs_path: str,
    preprocess_class_image_uri: str
):
    """
    Preprocesses correction data (YOLO txt) and original images
    to create patch stacks (.jpg) and labels (csv) for classification training.
    """
    
    return dsl.ContainerSpec(
        image=str(preprocess_class_image_uri),
        command=["python", "preprocess_corrections.py"],
        args=[
            "--corrections-gcs-dir", corrections_gcs_dir,
            "--original-sequences-gcs-dir", original_sequences_gcs_dir,
            "--output-classification-data-gcs-path", output_classification_data_gcs_path,
            "--accepted-predictions-gcs-dir", accepted_predictions_gcs_dir,
            "--patch-height", str(patch_height),
            "--patch-width", str(patch_width),
        ]
    )


@dsl.container_component
def evaluate_classification_models_container_op(
    project: str,
    location: str,
    evaluation_image_uri: str, 
    new_model: str,   
    production_model_uri: str, 
    evaluation_data_gcs_dir: str, 
    batch_size: int,
    metrics_name: str,
    comparison_threshold: float,
    patch_height: int,
    patch_width: int,
    deploy_decision: OutputPath(str),
    new_model_metrics: Output[Metrics],
    production_model_metrics: Output[Metrics]
):
    """Container component wrapper for model evaluation script."""
    return dsl.ContainerSpec(
        image=str(evaluation_image_uri), 
        command=["python", "evaluate_models.py"], 
        args=[
            "--project", project,
            "--location", location,
           
            "--new-model-uri", new_model,
            "--production-model-uri", production_model_uri,
            "--evaluation-data-gcs-dir", evaluation_data_gcs_dir,
            "--batch-size", str(batch_size),
            "--metrics-name", metrics_name,
            "--comparison-threshold", str(comparison_threshold),
            "--patch-height",patch_height,
            "--patch-width",patch_width,
            
            "--deploy-decision-output-path", deploy_decision,
            "--new-model-metrics-output-path", new_model_metrics.path,
            "--production-model-metrics-output-path", production_model_metrics.path,
        ]
    )


@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["google-cloud-storage"]
)
def copy_gcs_file_to_folder(
    source_gcs_file_path: str,
    destination_gcs_folder_path: str,
):
    """Copies a single file from a GCS location to a GCS folder,
    preserving the original file name.

    Args:
        source_gcs_file_path: The full GCS path to the source file
                              (e.g., "gs://source-bucket/path/to/file.txt").
        destination_gcs_folder_path: The GCS path to the destination folder
                                     (e.g., "gs://dest-bucket/path/to/folder/" or
                                      "gs://dest-bucket/path/to/folder").
                                     If copying to bucket root, use "gs://dest-bucket/" or "gs://dest-bucket".
    """
    import os
    from google.cloud import storage
    def parse_gcs_uri_for_file(file_path: str) -> tuple[str, str]:
        """Parses a GCS file path into bucket and blob name.
        Ensures the blob name is a valid file name (not empty or ending with '/').
        """
        if not file_path.startswith("gs://"):
            raise ValueError(f"GCS file path must start with gs://, but got '{file_path}'")
        
        path_sans_scheme = file_path[5:]
        if not path_sans_scheme: 
            raise ValueError("GCS file path cannot be empty after gs://")

        parts = path_sans_scheme.split('/', 1)
        bucket_name = parts[0]
        if not bucket_name: 
            raise ValueError("Bucket name cannot be empty in GCS file path.")

        blob_name = ""
        if len(parts) > 1:
            blob_name = parts[1]
        
        if not blob_name or blob_name.endswith('/'):
            raise ValueError(
                f"Invalid GCS file path. It must point to a specific file, "
                f"not a directory or bucket root: '{file_path}'"
            )
            
        return bucket_name, blob_name

    def parse_gcs_uri_for_folder(folder_path: str) -> tuple[str, str]:
        """Parses a GCS folder path into bucket and prefix.
        Ensures the prefix ends with '/' if not representing the bucket root.
        """
        if not folder_path.startswith("gs://"):
            raise ValueError(f"GCS folder path must start with gs://, but got '{folder_path}'")
        
        path_sans_scheme = folder_path[5:]
        if not path_sans_scheme: 
            raise ValueError("GCS folder path cannot be empty after gs://")

        parts = path_sans_scheme.split('/', 1)
        bucket_name = parts[0]
        if not bucket_name: 
            raise ValueError("Bucket name cannot be empty in GCS folder path.")

        prefix = ""
        if len(parts) > 1 and parts[1]:  
            prefix = parts[1]
            if not prefix.endswith('/'):
                prefix += '/'
        
        
        return bucket_name, prefix

    source_bucket_name, source_blob_full_path = parse_gcs_uri_for_file(source_gcs_file_path)
    dest_bucket_name, dest_folder_prefix = parse_gcs_uri_for_folder(destination_gcs_folder_path)


    source_file_basename = os.path.basename(source_blob_full_path)
    if not source_file_basename: 
        raise ValueError(f"Could not extract a valid file name from source path: {source_gcs_file_path}")

    
    destination_blob_full_path = dest_folder_prefix + source_file_basename

    client = storage.Client()

    source_bucket = client.bucket(source_bucket_name)
    dest_bucket = client.bucket(dest_bucket_name)
    source_blob = source_bucket.blob(source_blob_full_path)

    if not source_blob.exists(client):
        error_message = f"Source file gs://{source_bucket_name}/{source_blob_full_path} not found."
        print(error_message)
        raise FileNotFoundError(error_message)

    print(f"Attempting to copy gs://{source_bucket_name}/{source_blob_full_path} "
          f"to gs://{dest_bucket_name}/{destination_blob_full_path}")

    source_bucket.copy_blob(
        blob=source_blob,
        destination_bucket=dest_bucket,
        new_name=destination_blob_full_path  
    )

    print(f"Successfully copied {source_gcs_file_path} "
          f"to gs://{dest_bucket_name}/{destination_blob_full_path}")
    

@dsl.component(
    base_image="python:3.9",  
    packages_to_install=["requests", "google-auth"]
)
def http_post_request_oidc(
    target_url: str, 
    
) -> str:
    """Makes an authenticated HTTP POST request to the target_url using OIDC."""
    import requests
    import google.auth.transport.requests
    import google.oauth2.id_token
    import logging 

    logger = logging.getLogger("PipelineHttpCall")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        audience = target_url
        if "/reload-models" in target_url: 
             audience = target_url.split("/reload-models")[0]


        logger.info(f"Attempting to fetch OIDC token for audience: {audience}")
        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
        logger.info("Successfully fetched OIDC token.")

        headers = {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }

        logger.info(f"Making POST request to: {target_url}")
        response = requests.post(target_url, headers=headers, json={})
        response.raise_for_status()  

        logger.info(f"Successfully called endpoint. Status: {response.status_code}, Response: {response.text}")
        return f"Call successful: {response.status_code} - {response.text}"

    except Exception as e:
        logger.error(f"Error making HTTP POST request to {target_url}: {e}", exc_info=True)
        raise


@pipeline(
    name="mri-unified-training-pipeline",
    description="A pipeline to conditionally train segmentation and process corrections for classification retraining."
)
def mri_unified_pipeline(
    project: str,
    location: str,
    pipeline_root: str,
    service_account: str,
    tensorboard_instance: str = "",
    run_segmentation_pipeline: bool = False,
    seg_base_dir: str = "",
    seg_output_dir: str = "",
    seg_model_output_dir: str = "",
    preprocess_seg_image_uri: str = "",
    train_seg_image_uri: str = "",
    seg_model_serving_image_uri: str = "",
    seg_model_display_name: str = "mri-segmentation-model",
    seg_epochs: int = 5,
    seg_batch_size: int = 4,
    seg_image_height: int = 512,
    seg_image_width: int = 512,
    original_processed_train_dir: str = "",
    original_processed_val_dir: str = "",   
    original_processed_test_dir: str = "",  
    class_train_image_uri: str = "",
    class_serving_image_uri: str = "",
    corrections_gcs_dir: str = "",
    accepted_predictions_gcs_dir: str = "",
    original_sequences_gcs_dir: str = "",
    preprocess_class_image_uri: str = "",
    class_model_display_name: str = "mri-classification-model-retrained",
    class_epochs: int = 50,
    class_batch_size: int = 32,
    tuner_epochs: int = 15,
    max_tuner_trials: int = 15,
    patch_height: int = 50,
    patch_width: int = 100,
    classification_processed_data_gcs_path: str = "",
    evaluation_data_gcs_dir: str = "",
    evaluation_image_uri: str = "",    
    production_model_uri: str = "", 
    evaluation_metric_name: str = "accuracy",   
    evaluation_metric_threshold: float = 0.0,  
    target_url: str = "",

):
    
    with KFP_If(run_segmentation_pipeline == True, name="segmentation-train-flow"):
        preprocess_seg_op = preprocess_segmentation_data(
            base_dir=seg_base_dir,
            output_dir=seg_output_dir,
            image_height=seg_image_height,
            image_width=seg_image_width,
            preprocess_seg_image_uri=str(preprocess_seg_image_uri),
        ).set_caching_options(enable_caching=True)

        train_seg_task = CustomTrainingJobOp(
            project=project,
            location=location,
            display_name="mri-segmentation-training-job",
            worker_pool_specs=[{
                "machine_spec": { "machine_type": "n1-standard-4", "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1 },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": str(train_seg_image_uri),
                    "command": ["python", "seg-training.py"],
                    "args": [
                        "--tfrecord_dir", seg_output_dir, 
                        "--model_output_dir", seg_model_output_dir,
                        "--epochs", str(seg_epochs),
                        "--batch_size", str(seg_batch_size),
                    ],
                },
            }],
            service_account=service_account,
            base_output_directory=seg_model_output_dir,
            tensorboard=tensorboard_instance if tensorboard_instance else None,
        ).after(preprocess_seg_op)
        train_seg_task.set_caching_options(enable_caching=False)

        create_seg_unmanaged_model_op = create_unmanaged_model(
            model_serving_container_image_uri=str(seg_model_serving_image_uri),
            model_artifact_dir=seg_model_output_dir 
        ).after(train_seg_task)

        upload_seg_model_op = ModelUploadOp(
            project=project,
            location=location,
            display_name=seg_model_display_name,
            unmanaged_container_model=create_seg_unmanaged_model_op.outputs["unmanaged_model"],
        ).after(create_seg_unmanaged_model_op)
        upload_seg_model_op.set_display_name("Upload Segmentation Model")

    
    model_path = f"{pipeline_root}/class_training_job_outputs/{dsl.PIPELINE_JOB_ID_PLACEHOLDER}"

    
    preprocess_corrections_op = preprocess_corrections_for_classification(
        project=project,
        location=location,
        corrections_gcs_dir=corrections_gcs_dir,
        original_sequences_gcs_dir=original_sequences_gcs_dir,
        patch_height=patch_height,
        patch_width=patch_width,
        accepted_predictions_gcs_dir=accepted_predictions_gcs_dir,
        preprocess_class_image_uri=str(preprocess_class_image_uri),
        output_classification_data_gcs_path=classification_processed_data_gcs_path

    )
    preprocess_corrections_op.set_caching_options(enable_caching=False)
    preprocess_corrections_op.set_display_name("Preprocess Corrections for Classification")
    
    train_class_task = CustomTrainingJobOp(
        project=project,
        location=location,
        display_name="mri-classification-retraining-job",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-8", 
                "accelerator_type": "NVIDIA_TESLA_T4", 
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": class_train_image_uri,
                "command": ["python", "class_training.py"], 
                "args": [
                    
                    "--original-train-dir", original_processed_train_dir,
                    "--original-val-dir", original_processed_val_dir,
                    "--original-test-dir", original_processed_test_dir,
                    
                    "--corrected-train-dir", classification_processed_data_gcs_path,
                    
                    "--patch-height", str(patch_height),
                    "--patch-width", str(patch_width),
                    "--epochs", str(class_epochs),
                    "--batch-size", str(class_batch_size),
                    "--tuner-epochs", str(tuner_epochs),
                    "--max-tuner-trials", str(max_tuner_trials),
                    "--project-id", project, 
                    "--location", location, 
                ],
            },
        }],
        
        base_output_directory=model_path,
        service_account=service_account,
        tensorboard=tensorboard_instance if tensorboard_instance else None,
    ).after(preprocess_corrections_op) 
    train_class_task.set_caching_options(enable_caching=False)

    
    create_class_unmanaged_model_op = create_unmanaged_model(
        model_serving_container_image_uri=str(class_serving_image_uri),
        model_artifact_dir=f"{model_path}/model/"
    ).after(train_class_task).set_caching_options(enable_caching=True)

    upload_class_model_op = ModelUploadOp(
        project=project,
        location=location,
        display_name=class_model_display_name,
        unmanaged_container_model=create_class_unmanaged_model_op.outputs["unmanaged_model"],
    ).after(create_class_unmanaged_model_op).set_caching_options(enable_caching=False)
    
    

    evaluate_op = evaluate_classification_models_container_op(
        project=project,
        location=location,
        evaluation_image_uri=evaluation_image_uri, 
        new_model=f"{model_path}/model/classification_model.keras", 
        production_model_uri=production_model_uri,
        evaluation_data_gcs_dir=evaluation_data_gcs_dir,
        batch_size=class_batch_size, 
        metrics_name=evaluation_metric_name,
        comparison_threshold=evaluation_metric_threshold,
        patch_height=patch_height,
        patch_width=patch_width,
    ).after(upload_class_model_op) 
    evaluate_op.set_display_name("Evaluate New Model vs Production").set_caching_options(enable_caching=False)
    
    with KFP_If(evaluate_op.outputs['deploy_decision'] == "true", name="copy-to-production-gcs"):
        copy_model_op = copy_gcs_file_to_folder(
            
            source_gcs_file_path=f"{model_path}/model/classification_model.keras",
            
            destination_gcs_folder_path=production_model_uri
 
        ).after(evaluate_op, upload_class_model_op) 
        copy_model_op.set_display_name("Copy Model to Production GCS Path").set_caching_options(enable_caching=False)
        
        reload_model = http_post_request_oidc(target_url=target_url).after(copy_model_op)
        reload_model.set_display_name("Call /reload-model for model refresh in backend").set_caching_options(enable_caching=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-only', action='store_true', help='Only compile the pipeline to JSON.')

    parser.add_argument('--project-id', type=str, required=not "--compile-only" in sys.argv)
    parser.add_argument('--region', type=str, default='us-central1')
    parser.add_argument('--pipeline-root', type=str, required=not "--compile-only" in sys.argv)
    parser.add_argument("--service-account", type=str, required=not "--compile-only" in sys.argv)
    parser.add_argument("--display-name", type=str, default='mri-unified-run')
    parser.add_argument("--tensorboard-instance", type=str, default="", help="Tensorboard instance URI (optional).")


    parser.add_argument("--run-segmentation-pipeline", type=(lambda x: str(x).lower() == 'true'), default=False, help="Set to true to run segmentation preprocessing and training.")
    parser.add_argument("--seg-base-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to raw data for segmentation.")
    parser.add_argument("--seg-output-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path for segmentation TFRecords/processed data.")
    parser.add_argument("--seg-model-output-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to store the trained segmentation model.")
    parser.add_argument("--preprocess-seg-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Docker image URI for segmentation preprocessing.")
    parser.add_argument("--train-seg-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Docker image URI for segmentation training.")
    parser.add_argument("--seg-model-serving-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Serving container image URI for the segmentation model.")
    parser.add_argument("--seg-model-display-name", type=str, default="mri-segmentation-model")
    parser.add_argument("--seg-epochs", type=int, default=5)
    parser.add_argument("--seg-batch-size", type=int, default=4)
    parser.add_argument("--seg-image-height", type=int, default=512)
    parser.add_argument("--seg-image-width", type=int, default=512)


    parser.add_argument("--corrections-gcs-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to the corrections directory.")
    parser.add_argument("--original-sequences-gcs-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to the original sequences parent directory.")
    parser.add_argument("--accepted-predictions-gcs-dir", required=not "--compile-only" in sys.argv, help="GCS directory containing accepted without corrections files.") # New argument
    parser.add_argument("--classification-processed-data-gcs-path", type=str, required=not "--compile-only" in sys.argv, help="GCS path (gs://...) where processed classification data (patches/labels.csv) should be saved.")
    parser.add_argument("--preprocess-class-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Docker image URI for the preprocess-corrections component.")
    parser.add_argument("--original-processed-train-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to PREPROCESSED original training data (dir with labels.csv & patches).")
    parser.add_argument("--original-processed-val-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to PREPROCESSED original validation data.")
    parser.add_argument("--original-processed-test-dir", type=str, required=not "--compile-only" in sys.argv, help="GCS path to PREPROCESSED original test data.")
    parser.add_argument("--class-train-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Docker image URI for classification training.")
    parser.add_argument("--class-serving-image-uri", type=str, required=not "--compile-only" in sys.argv, help="Serving container image URI for the classification model.")
    parser.add_argument("--class-model-display-name", type=str, default="mri-classification-model-retrained")
    parser.add_argument("--class-epochs", type=int, default=50)
    parser.add_argument("--class-batch-size", type=int, default=32)
    parser.add_argument("--tuner-epochs", type=int, default=15)
    parser.add_argument("--max-tuner-trials", type=int, default=10)
    parser.add_argument("--patch-height", type=int, default=30)
    parser.add_argument("--patch-width", type=int, default=50)

    parser.add_argument("--evaluation-data-gcs-dir", type=str, required=not "--compile-only" in sys.argv)
    parser.add_argument("--evaluation-image-uri", type=str, required=not "--compile-only" in sys.argv)
    parser.add_argument("--production-model-uri", type=str, default="", required=False, help="Full Model Registry path (projects/.../models/name@alias) of the production model. Leave empty if none.")
    parser.add_argument("--evaluation-metric-name", type=str, default="accuracy")
    parser.add_argument("--evaluation-metric-threshold", type=float, default=0.0)

    parser.add_argument("--target-url", type=str,required=not "--compile-only" in sys.argv)

    args = parser.parse_args()


    pipeline_func = mri_unified_pipeline
    pipeline_filename = 'unified_pipeline.json' 

    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=pipeline_filename
    )
    print(f"Pipeline compiled to {pipeline_filename}")

    if not args.compile_only:
        tensorboard_resource_name = args.tensorboard_instance if args.tensorboard_instance else None

        aip.init(project=args.project_id, location=args.region)

        parameter_values={
                'project': args.project_id,
                'location': args.region,
                'pipeline_root': args.pipeline_root,
                'service_account': args.service_account,
                'tensorboard_instance': tensorboard_resource_name,

                'run_segmentation_pipeline': args.run_segmentation_pipeline,
                'seg_base_dir': args.seg_base_dir,
                'seg_output_dir': args.seg_output_dir,
                'seg_model_output_dir': args.seg_model_output_dir,
                'preprocess_seg_image_uri': args.preprocess_seg_image_uri,
                'train_seg_image_uri': args.train_seg_image_uri,
                'seg_model_serving_image_uri': args.seg_model_serving_image_uri,
                'seg_model_display_name': args.seg_model_display_name,
                'seg_epochs': args.seg_epochs,
                'seg_batch_size': args.seg_batch_size,
                'seg_image_height': args.seg_image_height,
                'seg_image_width': args.seg_image_width,

                'classification_processed_data_gcs_path': args.classification_processed_data_gcs_path,
                'corrections_gcs_dir': args.corrections_gcs_dir,
                'accepted_predictions_gcs_dir': args.accepted_predictions_gcs_dir,
                'original_sequences_gcs_dir': args.original_sequences_gcs_dir, 
                'preprocess_class_image_uri': args.preprocess_class_image_uri,
                'original_processed_train_dir': args.original_processed_train_dir, 
                'original_processed_val_dir': args.original_processed_val_dir,     
                'original_processed_test_dir': args.original_processed_test_dir,    
                'class_train_image_uri': args.class_train_image_uri,             
                'class_serving_image_uri': args.class_serving_image_uri,
                'class_model_display_name': args.class_model_display_name,          
                'class_epochs': args.class_epochs,                               
                'class_batch_size': args.class_batch_size,                         
                'tuner_epochs': args.tuner_epochs,                               
                'max_tuner_trials': args.max_tuner_trials,                         
                'patch_height': args.patch_height,                               
                'patch_width': args.patch_width,                                 
                'evaluation_data_gcs_dir': args.evaluation_data_gcs_dir,
                'evaluation_image_uri': args.evaluation_image_uri,
                'production_model_uri': args.production_model_uri, 
                'evaluation_metric_name': args.evaluation_metric_name,
                'evaluation_metric_threshold': args.evaluation_metric_threshold,
                'target_url': args.target_url,
            }

        job = aip.PipelineJob(
            display_name=args.display_name,
            template_path=pipeline_filename,
            pipeline_root=args.pipeline_root,
            parameter_values=parameter_values,
            enable_caching=None 
        )

        job.run(service_account=args.service_account if args.service_account else None)
        print(f"Pipeline job submitted. Display Name: {args.display_name}")