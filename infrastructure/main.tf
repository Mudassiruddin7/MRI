# Configure the Google Cloud provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Create the raw data bucket
module "raw_data_bucket" {
  source          = "./modules/cloud_storage"
  bucket_suffix   = "raw-data"
  region          = var.region
  project_id      = var.project_id
}

# Create the processed data bucket
module "processed_data_bucket" {
  source          = "./modules/cloud_storage"
  bucket_suffix   = "processed-data"
  region          = var.region
  project_id      = var.project_id
}

module "pipeline_data_bucket" {
  source          = "./modules/cloud_storage"
  bucket_suffix   = "pipeline"
  region          = var.region
  project_id      = var.project_id
}

module "deployment_data_bucket" {
  source          = "./modules/cloud_storage"
  bucket_suffix   = "deployment"
  region          = var.region
  project_id      = var.project_id
}
module "build_data_bucket" {
  source          = "./modules/cloud_storage"
  bucket_suffix   = "build-artifacts"
  region          = var.region
  project_id      = var.project_id
}