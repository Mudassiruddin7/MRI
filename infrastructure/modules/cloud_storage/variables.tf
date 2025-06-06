variable "bucket_suffix" {
  type        = string
  description = "The suffix for the Cloud Storage bucket name (e.g., 'raw-data')."
}

variable "region" {
  type        = string
  description = "The Google Cloud region."
}

variable "versioning_enabled" {
  type        = bool
  description = "Whether to enable versioning for the bucket."
  default     = true
}

variable "project_id" {
  type        = string
  description = "The Google Cloud project ID."
}