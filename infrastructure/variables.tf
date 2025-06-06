variable "project_id" {
  type        = string
  description = "The Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The Google Cloud region."
  default     = "us-central1"
}