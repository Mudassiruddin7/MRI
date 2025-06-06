resource "google_storage_bucket" "bucket" {
  name                        = "${var.project_id}-${var.bucket_suffix}"
  location                    = var.region
  uniform_bucket_level_access = true
  storage_class               = "STANDARD"

  versioning {
    enabled = var.versioning_enabled
  }
}