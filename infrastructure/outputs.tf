output "raw_data_bucket_url" {
  value = module.raw_data_bucket.bucket_url
}

output "raw_data_bucket_name" {
    value = module.raw_data_bucket.bucket_name
}

output "processed_data_bucket_url" {
  value = module.processed_data_bucket.bucket_url
}
output "processed_data_bucket_name" {
    value = module.processed_data_bucket.bucket_name
}