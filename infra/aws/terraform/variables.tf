variable "project_name" {
  description = "Name prefix for Neuropod resources."
  type        = string
  default     = "neuropod"
}

variable "region" {
  description = "AWS region for the reference deployment."
  type        = string
  default     = "us-east-1"
}

variable "api_image" {
  description = "API container image URI. Usually the ECR API image URL after build/push."
  type        = string
  default     = "replace-with-api-image-uri"
}

variable "worker_image" {
  description = "Worker container image URI. Usually the ECR worker image URL after build/push."
  type        = string
  default     = "replace-with-worker-image-uri"
}

variable "database_url_parameter_name" {
  description = "SSM parameter name containing the Postgres connection URL."
  type        = string
  default     = "/neuropod/database-url"
}

variable "placeholder_secret_value" {
  description = "Initial placeholder for SecureString parameters. Replace out of band before real use."
  type        = string
  default     = "replace-me"
  sensitive   = true
}

variable "bedrock_model_arn_pattern" {
  description = "ARN pattern for Bedrock models/inference profiles the worker may invoke."
  type        = string
  default     = "arn:aws:bedrock:*::foundation-model/*"
}

variable "tags" {
  description = "Additional tags."
  type        = map(string)
  default     = {}
}
