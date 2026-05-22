# Terraform Scaffold

This Terraform is a reference scaffold for interview discussion and future deployment. It has not been applied as part of the local portfolio demo.

It creates or models:

- ECR repositories for API and worker images
- ECS cluster and Fargate task definitions
- IAM task/execution roles
- SSM SecureString parameters for provider keys
- S3 audio bucket
- CloudWatch log groups
- Disabled EventBridge schedule for worker runs
- Bedrock `InvokeModel` permission for Claude through IAM

Before applying, provide real VPC/subnet/security group/database values and review names, regions, and cost settings.
