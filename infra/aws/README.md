# AWS Reference Architecture

This directory is a reference deployment path, not evidence of a live deployment. I do not keep Neuropod running on AWS because the local/Docker demo is enough for portfolio review, and Bedrock Knowledge Bases would require OpenSearch Serverless with an idle cost floor.

## Service Mapping

- **ECR**: store API and worker container images.
- **ECS Fargate**: run the long paper pipeline as a task; optionally run the API as a service.
- **Lambda container or ECS service**: run FastAPI for web/API traffic.
- **IAM roles**: give API/worker least-privilege access to S3, SSM, CloudWatch, and Bedrock.
- **Parameter Store**: store provider configuration and API keys as SecureString values.
- **S3**: store generated audio and optionally cached PDFs.
- **Postgres + pgvector**: store users, papers, chunks, jobs, and episodes.
- **EventBridge**: optional scheduled refresh that triggers a Fargate worker task.
- **Bedrock Claude**: optional script generation provider through `InvokeModel`.
- **Bedrock Knowledge Bases**: evaluated alternative managed RAG path; not the default because OpenSearch Serverless is expensive when idle.
- **Anthropic/OpenAI direct APIs**: direct provider path for script generation, embeddings, and TTS fallback.

## Deployment Stance

The app has been built to make this AWS path straightforward, but the repo should be described as **AWS-ready/reference-IaC**, not currently hosted. The Terraform scaffold focuses on the pieces that are useful to discuss in an interview: container registries, task roles, Parameter Store, S3, Bedrock permissions, and scheduled worker design.

## Practical Production Path

1. Build and push `Dockerfile` and `Dockerfile.worker` images to ECR.
2. Apply the Terraform scaffold after filling in network/database variables.
3. Store provider keys in Parameter Store.
4. Run FastAPI on Lambda container or ECS behind API Gateway/ALB.
5. Run script generation and optional TTS as ECS Fargate tasks.
6. Keep Bedrock Knowledge Bases as an enterprise option only when managed retrieval is worth the OpenSearch Serverless baseline cost.

The current application defaults to SQLite and local disk so it remains free and easy to demo on a laptop.
