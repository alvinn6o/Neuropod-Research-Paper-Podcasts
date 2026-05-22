locals {
  name = var.project_name
  tags = merge(
    {
      Project = var.project_name
      App     = "neuropod"
    },
    var.tags
  )
}

resource "aws_ecr_repository" "api" {
  name                 = "${local.name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_ecr_repository" "worker" {
  name                 = "${local.name}-worker"
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = local.tags
}

resource "aws_s3_bucket" "audio" {
  bucket_prefix = "${local.name}-audio-"
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "audio" {
  bucket                  = aws_s3_bucket.audio.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${local.name}/worker"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_ecs_cluster" "this" {
  name = local.name
  tags = local.tags
}

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "api_task" {
  name               = "${local.name}-api-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.tags
}

resource "aws_iam_role" "worker_task" {
  name               = "${local.name}-worker-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.tags
}

resource "aws_ssm_parameter" "database_url" {
  name  = var.database_url_parameter_name
  type  = "SecureString"
  value = var.placeholder_secret_value
  tags  = local.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "anthropic_api_key" {
  name  = "/${local.name}/anthropic-api-key"
  type  = "SecureString"
  value = var.placeholder_secret_value
  tags  = local.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "openai_api_key" {
  name  = "/${local.name}/openai-api-key"
  type  = "SecureString"
  value = var.placeholder_secret_value
  tags  = local.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "elevenlabs_api_key" {
  name  = "/${local.name}/elevenlabs-api-key"
  type  = "SecureString"
  value = var.placeholder_secret_value
  tags  = local.tags

  lifecycle {
    ignore_changes = [value]
  }
}

data "aws_iam_policy_document" "runtime" {
  statement {
    sid = "ReadConfig"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
    ]
    resources = [
      aws_ssm_parameter.database_url.arn,
      aws_ssm_parameter.anthropic_api_key.arn,
      aws_ssm_parameter.openai_api_key.arn,
      aws_ssm_parameter.elevenlabs_api_key.arn,
    ]
  }

  statement {
    sid = "AudioStorage"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.audio.arn}/*"]
  }

  statement {
    sid       = "InvokeBedrockClaude"
    actions   = ["bedrock:InvokeModel"]
    resources = [var.bedrock_model_arn_pattern]
  }
}

resource "aws_iam_policy" "runtime" {
  name   = "${local.name}-runtime"
  policy = data.aws_iam_policy_document.runtime.json
  tags   = local.tags
}

resource "aws_iam_role_policy_attachment" "api_runtime" {
  role       = aws_iam_role.api_task.name
  policy_arn = aws_iam_policy.runtime.arn
}

resource "aws_iam_role_policy_attachment" "worker_runtime" {
  role       = aws_iam_role.worker_task.name
  policy_arn = aws_iam_policy.runtime.arn
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.api_task.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.api_image
      essential = true
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "NEUROPOD_AUTH_MODE", value = "cognito" },
        { name = "NEUROPOD_AUDIO_BACKEND", value = "s3" },
        { name = "NEUROPOD_S3_BUCKET", value = aws_s3_bucket.audio.bucket },
        { name = "NEUROPOD_GENERATE_AUDIO_ON_PIPELINE", value = "false" }
      ]
      secrets = [
        { name = "NEUROPOD_DATABASE_URL", valueFrom = aws_ssm_parameter.database_url.arn },
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_ssm_parameter.anthropic_api_key.arn },
        { name = "OPENAI_API_KEY", valueFrom = aws_ssm_parameter.openai_api_key.arn },
        { name = "ELEVENLABS_API_KEY", valueFrom = aws_ssm_parameter.elevenlabs_api_key.arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "api"
        }
      }
    }
  ])

  tags = local.tags
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.worker_task.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = var.worker_image
      essential = true
      environment = [
        { name = "NEUROPOD_AUDIO_BACKEND", value = "s3" },
        { name = "NEUROPOD_S3_BUCKET", value = aws_s3_bucket.audio.bucket },
        { name = "NEUROPOD_BEDROCK_OPERATOR", value = "true" }
      ]
      secrets = [
        { name = "NEUROPOD_DATABASE_URL", valueFrom = aws_ssm_parameter.database_url.arn },
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_ssm_parameter.anthropic_api_key.arn },
        { name = "OPENAI_API_KEY", valueFrom = aws_ssm_parameter.openai_api_key.arn },
        { name = "ELEVENLABS_API_KEY", valueFrom = aws_ssm_parameter.elevenlabs_api_key.arn }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.worker.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "worker"
        }
      }
    }
  ])

  tags = local.tags
}

resource "aws_cloudwatch_event_rule" "daily_refresh" {
  name                = "${local.name}-daily-refresh"
  description         = "Reference schedule for running Neuropod worker tasks."
  schedule_expression = "cron(0 13 * * ? *)"
  state               = "DISABLED"
  tags                = local.tags
}
