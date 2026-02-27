output "s3_bucket_name" {
  value       =  aws_s3_bucket.logs_bucket.bucket
  description = "Name of the S3 bucket for storing logs"
}

output "sagemaker_execution_role_arn" {
  value       = aws_iam_role.sagemaker_execution_role.arn
  description = "ARN of the SageMaker execution role"
}

output "vpc_id" {
  value       = aws_vpc.main.id
  description = "ID of the VPC"
}

output "subnet_id" {
  value       = aws_subnet.public_subnet.id
  description = "ID of the public subnet"
}

output "security_group_id" {
  value       = aws_security_group.sagemaker_sg.id
  description = "ID of the SageMaker security group"
}

output "notebook_instance_name" {
  value       = aws_sagemaker_notebook_instance.poc_notebook.name
  description = "Name of the SageMaker notebook instance"
}