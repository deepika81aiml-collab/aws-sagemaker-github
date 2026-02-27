variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-east-1"  
}



variable "project_name" {
  description = "project name "
  type = string
  default="sagemaker-basic-poc"

}