# GitHub Secrets Configuration Template

This document lists all the GitHub secrets required for the CI/CD pipeline to work properly.

⚠️ **SECURITY WARNING**: Never commit actual secrets to Git! Use this template to configure secrets in GitHub.

## Required Secrets

Add these secrets in your GitHub repository:
**Settings → Secrets and variables → Actions → Repository secrets**

### 1. DagHub/MLflow Credentials
```
Name: CAPSTONE_TEST
Value: [YOUR_DAGSHUB_TOKEN_HERE]
Description: DagHub token for MLflow tracking
```

### 2. AWS Credentials
```
Name: AWS_ACCESS_KEY_ID
Value: [YOUR_AWS_ACCESS_KEY_HERE]
Description: AWS access key for S3 and ECR access

Name: AWS_SECRET_ACCESS_KEY
Value: [YOUR_AWS_SECRET_KEY_HERE]
Description: AWS secret key for S3 and ECR access
```

### 3. AWS Infrastructure Details
```
Name: AWS_ACCOUNT_ID
Value: [YOUR_AWS_ACCOUNT_ID]
Description: Your AWS account ID

Name: ECR_REPOSITORY
Value: sentiment-analysis-app
Description: ECR repository name for Docker images

Name: EKS_CLUSTER_NAME
Value: flask-app-cluster
Description: EKS cluster name for deployment
```

## Security Best Practices

1. **Never commit secrets**: Use GitHub Secrets or environment variables
2. **Rotate credentials**: Regularly update AWS access keys
3. **Least privilege**: Grant minimal required permissions
4. **Monitor access**: Review AWS CloudTrail logs regularly
5. **Use IAM roles**: Prefer IAM roles over access keys when possible