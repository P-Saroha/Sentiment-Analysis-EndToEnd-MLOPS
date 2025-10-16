# Monitor EKS Cluster and Nodegroup Status
# Run this script to check the status of your EKS infrastructure

Write-Host "🔍 Monitoring EKS Cluster: sentiment-analysis-cluster" -ForegroundColor Green
Write-Host "Region: ap-south-1" -ForegroundColor Green
Write-Host "=================================================="

# Activate environment
Write-Host "Activating environment..." -ForegroundColor Yellow
.\myenv\Scripts\Activate.ps1

# Check cluster status
Write-Host "`n📊 Cluster Status:" -ForegroundColor Cyan
$clusterStatus = aws eks --region ap-south-1 describe-cluster --name sentiment-analysis-cluster --query "cluster.status" --output text
Write-Host "Cluster: $clusterStatus" -ForegroundColor $(if ($clusterStatus -eq "ACTIVE") {"Green"} else {"Yellow"})

# Check nodegroup status
Write-Host "`n🖥️ Nodegroup Status:" -ForegroundColor Cyan
$nodegroupStatus = aws eks --region ap-south-1 describe-nodegroup --cluster-name sentiment-analysis-cluster --nodegroup-name freetier-workers --query "nodegroup.status" --output text
Write-Host "Nodegroup (freetier-workers): $nodegroupStatus" -ForegroundColor $(if ($nodegroupStatus -eq "ACTIVE") {"Green"} else {"Yellow"})

# If nodegroup is active, check nodes
if ($nodegroupStatus -eq "ACTIVE") {
    Write-Host "`n🚀 Kubectl Node Status:" -ForegroundColor Cyan
    kubectl get nodes
    
    Write-Host "`n📋 Cluster Info:" -ForegroundColor Cyan
    kubectl cluster-info
    
    Write-Host "`n✅ EKS cluster is ready for deployment!" -ForegroundColor Green
    Write-Host "You can now push to main branch to trigger CI/CD deployment" -ForegroundColor Green
} else {
    Write-Host "`n⏳ Nodegroup is still being created..." -ForegroundColor Yellow
    Write-Host "This typically takes 5-10 minutes. Please wait and run this script again." -ForegroundColor Yellow
}

Write-Host "`n=================================================="
$currentTime = Get-Date
Write-Host "Monitoring completed at $currentTime" -ForegroundColor Green