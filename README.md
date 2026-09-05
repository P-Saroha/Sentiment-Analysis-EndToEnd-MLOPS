#  My MLOps Learning Journey - Sentiment Analysis Pipeline

<div align="center">
  
[![Learning MLOps](https://img.shields.io/badge/Status-Learning%20MLOps-orange.svg)](https://github.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Learning-lightblue.svg)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-First%20Time-yellow.svg)](https://kubernetes.io)
[![AWS](https://img.shields.io/badge/AWS-Free%20Tier-green.svg)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experimenting-purple.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Learning-pink.svg)](https://dvc.org)

My Journey Building My First End-to-End MLOps Pipeline

*Documenting my learning process, challenges faced, and lessons learned*

</div>

---

## What I Built

### My First Cloud Application
**Previous Live Demo:** [http://a411ce29e025a4cdda5b02f5b7a541dd-1451511241.ap-south-1.elb.amazonaws.com:5000](http://a411ce29e025a4cdda5b02f5b7a541dd-1451511241.ap-south-1.elb.amazonaws.com:5000)

**Status:** Previously Live - App was successfully running on AWS EKS but deactivated to manage costs

### Local Development Version
**URL:** [http://localhost:5000](http://localhost:5000)

**How to run:** `docker run -p 5000:5000 sentiment-analysis-app:local`

---

## My Learning Journey

### What I Accomplished (with lots of trial and error!)

**CI/CD Pipeline (Biggest Challenge!)**
- Started with builds taking 12+ hours (frustrating!)
- Learned about optimization and got it down to 2-5 minutes
- Discovered GitHub Actions and conditional testing
- Lesson learned: Small optimizations compound into huge improvements

**First Time with AWS & Kubernetes**
- Set up my first EKS cluster (scary but exciting!)
- Learned about t3.micro limitations the hard way
- Got familiar with kubectl commands
- Challenge: Free tier constraints taught me resource optimization

### Technologies I Learned
| Component | Technology | My Experience |
|-----------|------------|---------------|
| **ML Framework** | Scikit-learn + NLTK | Learning basics of ML pipelines |
| **Experiment Tracking** | MLflow + DagHub | First time tracking experiments |
| **Data Versioning** | DVC + S3 | Discovered the importance of data versioning |
| **Containerization** | Docker | My first time containerizing an app |
| **Orchestration** | Kubernetes (EKS) | Steep learning curve but got it working! |
| **CI/CD** | GitHub Actions | Learned automation the hard way |
| **Monitoring** | Prometheus + Custom Metrics | Basic monitoring setup |
| **Cloud Provider** | AWS (Free Tier) | First cloud deployment experience |

---

## What I Learned Along the Way

### Key Learning Areas

1. **Building My First MLOps Pipeline**
   - Started with Cookiecutter template (confusing at first!)
   - Learned the importance of project structure
   - Discovered how different components work together

2. **DevOps Skills (Trial & Error)**
   - **CI/CD Optimization:** From 12+ hours to 2-5 minutes (lots of debugging!)
   - **Docker:** First time writing Dockerfiles and understanding containers
   - **AWS:** Learning AWS services one error message at a time

3. **Automation Journey**
   - Started doing everything manually
   - Gradually automated model training with DVC
   - Built CI/CD pipeline (broke it many times, learned from each mistake!)

4. **Cloud Learning Curve**
   - First EKS cluster setup (intimidating but rewarding)
   - Understanding AWS free tier limitations
   - Learning kubectl commands and Kubernetes concepts

5. **Monitoring & Observability**
   - Added basic Prometheus metrics
   - Learning about application health checks
   - Understanding the importance of monitoring

---

## My Pipeline Architecture (What I Built)

```
MLOPS PIPELINE ARCHITECTURE

GitHub Repository → CI/CD (GitHub Actions) → Docker Build → AWS ECR
        ↓                    ↓                                  ↓
     Code Push         Test & Validate                  Container Registry
                       (Conditional)                            ↓
                                                        AWS EKS Kubernetes Cluster
                                                                  ↓
                                                        Load Balancer → Production App

Data Pipeline (Local):
Raw Data (S3) → DVC Ingestion → Preprocessing → Feature Engineering 
    → Model Training (MLflow tracked) → Evaluation → Deployment
```

---

## How I Organized My Project

*Following the Cookiecutter Data Science template (learned about it during this project!)*

```
Sentiment-Analysis-EndToEnd-MLOPS/
├── Flask App
│   ├── flask_app/                    # First time building a web app!
│   │   ├── app.py                   # Main app with basic Prometheus metrics
│   │   └── templates/               # HTML files (learned some web dev)
│   └── Dockerfile                   # My first Dockerfile (lots of googling!)
│
├── ML Pipeline (Built step by step)
│   ├── src/
│   │   ├── data/                    # Data loading and cleaning
│   │   ├── features/                # Feature engineering (TF-IDF)
│   │   ├── model/                   # Model training and evaluation
│   │   └── connections/             # AWS S3 connections
│   ├── dvc.yaml                     # DVC pipeline (learned about ML pipelines)
│   ├── params.yaml                  # Hyperparameters file
│   └── run_complete_pipeline.py     # Runs everything (very satisfying!)
│
├── DevOps Configuration
│   ├── .github/workflows/ci.yaml    # GitHub Actions (broke this many times)
│   ├── deployment.yaml              # Kubernetes config (first K8s experience)
│   ├── requirements.txt             # All the packages I needed
│   ├── requirements-test.txt        # Optimized for CI (learned about this)
│   └── projectflow.txt              # My detailed notes and learning journey
│
├── Data & Models
│   ├── data/                        # Datasets (now versioned with DVC!)
│   ├── models/                      # Saved models
│   └── notebooks/                   # Jupyter notebooks for experiments
│
└── Testing & Environment
    ├── tests/                       # Basic tests (still improving)
    └── myenv/                       # My virtual environment
```

---

## Technology Stack

### Machine Learning
- **Framework:** Scikit-learn with optimized hyperparameters
- **NLP:** NLTK for text preprocessing and feature extraction
- **Model:** Logistic Regression with TF-IDF vectorization
- **Tracking:** MLflow with DagHub integration

### MLOps Pipeline
- **Data Versioning:** DVC with S3 backend
- **Experiment Tracking:** MLflow + DagHub
- **Pipeline Automation:** DVC pipelines with dependency management
- **Model Registry:** MLflow Model Registry for production models

### Cloud Infrastructure
- **Container Registry:** AWS ECR
- **Orchestration:** Amazon EKS (Kubernetes)
- **Storage:** S3 for data and artifacts
- **Networking:** Application Load Balancer
- **Monitoring:** Prometheus metrics integration

### DevOps
- **CI/CD:** GitHub Actions with multi-stage optimization
- **Containerization:** Docker with multi-stage builds
- **Infrastructure:** Kubernetes with production-ready configurations
- **Security:** IAM roles, secrets management, non-root containers

---

## What I Achieved (And Learned From Mistakes!)

### CI/CD Learning Journey
| Metric | When I Started | After Many Iterations | What I Learned |
|--------|----------------|----------------------|----------------|
| **Build Time** | 12+ hours (painful!) | 2-5 minutes | Small changes, big impact |
| **Test Strategy** | Ran everything always | Conditional testing | Smart automation saves time |
| **Deploy Process** | Manual (stressful) | Automated (satisfying!) | Automation reduces errors |

### My Model Results
- **Accuracy:** ~85% (not bad for my first ML model!)
- **Prediction Speed:** Pretty fast (<100ms)
- **What I Learned:** Hyperparameter tuning matters a lot

### Free Tier Challenges & Solutions
- **t3.micro Limitations:** Learned about resource constraints the hard way
- **Memory Issues:** Had to optimize Docker image size
- **Cost Awareness:** Everything running on AWS Free Tier (budget-friendly!)
- **Uptime:** App actually stays running (still amazed by this)

---

## Try It Yourself!

### 1. Running Locally (Easiest Way)

```bash
# Clone my project
git clone https://github.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.git
cd Sentiment-Analysis-EndToEnd-MLOPS

# Set up virtual environment (I use myenv)
python -m venv myenv
.\myenv\Scripts\activate   # Windows (I'm on Windows!)

# Install everything you need
pip install -r requirements.txt

# Run the complete pipeline (takes a few minutes)
python run_complete_pipeline.py

# Start the Flask app
cd flask_app
python app.py
```

### 2. Docker Version (If You Want to Try Containers)

```bash
# Build the image (first time took me a while to get this right!)
docker build -t sentiment-analysis-app .

# Run it
docker run -p 5000:5000 sentiment-analysis-app

# Open http://localhost:5000
```

### 3. My Cloud Deployment Process

*Note: This involves AWS setup and costs (though I use free tier)*

```bash
# You'll need AWS CLI and kubectl installed
# My deployment file (took many iterations to get right!)
kubectl apply -f deployment.yaml

# Check if it's working
kubectl get svc flask-app-service -n sentiment-analysis
```

---

## MLflow Experiment Tracking

### Live Dashboard
**URL:** [https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow](https://dagshub.com/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS.mlflow)

### Tracked Metrics
- Model accuracy, precision, recall, F1-score
- Training time and resource usage
- Hyperparameter optimization results
- Model artifact versioning

### Model Registry
- Production model versioning
- A/B testing capabilities
- Model deployment tracking
- Performance monitoring

---

## Data Pipeline (DVC)

### Pipeline Stages

1. **Data Ingestion** → Raw data from multiple sources
2. **Data Preprocessing** → Cleaning and normalization  
3. **Feature Engineering** → TF-IDF vectorization
4. **Model Building** → Training with hyperparameter optimization
5. **Model Evaluation** → Performance metrics and validation
6. **Model Registration** → MLflow model registry

### Data Versioning
- **Storage:** S3 bucket (`s3://newdatabucket2025`)
- **Versioning:** Git-like data tracking
- **Reproducibility:** Exact data lineage
- **Collaboration:** Team data sharing

---

## CI/CD Pipeline

### Advanced GitHub Actions Workflow

```yaml
Pipeline Stages:
├── 1. Quick Validation (30 seconds)
│   ├── Syntax checking
│   ├── Import validation
│   └── Basic linting
│
├── 2. Smart Testing (conditional)
│   ├── Unit tests
│   ├── Integration tests  
│   └── Model validation
│
├── 3. Docker Build & Push
│   ├── Multi-stage optimization
│   ├── Security scanning
│   └── ECR deployment
│
└── 4. EKS Deployment
    ├── Cluster validation
    ├── Rolling deployment
    └── Health checks
```

### Performance Optimizations

- **Conditional Testing:** Skip full tests on non-main branches
- **Dependency Caching:** Reuse installed packages
- **Parallel Stages:** Multiple jobs running simultaneously
- **Smart Triggers:** Build only when needed

---

## Configuration & Secrets

### Required GitHub Secrets

```bash
# MLflow & DagHub
CAPSTONE_TEST=your_dagshub_token

# AWS Configuration  
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
AWS_ACCOUNT_ID=your_account_id
ECR_REPOSITORY=sentiment-analysis-project
```

### Environment Configuration

- **Production:** Kubernetes secrets and ConfigMaps
- **Staging:** Environment-specific configurations
- **Development:** Local `.env` file support

---

## Monitoring & Observability

### Prometheus Metrics

```python
# Custom application metrics
app_request_count_total          # Total requests
app_request_latency_seconds      # Response time
model_prediction_count_total     # Prediction volume
app_health_status               # Application health
```

### Health Checks
- **Liveness:** Application responsiveness
- **Readiness:** Service availability  
- **Startup:** Container initialization

### Observability Stack
- **Metrics:** Prometheus + custom metrics
- **Logging:** Structured JSON logging
- **Monitoring:** Kubernetes native monitoring

---

## Production Features

### Security
- Non-root container execution
- Secrets management with Kubernetes
- IAM roles and least privilege access
- Network policies and security groups

### Scalability  
- Horizontal Pod Autoscaling (HPA)
- Cluster autoscaling with EKS
- LoadBalancer for traffic distribution
- Resource limits and requests

### Reliability
- Rolling deployments with zero downtime
- Health checks and self-healing
- Multi-AZ deployment for high availability
- Backup and disaster recovery

---

## Use Cases & Applications

### Enterprise Applications
- Customer feedback analysis
- Social media monitoring
- Product review analysis
- Support ticket classification

### Research & Development
- MLOps best practices demonstration
- Scalable ML infrastructure patterns
- CI/CD optimization techniques
- Cloud-native application design

### Learning & Education
- Complete MLOps pipeline example
- Production deployment patterns
- DevOps and ML integration
- Cloud engineering practices

---

## What I Want to Learn Next

### Things I'd Like to Try
- [ ] Better Monitoring: Maybe set up Grafana (heard it's cool)
- [ ] Model Improvements: Try different algorithms, maybe transformers
- [ ] Better Testing: Write more comprehensive tests
- [ ] Real-time Features: Stream processing sounds interesting
- [ ] Mobile App: Make this accessible on phones

### Skills I'm Still Developing
- [ ] Advanced Kubernetes: I barely scratched the surface
- [ ] Better DevOps: Infrastructure as Code, better monitoring
- [ ] ML Engineering: Model serving, A/B testing
- [ ] Security: Proper secrets management, security scanning
- [ ] Cost Optimization: Make this even more cost-effective

---

## Want to Help Me Learn?

I'm still learning, so if you see ways to improve this project, I'd love to learn from you!

### Areas Where I Could Use Help
- **Code Quality:** Better Python practices, cleaner code
- **Testing:** More comprehensive test coverage
- **Documentation:** Making this README even better
- **Performance:** Optimizing the ML pipeline
- **Security:** Best practices for production deployments

### Things I'm Working On
- Better error handling
- More efficient Docker builds
- Improved monitoring and logging
- Better documentation

*Feel free to open issues or suggest improvements - I'm here to learn!*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MLflow & DagHub** for experiment tracking platform
- **AWS** for robust cloud infrastructure
- **Kubernetes** community for orchestration excellence
- **Scikit-learn & NLTK** for ML capabilities
- **Docker** for containerization platform

---

## Common Issues I Faced (You Might Too!)

### Problems I Solved Along the Way

**Docker Issues:**
- *Problem:* Environment variables not working in container
- *Solution:* Learned about proper secrets management

**AWS Free Tier Challenges:**
- *Problem:* t3.micro nodes running out of memory
- *Solution:* Optimized resource requests and limits

**CI/CD Pain Points:**
- *Problem:* Builds taking forever
- *Solution:* Conditional testing and dependency optimization

**Kubernetes Learning Curve:**
- *Problem:* Pods not starting
- *Solution:* Understanding resource constraints and proper configurations

---

<div align="center">

## My Learning Stats

![GitHub stars](https://img.shields.io/github/stars/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS?style=social)
![GitHub forks](https://img.shields.io/github/forks/P-Saroha/Sentiment-Analysis-EndToEnd-MLOPS?style=social)

Built with lots of coffee, determination, and learning by [Parveen Saroha](https://github.com/P-Saroha)

*Documenting My MLOps Learning Journey*

</div>

---

**App was successfully deployed on AWS EKS but is currently deactivated to manage costs.**

*Still can't believe I got it working and deployed to production!*
