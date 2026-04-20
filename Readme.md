# Kubeflow ML Orchestrator

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Kubeflow](https://img.shields.io/badge/Kubeflow-Pipelines_v2-blue)](https://www.kubeflow.org/)
[![Minikube](https://img.shields.io/badge/Minikube-Local_K8s-green)](https://minikube.sigs.k8s.io/)

## 📖 Overview
This project implements an end-to-end, fully parameterized Machine Learning pipeline using **Kubeflow Pipelines (KFP V2)** deployed on a local **Minikube** Kubernetes cluster. It orchestrates the ingestion, preprocessing, training, and evaluation of predictive models using the Breast Cancer Wisconsin dataset.

The pipeline is designed to be highly reproducible and scalable, dynamically routing data to one of three optimized models (SVM, Random Forest, or Neural Network) based on user-defined parameters injected directly through the Kubeflow UI.

## 🚀 Features
* **Scalable Orchestration:** Built to run on Kubernetes via Minikube, enabling concurrent parallel pipeline executions and robust resource management.
* **Fully Parameterized UI:** The pipeline accepts runtime parameters via the Kubeflow UI, including:
  * Model Type Selection (`SVM`, `RF`, `NN`)
  * Train/Test Split Size
  * Scaler Type (`StandardScaler` vs `MinMaxScaler`)
  * Model-specific Hyperparameters (e.g., `n_estimators`, `max_iter`)
  * Random Seed (for guaranteed reproducibility)
* **Multi-Model Optimization:** * **Neural Network:** Optimized using `RandomizedSearchCV`.
  * **Random Forest:** Enhanced with `SelectFromModel` automated feature selection.
  * **Support Vector Machine (SVM):** Tuned using a `DEAP` Genetic Algorithm.
* **Automated Artifact Management:** Utilizes Minio object storage for seamlessly passing datasets, trained `.joblib` models, and evaluation metrics between isolated, containerized pipeline components.

## 📂 Project Structure
```text
kubeflow_project/
├── components/
│   ├── data_ingestion.py    # Loads dataset and formats target variables
│   ├── preprocessing.py     # Handles scaling and Train/Test splitting
│   ├── train_svm.py         # SVM training with Genetic Algorithm optimization
│   ├── train_rf.py          # Random Forest training with Feature Selection
│   ├── train_nn.py          # Neural Network training with RandomizedSearchCV
│   └── evaluate.py          # Calculates Accuracy, F1, Precision, Recall & Confusion Matrix
├── pipelines/
│   └── pipeline_v1.py       # Master DAG utilizing KFP dsl.If conditions
├── ultimate_master_pipeline.yaml # Compiled KFP V2 specification file
├── requirements.txt         # Local development dependencies
└── README.md                # Project documentation
```

## 🛠️ Tech Stack
* **Orchestration:** Kubeflow Pipelines (KFP V2), Argo Workflows
* **Infrastructure:** Minikube, Docker, Kubernetes
* **Machine Learning:** Scikit-Learn, Pandas, Numpy 1.26.4
* **Optimization:** DEAP (Distributed Evolutionary Algorithms in Python)

## ⚙️ How to Run Locally

### 1. Start the Kubernetes Cluster
Ensure Docker is running, then start Minikube with sufficient resources for ML workloads:
```bash
minikube start --driver=docker --cpus=4 --memory=8192
```

### 2. Access the Kubeflow Dashboard
Port-forward the UI service to your local machine to access the dashboard:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Navigate to `http://localhost:8080` in your web browser.

### 3. Compile the Pipeline
If you modify the Python component files, recompile the YAML specification:
```bash
python3 pipelines/pipeline_v1.py
```

### 4. Execute in Kubeflow
1. Open the Kubeflow UI and navigate to **Pipelines** -> **Upload Pipeline**.
2. Upload the compiled `ultimate_master_pipeline.yaml` file.
3. Click **Create Run**.
4. Enter your desired hyperparameters, scaling strategies, and random seed.
5. Click **Start** and monitor the DAG execution!

## 📊 Results & Reproducibility
The pipeline successfully guarantees deterministic execution; running the pipeline multiple times with the exact same `random_seed` yields identical data splits and model performance. During experimentation, the **Neural Network** architecture yielded the highest performance, achieving **~97.3% Accuracy** and F1-score on the testing split.
```
