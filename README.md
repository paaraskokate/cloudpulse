# CloudPulse 📡
### ML-Based Predictive Workload Monitoring & Intelligent Resource Allocation

CloudPulse is a machine learning-based cloud workload monitoring and resource allocation system developed for the Laboratory Assessment of Advanced Cloud Computing (LABCA).

The project forecasts CPU utilization for the next five-minute monitoring interval using a Random Forest regression model, generates bounded resource capacity recommendations, and compares three allocation strategies: Fixed, Reactive, and Predictive.

The system integrates Amazon EC2, Amazon CloudWatch, Python-based machine learning, and a Streamlit dashboard to demonstrate how predictive analytics can support more efficient cloud resource utilization.

> **Core principle:** Real AWS telemetry and synthetic experimental data are kept separate. Synthetic data is used for offline testing and experimentation, while real workload analysis relies on metrics retrieved from Amazon CloudWatch. Missing AWS data is treated as an error rather than fabricated.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [How It Works](#-how-it-works)
- [Installation and Setup](#-installation-and-setup)
- [Running the Project](#-running-the-project)
- [Allocation Strategies](#-allocation-strategies)
- [Safety and Reliability](#-safety-and-reliability)
- [Evaluation and Testing](#-evaluation-and-testing)
- [Troubleshooting](#-troubleshooting)
- [Security and Cost Considerations](#-security-and-cost-considerations)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [References](#-references)

---

## 🎯 Project Overview

Cloud resource management involves balancing performance, availability, and infrastructure costs.

Traditional resource allocation approaches have several limitations:

- **Fixed allocation:** Resources remain allocated even when workload demand is low, potentially wasting money.
- **Reactive allocation:** Resources are adjusted only after utilization crosses a threshold, which can delay responses to sudden workload changes.
- **Predictive allocation:** Forecasts upcoming workload demand to support proactive capacity recommendations.

CloudPulse explores the third approach by combining time-series feature engineering, machine learning, controller safeguards, and policy evaluation.

The project is designed as an educational prototype demonstrating the integration of cloud monitoring and predictive resource management.

## 🎯 Objectives

1. Retrieve real CPU utilization metrics from Amazon CloudWatch.
2. Process time-series data to generate features for machine learning.
3. Predict CPU utilization for the next five-minute interval using Random Forest regression.
4. Recommend resource capacity within predefined operational limits.
5. Implement fallback mechanisms and cooldown rules to avoid unsafe or excessive scaling decisions.
6. Compare fixed, reactive, and predictive resource allocation policies.
7. Provide a dashboard for monitoring workload history, forecasts, recommendations, and evaluation results.
8. Maintain an auditable workflow that distinguishes real telemetry from synthetic experimental data.

---

## 🏗️ System Architecture

The following diagram illustrates the flow of workload telemetry through the CloudPulse system.

```text
                 ┌─────────────────────────┐
                 │      Amazon EC2         │
                 │    Test Workload        │
                 └────────────┬────────────┘
                              │
                              │ CPU Metrics
                              ▼
                 ┌─────────────────────────┐
                 │    Amazon CloudWatch    │
                 │   Workload Monitoring   │
                 └────────────┬────────────┘
                              │
                              │ boto3 API
                              ▼
                 ┌─────────────────────────┐
                 │   cloud/aws_metrics.py  │
                 │   Metric Collection     │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      ml/features.py     │
                 │  Time-Series Features   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     ml/train_model.py   │
                 │  Random Forest Model    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ controller/allocator.py │
                 │ Capacity Recommendations│
                 └────────────┬────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
     ┌────────────────────┐      ┌────────────────────┐
     │ Policy Evaluation  │      │ Safety and Audit   │
     │ Fixed / Reactive / │      │ Fallback / Cooldown│
     │ Predictive         │      │ Decision Logging   │
     └──────────┬─────────┘      └──────────┬─────────┘
                └─────────────┬─────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │       app.py            │
                 │  Streamlit Dashboard    │
                 └─────────────────────────┘
```

### Architecture Components

| Component | Responsibility |
|---|---|
| Amazon EC2 | Provides the workload whose CPU utilization is monitored. |
| Amazon CloudWatch | Collects and exposes CPU utilization metrics. |
| AWS Metrics Adapter | Retrieves real telemetry using boto3 and stores the resulting dataset with provenance metadata. |
| Feature Engineering | Generates lag, rolling-window, and calendar features from time-series data. |
| Random Forest Model | Forecasts CPU utilization for the next monitoring interval. |
| Resource Allocator | Converts forecasts into bounded capacity recommendations. |
| Policy Evaluation | Compares fixed, reactive, and predictive allocation strategies. |
| Streamlit Dashboard | Displays monitoring data, forecasts, recommendations, and evaluation results. |
| Audit Logging | Records controller decisions and dry-run activity for inspection. |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application, ML pipeline, and controller logic |
| Amazon EC2 | Cloud workload environment |
| Amazon CloudWatch | CPU utilization monitoring |
| boto3 | Python integration with AWS services |
| pandas | Data processing and time-series manipulation |
| scikit-learn | Random Forest regression and model evaluation |
| Streamlit | Interactive monitoring dashboard |
| joblib | Saving and loading trained models |
| Python unittest | Automated testing |
| AWS CLI | Optional AWS configuration and administration |

---

## 📂 Repository Structure

The repository is organized into modules to separate AWS integration, machine learning, resource allocation, evaluation, and dashboard functionality.

```text
cloudpulse/
│
├── app.py
│
├── cloud/
│   └── aws_metrics.py
│
├── controller/
│   ├── allocator.py
│   └── policy.py
│
├── data/
│   ├── generate_synthetic_data.py
│   ├── synthetic_cpu.csv
│   ├── real_cpu.csv
│   └── audit_log.jsonl
│
├── evaluation/
│   └── compare_policies.py
│
├── ml/
│   ├── features.py
│   └── train_model.py
│
├── models/
│   ├── rf_cpu.joblib
│   ├── rf_real.joblib
│   └── training_report_real.json
│
├── tests/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*Note: The diagram shows the main files and representative generated artifacts. The exact contents of individual folders may differ depending on which generated files and model artifacts are present in your checkout.*

### File and Directory Descriptions

#### 1. `app.py`

The main entry point of the application.

Responsibilities include:

- Launching the Streamlit dashboard.
- Displaying CPU utilization history and forecast information.
- Presenting capacity recommendations.
- Showing policy evaluation results.
- Exposing controller safety information and audit records.
- Providing controls for retrieving real AWS metrics and recording dry-run decisions.

#### 2. `cloud/`

Contains code responsible for integrating the application with AWS monitoring services.

**`cloud/aws_metrics.py`**

Collects CPU utilization metrics from Amazon CloudWatch using boto3.

Its responsibilities include:

- Retrieving historical CPU utilization.
- Supporting configurable EC2 instance IDs, AWS regions, and collection windows.
- Exporting real telemetry into a CSV dataset.
- Preserving collection metadata for data provenance.
- Reporting errors when usable AWS telemetry is unavailable.

#### 3. `controller/`

Contains the resource allocation and decision-making logic.

**`controller/allocator.py`**

Converts CPU utilization forecasts into resource capacity recommendations.

The allocator incorporates safeguards such as:

- Capacity bounds.
- High-CPU fallback behaviour.
- Cooldown periods.
- Sustained-low-utilization checks for scale-down decisions.
- Dry-run operation and decision auditing.

**`controller/policy.py`**

Contains resource allocation policy logic used to compare different approaches to workload management.

The policies represent fixed, reactive, and predictive allocation strategies.

#### 4. `data/`

Stores datasets and generated runtime data used by the application.

**`data/generate_synthetic_data.py`**

Generates synthetic CPU utilization time-series data for offline development, testing, and experimentation.

**`data/synthetic_cpu.csv`**

Contains synthetic workload data generated by the offline data-generation pipeline, when that artifact has been created.

**`data/real_cpu.csv`**

Stores CPU utilization data collected from Amazon CloudWatch.

This dataset is intended for real AWS workload analysis and model training.

**`data/audit_log.jsonl`**

Stores controller decision records in JSON Lines format when audit logging has been performed.

These records help distinguish dry-run recommendations from actual cloud operations.

#### 5. `evaluation/`

Contains tools for evaluating resource allocation strategies.

**`evaluation/compare_policies.py`**

Compares fixed, reactive, and predictive allocation policies on the same input samples.

The evaluation can examine:

- Average allocated capacity.
- Estimated instance-hours.
- Estimated infrastructure cost.
- Fallback events.
- Differences in resource allocation behaviour.

Cost values generated by the evaluation are estimates, not AWS billing statements.

#### 6. `ml/`

Contains the machine learning pipeline.

**`ml/features.py`**

Prepares time-series features for forecasting, including lagged values, rolling statistics, and calendar-related features.

Feature engineering must preserve temporal ordering to avoid introducing future information into model inputs.

**`ml/train_model.py`**

Trains and evaluates the Random Forest regression model.

Its responsibilities include:

- Loading the selected dataset.
- Preparing forecasting features.
- Using chronological training and testing splits.
- Evaluating predictions using MAE and RMSE.
- Saving the trained model.
- Optionally exporting a training report.

#### 7. `models/`

Stores trained model artifacts and evaluation reports generated by the training pipeline.

| File | Purpose |
|---|---|
| `rf_cpu.joblib` | Example saved Random Forest model trained on the selected offline dataset, if generated. |
| `rf_real.joblib` | Saved Random Forest model trained using real AWS telemetry, if generated. |
| `training_report_real.json` | Machine-readable training and evaluation results for the real-data model, if generated. |

These files are generated artifacts rather than source code. Their presence depends on whether the corresponding training commands have been executed and whether the artifacts are tracked in Git.

#### 8. `tests/`

Contains automated tests for verifying application behaviour.

Tests may cover:

- Capacity-bound enforcement.
- High-CPU fallback logic.
- Cooldown behaviour.
- Dry-run safety.
- Missing resource configuration.
- Error handling for empty AWS telemetry.

Run the test suite using Python's `unittest` framework.

#### 9. `requirements.txt`

Lists the Python dependencies required to run the application.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

#### 10. `.env.example`

Provides an example configuration template for supported environment variables.

Use it as a reference when configuring the application. Do not store real credentials, secret keys, or other sensitive values in this file.

#### 11. `.gitignore`

Specifies files and directories that Git should ignore, such as local virtual environments, caches, generated files, and local secrets.

#### 12. `README.md`

Documents the project, architecture, setup process, execution commands, implementation details, and limitations.

---

## ⚙️ How It Works

CloudPulse follows a sequence of data collection, feature engineering, forecasting, decision-making, and evaluation.

### Step 1: Workload Monitoring

An EC2 instance produces CPU utilization metrics that are collected by Amazon CloudWatch.

### Step 2: Data Collection

The AWS adapter retrieves historical metrics through boto3 and stores the results in a CSV file.

Collection metadata is preserved so that the origin and timing of the data can be inspected.

### Step 3: Feature Engineering

The machine learning pipeline transforms the time-series data into features such as:

- Previous CPU utilization values.
- Rolling-window statistics.
- Time-based calendar features.

The features are used to estimate CPU utilization at the next five-minute monitoring interval.

### Step 4: Forecasting

A Random Forest regression model predicts upcoming CPU utilization.

Model evaluation uses chronological train/test splits because time-series observations must remain ordered.

### Step 5: Capacity Recommendation

The controller uses the forecast to recommend resource capacity.

The recommendation is constrained by safety limits and may fall back to a conservative decision when utilization crosses a configured threshold.

### Step 6: Policy Comparison

The evaluation module compares fixed, reactive, and predictive resource allocation policies using comparable input samples.

### Step 7: Dashboard Visualization

The Streamlit dashboard presents the monitoring history, forecast information, capacity recommendations, policy comparisons, and controller audit information.

---

## 🚀 Installation and Setup

### Prerequisites

Before running CloudPulse, ensure you have:

- Python installed.
- Git installed.
- Internet connectivity for dependency installation.
- An AWS account and appropriate credentials if you intend to use real AWS telemetry.
- An EC2 instance publishing CPU metrics to CloudWatch for the real-data workflow.

AWS access is not required for the synthetic-data workflow.

### Step 1: Clone the Repository

```bash
git clone https://github.com/paaraskokate/cloudpulse.git
cd cloudpulse
```

### Step 2: Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure AWS Credentials

For real AWS telemetry collection, configure an AWS profile using the AWS CLI:

```bash
aws configure --profile cloudpulse
```

Use an IAM identity with only the permissions required for the operations you intend to perform.

Do not hardcode AWS access keys or secret keys into the source code or commit credential files to the repository.

---

## ▶️ Running the Project

Run the following commands from the repository root with the virtual environment activated.

### Option 1: Run the Synthetic-Data Workflow

Generate synthetic CPU utilization data:

```bash
python -m data.generate_synthetic_data --days 3 --out data/synthetic_cpu.csv
```

Train a Random Forest model:

```bash
python -m ml.train_model --input data/synthetic_cpu.csv --model-out models/rf_cpu.joblib
```

Compare allocation policies:

```bash
python -m evaluation.compare_policies --input data/synthetic_cpu.csv --model models/rf_cpu.joblib
```

This workflow supports offline experimentation without requiring AWS credentials.

**Important:** Results obtained from synthetic data must be labelled as synthetic and must not be presented as measurements from a real AWS workload.

### Option 2: Collect Real AWS Telemetry

Replace `<your-instance-id>` and `<your-aws-region>` with the EC2 instance ID and AWS region you are authorized to monitor.

```bash
python -m cloud.aws_metrics --instance-id <your-instance-id> --region <your-aws-region> --hours 6
```

Train the model using the collected real telemetry:

```bash
python -m ml.train_model --input data/real_cpu.csv --model-out models/rf_real.joblib --report-out models/training_report_real.json
```

Evaluate the allocation policies:

```bash
python -m evaluation.compare_policies --input data/real_cpu.csv --model models/rf_real.joblib
```

The real-data workflow requires valid AWS credentials, appropriate IAM permissions, and available CloudWatch datapoints.

If the collection command returns no usable metrics, verify the instance ID, region, collection window, permissions, and metric availability.

### Option 3: Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that address in your browser to access the dashboard.

Select the intended data mode and verify the telemetry source before interpreting displayed forecasts or evaluation results.

### Option 4: Run Automated Tests

```bash
python -m unittest discover -s tests -t . -v
```

The test suite verifies important application behaviours, including controller safeguards and error handling.

The exact test count may change as the codebase evolves.

---

## 📊 Allocation Strategies

CloudPulse compares three approaches to resource allocation.

| Strategy | Description | Main Consideration |
|---|---|---|
| Fixed Allocation | Maintains a constant resource capacity. | Simple to operate, but may waste resources during idle periods or lack capacity during workload spikes. |
| Reactive Allocation | Adjusts capacity in response to observed utilization thresholds. | Responds to actual demand, but may react after a workload change has occurred. |
| Predictive Allocation | Uses forecast CPU utilization to recommend capacity changes. | May anticipate demand, but depends on forecast quality and must be constrained by safety rules. |

The evaluation compares the strategies using identical chronological samples to make the results more comparable.

The predictive policy does not bypass controller safeguards. Recommendations remain subject to configured bounds, fallback behaviour, and cooldown rules.

---

## 🛡️ Safety and Reliability

CloudPulse includes safeguards intended to make resource recommendations more controlled and auditable.

### Capacity Bounds

Recommendations are constrained to configured minimum and maximum capacity values.

### High-CPU Fallback

A fallback mechanism can respond when CPU utilization crosses the configured safety threshold.

### Cooldown Period

A cooldown prevents decisions from triggering repeated capacity changes too frequently.

### Sustained-Low-Utilization Check

Scale-down decisions require low utilization to persist for a configured duration rather than relying on a single observation.

### Dry-Run by Default

The default workflow records recommendations without applying AWS scaling operations.

A dry-run recommendation is not evidence that an AWS resource was actually scaled.

### Audit Logging

Controller decisions can be recorded for later inspection.

### Time-Series Evaluation

Chronological train/test splits preserve the temporal order of observations and reduce the risk of data leakage.

### No Fabricated AWS Telemetry

An empty AWS response is treated as an error rather than an opportunity to generate fake real-world measurements.

---

## 🧪 Evaluation and Testing

CloudPulse evaluates forecasting performance and resource allocation behaviour.

### Forecasting Metrics

The model can be evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual CPU utilization.
- **Root Mean Squared Error (RMSE):** Measures prediction error while assigning greater weight to larger errors.
- **Persistence Baseline:** Provides a simple reference forecast based on the most recent available observation.

Comparing the Random Forest model with a baseline helps determine whether the model adds predictive value.

### Policy Evaluation

The policy-comparison module evaluates fixed, reactive, and predictive allocation using comparable samples.

Potential evaluation outputs include average allocated capacity, instance-hours, estimated costs, and fallback events.

### Interpreting Results

Results depend on the data used for training and evaluation.

A short dataset collected from a mostly idle instance may produce very low prediction errors without demonstrating that the model will perform equally well under variable or production-scale workloads.

Real-data results and synthetic experiment results should always be reported separately.

Cost calculations are estimates based on the evaluation assumptions and should not be interpreted as actual AWS charges.

---

## 🔧 Troubleshooting

| Problem | Possible Cause | Suggested Action |
|---|---|---|
| AWS profile not found | Incorrect or missing AWS profile configuration. | Verify the profile name and configure credentials using the AWS CLI. |
| No CloudWatch datapoints returned | Incorrect instance ID or region, insufficient history, or unavailable metrics. | Verify the instance ID, region, time window, permissions, and metric availability. |
| AWS `AccessDenied` error | Insufficient IAM permissions. | Review the IAM policy and grant only the permissions required by the selected operation. |
| Unexpected forecast values | Model trained on a different dataset or insufficient training data. | Inspect the selected model, dataset, training report, and persistence baseline. |
| Dashboard shows stale data | Metrics have not been refreshed. | Trigger a data refresh and inspect the displayed collection timestamp. |
| Dependency installation fails | Incompatible Python or package versions. | Verify the Python version and recreate the virtual environment if needed. |
| Insufficient data for training | Too few valid time-series observations. | Collect additional metrics before training or evaluation. |

---

## 🔐 Security and Cost Considerations

CloudPulse is an educational prototype and should not be treated as a production-ready autoscaling service.

Follow these precautions when running the project:

- Never commit AWS access keys, secret keys, or credential files.
- Use IAM identities with least-privilege permissions.
- Prefer read-only AWS access during development and evaluation.
- Keep live scaling disabled unless it has been deliberately configured and reviewed.
- Use AWS billing alerts or budgets before creating cloud resources.
- Remember that AWS Free Tier eligibility does not mean every resource or usage pattern is free.
- Terminate test EC2 instances after use.
- Remove unused Auto Scaling Groups, launch templates, EBS volumes, and public IP resources.
- Review the AWS Billing console after completing experiments.
- Avoid publishing screenshots that expose sensitive account, credential, or resource information.

---

## ⚠️ Limitations and Future Improvements

### Current Limitations

- The current forecasting workflow focuses on CPU utilization rather than a complete set of workload metrics.
- Model performance depends on the volume, quality, and diversity of the collected data.
- A short period of mostly idle CPU utilization is insufficient to validate performance under diverse workload conditions.
- Synthetic experiments cannot substitute for evaluation against real workloads.
- Policy cost estimates depend on the assumptions used by the evaluation logic.
- The project is an educational prototype rather than a production autoscaling platform.

### Future Improvements

Potential extensions include:

- Integrating memory utilization and additional metrics through the appropriate CloudWatch Agent configuration.
- Evaluating the model on longer monitoring windows and more diverse workloads.
- Comparing additional forecasting models and allocation baselines.
- Improving monitoring, error reporting, and experiment reproducibility.
- Deploying the dashboard behind appropriate authentication and network controls.
- Conducting a controlled live-scaling demonstration with explicit permissions, rollback procedures, and cleanup steps.

---

## 📚 References

- [Amazon EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Amazon CloudWatch Documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html)
- [Amazon EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [Amazon CloudWatch Pricing](https://aws.amazon.com/cloudwatch/pricing/)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🎓 Academic Context

CloudPulse was developed as a laboratory project for the Advanced Cloud Computing subject.

The project demonstrates the application of cloud monitoring, machine learning, time-series forecasting, resource allocation, and policy evaluation within an AWS-oriented environment.

The implementation emphasizes data provenance, reproducible evaluation, bounded recommendations, and safe operation.



<p align="center">
  <b>CloudPulse 📡</b><br>
  Forecast. Recommend. Scale. Safely.
</p>
```

