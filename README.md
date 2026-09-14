# project

Project using NLP to classify if a tweet is about a real disaster or not. The is a part of the
[MLOps course at DTU](https://skaftenicki.github.io/dtu_mlops/latest/) and the focus will therefore be to learn and use MLOps methods.

## Data
The project will use the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) from Kaggle.

## Model and training
The model consist of [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) from the transformers library and a simple linear classifier.
The training is implemented using PyTorch wrapped in PyTorch Lightning to handle checkpoint, logging, device manement etc.



## Frameworks & Tools
| Tool                      | Purpose                                                      |
| :------------------------ | :----------------------------------------------------------- |
| **UV**                    | Project and package manager                                  |
| **PyTorch**               | Deep learning framework for model architecture and training  |
| **PyTorch Lightning**     | Boilerplate for PyTorch                                      |
| **DVC**                   | Data versioning linked to GCP Cloud Storage                  |
| **Docker**                | Containerization                                             |
| **GitHub Actions**        | CI/CD for unit testing, linting, and automated cloud deployments  |
| **GCP Artifact Registry** | Docker image hosting                                         |
| **GCP Vertex AI**         | Cloud training infrastructure                        |
| **GCP Cloud Run**         | Serverless deployment for API and frontend                   |
| **FastAPI**               | Backend API for triggering cloud training jobs               |
| **Gradio**                | Frontend UI for image upload and inference                   |
| **Weights & Biases**      | Experiment tracking, metrics logging, and artifact storage   |
| **Hydra**                 | Configs management for reproducibility   |



## Project structure
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
