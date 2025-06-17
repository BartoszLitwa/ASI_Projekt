### Conda env config
conda env create -f environment.yml

# ASI_loanPredictor

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project with PySpark setup, which was generated using `kedro 0.19.12`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)

# Loan Prediction Kedro Project

This project uses the Kaggle Loan Prediction dataset to build a machine learning pipeline with Kedro, and provides a Streamlit UI for model inference, all containerized with Docker/Podman.

---

## 1. Environment Setup

### 1.1. Clone the Repository
```bash
git clone <your-repo-url>
cd ASI_Projekt
```

### 1.2. Create and Activate Conda Environment
```bash
conda env create -f environment.yml
conda activate asi_projekt
```

### 1.3. Deactivate Conda Environment
```bash
conda deactivate
```

### 1.4. Install Project Dependencies (if not done by environment.yml)
```bash
pip install -r requirements.txt
```

---

## 2. Prepare the Data

1. **Download the Kaggle dataset** from [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/ethicalstar/loan-prediction).
2. **Place the CSV file** as:
   ```
   data/01_raw/loan_data.csv
   ```

---

## 3. Run the Kedro Pipeline

This will:
- Split the data into train/test
- Clean and preprocess the data
- Perform feature engineering
- Train and evaluate a model
- Save the model and features for inference

```bash
kedro run
```

Outputs (model and features) will be saved in `data/06_models/`.

---

## 4. Run the Streamlit App (Docker/Podman)

### 4.1. Build the Docker Image
```bash
cd docker
podman build -t loan-predictor-app .
```
*Or use `docker build -t loan-predictor-app .` if you use Docker.*

### 4.2. Run the Container
```bash
podman run -p 8501:8501 -v ../data/06_models:/app/data/06_models loan-predictor-app
```
*Or use `docker run -p 8501:8501 -v ../data/06_models:/app/data/06_models loan-predictor-app` if you use Docker.*

### 4.3. Access the App
Open your browser at [http://localhost:8501](http://localhost:8501)

---

## 5. Project Structure

- `data/01_raw/loan_data.csv`: Raw Kaggle data
- `src/asi_loanpredictor/pipelines/`: Kedro pipelines for data preparation, processing, and model training
- `docker/`: Streamlit app and Dockerfile
- `data/06_models/`: Trained model and features for inference

---

## 6. Useful Commands

### Activate Environment
```bash
conda activate asi_projekt
```

### Deactivate Environment
```bash
conda deactivate
```

### Run Kedro Pipeline
```bash
kedro run
```

### Build Docker Image
```bash
cd docker
podman build -t loan-predictor-app .
```

### Run Docker Container
```bash
podman run -p 8501:8501 -v ../data/06_models:/app/data/06_models loan-predictor-app
```

---

## 7. Notes
- **Always run the Kedro pipeline before building/running the Docker image** to ensure the model and features are available for the Streamlit app.
- You can further customize the pipeline or Streamlit UI as needed.
- For troubleshooting, check the logs/output of both Kedro and the Streamlit app.

---

## 8. References
- [Kedro Documentation](https://docs.kedro.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/ethicalstar/loan-prediction)
