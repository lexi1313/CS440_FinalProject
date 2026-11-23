# Tool Competition - Skill Classification

## Competition Overview

The 2027 competition consists of building and assessing multi‑label classifiers that predict, for each issue, the set of domains and sub‑domains representing the skills required to solve it.


## Dataset

We release a dataset mined from 7,245 merged pull requests across 11 popular Java repositories (57,206 source files; 59,644 methods; 13,097 classes) annotated with 217 skill labels composed by domain/sub‑domains You may find the dataset in a SQLite database, named `skillscope_data.db`, [here](https://huggingface.co/datasets/NLBSE/SkillCompetition). This dataset is an updated version of the one submitted in the original [SkillScope](#citing-relevant-work) paper.

### Ready-made Table
Inside the database you will find a table named `nlbse_tool_competition_data_by_issue` that joins each pull request’s textual and code‑context features with its canonical domain/sub‑domain labels per issue. There is also a view `vw_nlbse_tool_competition_data_by_file` that labels each filename/function associated with each issue. 

- The `nlbse_tool_competition_data_by_issue` table contains a column for each domain and subdomain with an integer count of the number of matching APIs found for that issue. A value greater than zero indicates that domain/subdomain is present in the issue.

- The `vw_nlbse_tool_competition_data_by_file` is present for convenience purposes only and will not be used in evaluation of the model.

### Additional Metadata/Input Data is Allowed
Your model is allowed to use input data that is outside what is given in the database. You may use additional GitHub APIs to fetch more metadata, or download relevant files in an issue for further analysis. However, you must not use any third-party classification engine or the outputs present in `skillscope_data.db` as direct inputs to the model. 

### Baselines
Your models will be compared against the SkillScope Random‑Forest + TF‑IDF baselines reported in the paper by evaluating the overall prediction metrics against the issue classifications as recorded in the `nlbse_tool_competition_data_by_issue` table. The models you create should return a multi-label classification encoded in an one-hot encoded vector. However, it is the metrics of your models which will be the subject to the evaluation instead of the specific model output. Random Forest Baseline is provided in the folder `competition_baseline` with README and requirements.txt.

### Goal
Train, tune and evaluate your models on the provided splits and improve at least one of precision, recall, or micro‑F1 while not decreasing the remaining metrics relative to the best baseline.

## Our Implementation

We implemented and evaluate three models for multi-label skill classification: Random forest, Logistic Regression, and a Multi-Layer Perceptron. All three models use TF-IDF feautres extracted from issue and description text. 

### Random Forest
We implemented a Random Forest classifier as a baseline for comparing model performance. The model uses 100 decision trees trained on TF-IDF features to predict multiple skill labels for each issue. We evaluated its performance on the same validation split used by the other models to ensure consistent comparisons.

### Logistic Regression
We implemented Logistic Regression using a One-vs-Rest approach, which trains a separate binary classifier for each skill label. The model uses class balancing to account for differences in label frequency and TF-IDF features to represent issue text. We evaluated its performance against the Random Forest baseline using precision, recall, and F1 scores.

### Multi-Layer Perceptron Model
We implemented a Multi-Layer Perceptron neural network using TensorFlow and Keras. The model consists of two hidden layers with 256 and 128 units, batch normalization, and regularization to reduce overfitting. We trained the model using binary cross-entropy loss and early stopping, then evaluated multiple classification thresholds to identify a configuration that improves performance without degrading the baseline metrics.

## Evaluation
Model performance is evaluated using precision, recall, micro-F1, and macro-F1. The competition requires that a candidate model improve at least one of the required metrics without degradting the others. The `evaluation.py` script extracts saved validation metrics from `CS440_FinalProject.ipynb` and compares the model performance against the baseline thresholds. 

_It is important to note that this script reads previously computed outputs and does not retrain or independently recalculate performance metrics._

### Model Comparison Table
The following table summarizes the performance of each model using precision, recall, micro-F1, and macro-F1. Results were extracted using `evaluation.py`, which also checks for degradation relative to the baseline.

```
+---------------------+-------------+----------+------------+------------+----------------+
| Model               |   Precision |   Recall |   Micro-F1 |   Macro-F1 | Degradation?   |
+=====================+=============+==========+============+============+================+
| Random Forest       |      0.7201 |   0.3643 |     0.4838 |     0.0922 | Baseline       |
+---------------------+-------------+----------+------------+------------+----------------+
| Logistic Regression |      0.6041 |   0.3611 |     0.452  |     0.1698 | Yes            |
+---------------------+-------------+----------+------------+------------+----------------+
| MLP (0.68)          |      0.7175 |   0.3762 |     0.4936 |     0.1051 | Yes            |
+---------------------+-------------+----------+------------+------------+----------------+
| MLP (0.69)          |      0.7226 |   0.3701 |     0.4895 |     0.1025 | No             |
+---------------------+-------------+----------+------------+------------+----------------+
| MLP (0.70)          |      0.726  |   0.3638 |     0.4847 |     0.1003 | Yes            |
+---------------------+-------------+----------+------------+------------+----------------+
| MLP (0.71)          |      0.7282 |   0.3564 |     0.4785 |     0.0975 | Yes            |
+---------------------+-------------+----------+------------+------------+----------------+
```

### Results
We selected the MLP model with a classification threshold of 0.69 because it was the only evaluated configuration that improved precision, recall, and micro-F1 compared to the Random Forest baseline without degrading the remaining required metrics.

## Running the Code

### 1. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

### 2. Download the Dataset

Download `skillscope_data.db` from the [NLBSE Skill Competition dataset](https://huggingface.co/datasets/NLBSE/SkillCompetition). Ensure dataset is v2 to avoid errors.

When running locally, place the database in the `data` directory and update the `db_path` variable in the notebook to match its location.

### 3. Run the Models

Open `CS440_FinalProject.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

Run the data-loading and preprocessing cells before training the models. The notebook contains implementations of Random Forest, Logistic Regression, and a Multi-Layer Perceptron (MLP), along with their evaluation procedures.

The original competition baseline is included separately in the `competition_baseline` directory and is not required to run the team's model comparison.

### 4. Run the Evaluation

To view and compare the saved model results without retraining, run:

```
python evaluation.py
```

## Organizers

The skill classification competition is organized by: Fabio Santos (fabio.deabreusantos@colostate.edu), and Jacob Penney (jmp458@nau.edu)

## Submission Acceptance & Competition

Submissions will first be filtered to ensure they do not lower any of the three core metrics (precision, recall, micro‑F1) compared with the baseline. Among qualifying submissions, ranking is determined by the largest positive improvement in micro‑F1. Ties will be broken by (1) precision, then (2) runtime.

Submissions will be judged on:
- Clarity and completeness of the paper.
- Availability and reproducibility of code/tool (open‑source required).
- Correct reporting of metrics.
- Quality of documentation.

Accepted papers will appear in the **NLBSE ’27 proceedings**.

### Ranking Details

Participants submit a single multi‑label classifier. Rankings proceed in two stages:

1. Eligibility check: The submission must *not* reduce any of precision, recall, or micro‑F1 compared with the baseline.
2. Ordering: Qualifying submissions are ordered by the greatest positive ∆micro‑F1.

## Citing Relevant Work

### General Competition Citation:

```tex
@inproceedings{nlbse2027,
  author = {Mock, Moritz and Borsani, Thomas and Russo, Barbara and Santos, Fabio and and Penney, Jacob and Valenzuela{-}Toledo, Pablo and Kehrer, Timo and Panichella, Sebastiano },
  title={The NLBSE'27 Tool Competition and Challenge},
  booktitle={Proceedings of The 6th International Workshop on Natural Language-based Software Engineering (NLBSE'27)},
  year={2027}
}
```

### Skill Competition Citation:
```tex
@inproceedings{carter2025skillscope,
    title        = {SkillScope: A Tool to Predict Fine-Grained Skills Needed to Solve Issues on GitHub},
    author       = {Carter, Benjamin C. and Contreras, Jonathan Rivas and Llanes Villegas, Carlos A. and Acharya, Pawan and Utzerath, Jack and Farner, Adonijah O. and Jenkins, Hunter and Johnson, Dylan and Penney, Jacob and Steinmacher, Igor and Gerosa, Marco A. and Santos, Fabio},
    year         = 2025,
    booktitle    = {2025 IEEE/ACM International Workshop on Natural Language-Based Software Engineering (NLBSE)},
    volume       = {},
    number       = {},
    pages        = {9--12},
    doi          = {10.1109/NLBSE66842.2025.00007},
    keywords     = {Random Forest;Training;Java;Large language models;Semantics;Retrieval augmented generation;Machine learning;Open source software;Software engineering;Software development management;software engineering;skill categorization;open source software (OSS);machine learning;large language models}
}
```
```tex
@article{santos2023tag,
    title        = {Tag that issue: applying API-domain labels in issue tracking systems},
    author       = {Santos, Fabio and Vargovich, Joseph and Trinkenreich, Bianca and Santos, Italo and Penney, Jacob and Britto, Ricardo and Pimentel, Jo{\~a}o Felipe and Wiese, Igor and Steinmacher, Igor and Sarma, Anita and others},
    year         = 2023,
    journal      = {Empirical Software Engineering},
    publisher    = {Springer},
    volume       = 28,
    number       = 5,
    pages        = 116
}
```
```tex
@inproceedings{vargovich2023givemelabeledissues,
    title        = {Givemelabeledissues: An open source issue recommendation system},
    author       = {Vargovich, Joseph and Santos, Fabio and Penney, Jacob and Gerosa, Marco A and Steinmacher, Igor},
    year         = 2023,
    booktitle    = {2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR)},
    pages        = {402--406},
    organization = {IEEE}
}
```
