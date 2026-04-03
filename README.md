# Credit Risk Modeling | LendingClub Loan Default Prediction
## Dataset
The dataset used in this project is the **Lending Club Loan Data**, publicly available on Kaggle.
It contains real loan records issued through the Lending Club peer-to-peer lending platform
between 2007 and 2018, covering over 2.26 million observations and 150+ variables.

Each row represents a single loan and includes information available at the time of application
(borrower demographics, credit bureau attributes, loan terms) as well as post-origination
outcomes (payment history, default status, recovery amounts).

Dataset is too large for GitHub.
Download Raw Data from:
https://drive.google.com/file/d/1KZ8tMBhNlpTZTmRVzqqGyHLQNMVdI00e/view?usp=sharing
## Data Preprocessing
Before modeling, the dataset was cleaned and standardized to ensure compatibility with traditional logistic regression and machine learning models, which require numerical inputs.
- Variables containing identifiers, text descriptions, or information with little predictive relevance were removed.
- Categorical variables stored as character values were transformed into appropriate numerical formats, including ordinal encoding and dummy variables where necessary.
- Binary categorical variables were converted into numerical indicators (0/1).
- Date variables were standardized and transformed into numerical representations based on the number of days relative to the loan issuance date.
- Employment-related and other textual numerical variables were converted into numeric values.
These preprocessing steps ensured that all variables were in a consistent numeric format suitable for statistical and machine learning models.

The original dataset contains approximately 2.26 million observations and 158 variables. Data preprocessing was conducted to address missing values and ensure consistency across different types of loan applications.
Missing values were first analyzed by calculating the proportion of missing observations for each variable. 
- For variables with a low missing rate, observations containing missing values were removed.
- Variables with a high proportion of missing values were removed from the dataset, as they may reduce data quality and negatively affect model performance.

After preprocessing and data cleaning, the final dataset contains 2,013,321 observations and 102 variables, which are used for subsequent modeling and analysis.

Download Clean Data from:
https://drive.google.com/file/d/1SEpqWIyXYhcC58EbDmMpZVLa6Th2WkRM/view?usp=drive_link

## Objectives

1. Select the most informative features for default prediction using Information Value (IV)
2. Develop and benchmark four classification models for Probability of Default (PD)
3. Validate models against regulatory thresholds (KS ≥ 0.30, PSI < 0.10)
4. Quantify portfolio-level Expected Loss under the Basel II framework

---

## Key Mathematical Concepts

### 1. Probability of Default (PD)

PD is the likelihood that a borrower will fail to meet their debt obligations over a given horizon. In binary classification:

$$PD_i = P(Y_i = 1 \mid \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{x}_i^\top \boldsymbol{\beta}}}$$

where $Y_i = 1$ denotes default (Bad loan), $\mathbf{x}_i$ is the feature vector, and $\boldsymbol{\beta}$ is the estimated coefficient vector.

---

### 2. Expected Loss (EL)

The standard Basel II formula for Expected Loss:

$$EL = PD \times LGD \times EAD$$

| Parameter | Description | Value Used |
|-----------|-------------|------------|
| $PD$ | Probability of Default | Predicted by XGBoost |
| $LGD$ | Loss Given Default — fraction of exposure lost if default occurs | 45% (Basel II unsecured retail) |
| $EAD$ | Exposure at Default — outstanding loan balance at time of default | Actual loan balance |

**Portfolio-level Expected Loss:**

$$EL_{\text{portfolio}} = \sum_{i=1}^{N} PD_i \times LGD \times EAD_i$$

> **Result:** Predicted EL = **$56.40M** vs Actual Loss = **$56.54M** — a variance of only **−0.25%**

---

### 3. Information Value (IV)

IV measures the **discriminatory power** of a variable in separating Good from Bad loans:

$$IV = \sum_{i=1}^{k} \left( \%Good_i - \%Bad_i \right) \times \ln\left(\frac{\%Good_i}{\%Bad_i}\right)$$

where $k$ is the number of bins for the variable.

**IV Interpretation:**

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| 0.30 – 0.50 | Strong |
| > 0.50 | Very Strong |

---

### 4. Weight of Evidence (WoE)

Variables are binned and transformed via WoE before modeling:

$$WoE_i = \ln\left(\frac{\%Good_i}{\%Bad_i}\right)$$\

WoE transformation ensures monotonic relationships, handles non-linearity, and makes logistic regression coefficients directly interpretable as log-odds contributions.

---

## Model Evaluation Metrics

### AUC — Area Under the ROC Curve

$$AUC = P(\hat{p}_{bad} > \hat{p}_{good})$$

Measures the probability that the model ranks a randomly chosen Bad loan higher than a randomly chosen Good loan. AUC = 0.5 is random; AUC = 1.0 is perfect.

---

### Gini Coefficient

A linear transformation of AUC:

$$Gini = 2 \times AUC - 1$$

Ranges from 0 (random) to 1 (perfect). Widely used in credit scoring as a discrimination metric.

---

### KS Statistic (Kolmogorov-Smirnov)

Measures the **maximum separation** between the cumulative distribution of scores for Good and Bad loans:

$$KS = \max_t \left| F_{Good}(t) - F_{Bad}(t) \right|$$

where $F_{Good}(t)$ and $F_{Bad}(t)$ are the empirical CDFs of predicted scores for Good and Bad loans respectively.

**Regulatory threshold: KS ≥ 0.30 required for model sign-off.**

---

### Population Stability Index (PSI)

PSI measures the **shift in score distribution** between training and test populations:

$$PSI = \sum_{i=1}^{k} \left( \%Actual_i - \%Expected_i \right) \times \ln\left(\frac{\%Actual_i}{\%Expected_i}\right)$$

**PSI Interpretation:**

| PSI Value | Stability Status | Action |
|-----------|-----------------|--------|
| < 0.10 | Stable | No action required |
| 0.10 – 0.20 | Minor shift | Monitor closely |
| > 0.20 | Significant drift | Model redevelopment required |

> **XGBoost PSI = 0.0003** — near-zero drift, confirmed stable for production deployment.

---

### Sensitivity & Specificity

$$Sensitivity = \frac{TP}{TP + FN} \quad \text{(Recall for Bad loans)}$$

$$Specificity = \frac{TN}{TN + FP} \quad \text{(Recall for Good loans)}$$

$$Balanced\ Accuracy = \frac{Sensitivity + Specificity}{2}$$

Due to severe class imbalance (87% Good), overall accuracy is misleading — a model predicting everything as Good achieves 87% accuracy while being completely useless for credit screening.

---

### Kappa Statistic

Measures agreement beyond chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed accuracy and $p_e$ is expected accuracy under random assignment. $\kappa \approx 0$ confirms the model offers minimal improvement over random guessing on the minority class.

---

## Basel II Regulatory Framework

### The Three Pillars

| Pillar | Description |
|--------|-------------|
| Pillar 1 | Minimum Capital Requirements (credit, market, operational risk) |
| Pillar 2 | Supervisory Review Process |
| Pillar 3 | Market Discipline & Disclosure |

### Internal Ratings-Based (IRB) Approach

Under Basel II IRB, banks estimate their own risk parameters:

$$RWA = f(PD, LGD, EAD, M) \times 12.5$$

$$Capital\ Requirement = 8\% \times RWA$$

where $M$ is the effective maturity and $RWA$ is Risk-Weighted Assets.

### Expected Loss vs Unexpected Loss

$$EL = PD \times LGD \times EAD \quad \text{(covered by loan loss provisions)}$$

$$UL = \sqrt{PD \times (1-PD)} \times LGD \times EAD \quad \text{(covered by regulatory capital)}$$

Basel II requires banks to hold capital against **Unexpected Loss (UL)** — the deviation of actual losses from expected losses — not EL itself, which is covered by provisions.

### LGD = 45% — Basel II Foundation IRB

For unsecured retail exposures, Basel II prescribes a **supervisory LGD of 45%**, reflecting:
- Recovery rates on unsecured consumer loans historically average 50–60%
- Legal and administrative costs of recovery
- Time value of money during the recovery process

---

## 🛠️ Tools & Libraries

| Category | Tools |
|----------|-------|
| Language | R |
| Modeling | `caret`, `xgboost`, `rpart`, `glm` |
| Feature Selection | `Information`, `scorecard` |
| Data Processing | `tidyverse`, `dplyr`, `data.table` |
| Visualization | `ggplot2`, `pROC`, `ROCR` |
| Validation | Custom KS, PSI, EL functions |

---

---

## References

- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II).*
- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley.
- Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications.* SIAM.
- Anderson, R. (2007). *The Credit Scoring Toolkit.* Oxford University Press.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016.*
