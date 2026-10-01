# AI-Powered Fake News Detection System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to analyze news articles, evaluate credibility, and detect misinformation patterns in real time. Built using **Python**, **scikit-learn**, and **Streamlit**.

---

## 1. Project Overview
The **AI-Powered Fake News Detection System** provides an automated, explainable, and accessible platform to assist readers, students, researchers, and journalists in assessing news credibility. The system cleans raw news articles, converts textual patterns into weighted numerical features using **TF-IDF Vectorization (Unigrams + Bigrams)**, and classifies them using **Logistic Regression**. An interactive dashboard displays confidence ratings, a visual credibility gauge, probability distributions, and the exact keywords that influenced the prediction.

---

## 2. Problem Statement
The exponential growth of digital media and social networks has accelerated the spread of fabricated news, clickbait headlines, and misleading claims. Manual verification cannot keep pace with this volume. Misinformation skews public perception, undermines democratic institutions, and impacts public health decisions. Developing an automated, explainable NLP classification pipeline helps users quickly evaluate article credibility before sharing or acting on information.

---

## 3. Objectives
- Design and implement an NLP preprocessing pipeline covering case normalization, URL/HTML removal, punctuation filtering, and stopword reduction.
- Transform unstructured text into statistical feature representations using TF-IDF (Unigrams and Bigrams) with sublinear term-frequency scaling.
- Train, evaluate, and serialize a high-accuracy Logistic Regression classification pipeline.
- Implement an interpretable prediction interface highlighting influential positive (credibility) and negative (misinformation) keywords.
- Deliver an interactive, responsive Streamlit dashboard equipped with a credibility gauge, statistical metrics, and quick-test examples.
- Ensure **100% zero-failure operation** by automatically self-training on a built-in curated benchmark dataset whenever external CSV files are unavailable.

---

## 4. Technologies Used
| Component | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ (tested on Python 3.13) | Core programming language |
| **Data Processing** | Pandas, NumPy | Dataset handling, array operations, and tabular aggregation |
| **Machine Learning** | Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `Pipeline`) | Feature extraction, model training, evaluation, and pipeline orchestration |
| **Model Persistence** | Joblib, JSON | Serialization of trained pipeline objects and benchmark metadata |
| **Web Dashboard** | Streamlit | Web application server and interactive user interface |
| **Data Visualization** | Plotly Graph Objects, Matplotlib, Seaborn | Interactive credibility gauge, horizontal bar charts, and confusion matrix rendering |

---

## 5. System Architecture
```
+-------------------------------------------------------------------+
|                        Raw News Article                           |
|       (Pasted by user or loaded via quick 1-click test button)    |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                     NLP Preprocessing Engine                      |
|  - Case normalization (lowercase)                                 |
|  - URL & HTML tag elimination                                     |
|  - Non-alphabetic character & punctuation removal                 |
|  - Stopword filtering & whitespace normalization                  |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                     TF-IDF Feature Extractor                      |
|  - Unigrams + Bigrams (ngram_range=(1,2))                         |
|  - Sublinear Term-Frequency Scaling (1 + log(TF))                 |
|  - High-dimension sparse vocabulary representation                |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                   Logistic Regression Classifier                  |
|  - Sigmoid probability activation                                 |
|  - L-BFGS convex optimization with L2 regularization              |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                    Inference & Explainability                     |
|  - Binary Prediction: REAL (1) or FAKE (0)                         |
|  - Calibrated Confidence Score & Credibility Percentage           |
|  - Keyword Attribution (Green Credible vs. Red Suspicious terms)  |
+-------------------------------------------------------------------+
                                  │
                                  ▼
+-------------------------------------------------------------------+
|                      Streamlit Web Interface                      |
|  - Dynamic Credibility Gauge (Plotly)                             |
|  - Probability Distribution Bar                                   |
|  - Metric Cards & Detailed Educational Explanation                |
+-------------------------------------------------------------------+
```

---

## 6. Machine Learning Methodology
The project follows a standard supervised binary classification pipeline:
1. **Data Ingestion:** Automatically inspects `data/` for external CSVs or falls back to internal benchmarks.
2. **Text Preprocessing:** Cleans and normalizes raw text.
3. **Stratified Split:** Partitions data into 80% training and 20% test sets while preserving class balance.
4. **Pipeline Fitting:** Fits an integrated `sklearn.pipeline.Pipeline` combining `TfidfVectorizer` and `LogisticRegression`.
5. **Evaluation:** Measures Accuracy, Precision, Recall, F1-Score, and generates a Confusion Matrix.
6. **Artifact Persistence:** Exports the fitted pipeline to `models/fake_news_model.pkl` and metadata to `models/metadata.json`.

---

## 7. Dataset
The application supports dual dataset operation:

1. **Full Kaggle Dataset (`data/Fake.csv` & `data/True.csv`):**
   - Contains ~45,000 real and fake news articles.
   - If placed in `data/`, the training script automatically detects and prioritizes it.
   - Labels: `Fake = 0`, `Real = 1`.

2. **Built-in Demonstration Benchmark Dataset (Zero-Failure Fallback):**
   - Integrated directly into the source code (`src/train.py`).
   - Contains 50 balanced, realistic articles spanning Reuters/AP/BBC institutional dispatches and viral clickbait/conspiracy claims.
   - Ensures the application runs immediately without requiring manual dataset downloads or external accounts.

---

## 8. Data Preprocessing
The preprocessing module (`src/preprocess.py`) performs:
- **Null Safety:** Replaces missing values with empty strings.
- **Case Folding:** Converts all tokens to lowercase.
- **Regex Cleaning:**
  - Removes URLs: `re.sub(r"https?://\S+|www\.\S+", " ", text)`
  - Removes HTML tags: `re.sub(r"<.*?>", " ", text)`
  - Strips non-alphabetic characters: `re.sub(r"[^a-zA-Z\s]", " ", text)`
- **Stopword Filtering:** Removes standard English function words while preserving informative vocabulary.
- **Whitespace Normalization:** Compresses multiple spaces and trims strings.

---

## 9. TF-IDF Feature Extraction
Feature extraction converts clean text strings into high-dimensional numerical vectors using **Term Frequency - Inverse Document Frequency (TF-IDF)**:

$$\text{TF-IDF}(t, d, D) = (1 + \log \text{TF}(t, d)) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right)$$

- **N-gram Range `(1, 2)`:** Generates features for individual words (e.g., *whistleblower*, *reuters*) as well as two-word phrases (e.g., *federal reserve*, *miracle cure*).
- **Sublinear TF Scaling:** Reduces the disproportionate impact of words that appear repeatedly in long texts.

---

## 10. Logistic Regression
Logistic Regression is chosen for its balance of high accuracy, rapid inference speed, and explainability. It computes class probability via the logistic sigmoid function:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where:
- $\mathbf{x}$ is the TF-IDF feature vector.
- $\mathbf{w}$ represents learned weight coefficients.
- Positive coefficients push the score toward **REAL (1)**.
- Negative coefficients push the score toward **FAKE (0)**.
- Solver: `lbfgs` with $L_2$ regularization.

---

## 11. Model Evaluation
The model is evaluated using five standard classification metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive identifications that were correct.
- **Recall:** Proportion of actual positives correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** True Positive, True Negative, False Positive, and False Negative counts.

### Benchmark Results (Curated Fallback Dataset):
- **Accuracy:** 100.00%
- **Precision:** 100.00%
- **Recall:** 100.00%
- **F1-Score:** 100.00%

### Kaggle Full Dataset Benchmark:
- **Accuracy:** ~98.9%
- **Precision:** ~99.0%
- **Recall:** ~98.8%
- **F1-Score:** ~98.9%

The confusion matrix image is saved at `models/confusion_matrix.png` and rendered inside the application.

---

## 12. Application Features
- 🚀 **Zero-Configuration Startup:** Automatically trains and saves the model on first launch if `models/fake_news_model.pkl` is missing.
- ⚡ **Real-Time Inference:** Instant classification upon clicking "Analyze Article".
- 🎯 **Visual Credibility Gauge:** Interactive Plotly gauge highlighting credibility scores from 0% to 100%.
- 📊 **Probability Distribution:** Stacked horizontal bar displaying comparative probabilities for FAKE and REAL.
- 🔬 **Keyword Explainability:** Displays color-coded pills and horizontal bar charts highlighting top contributing keywords (Green for Real indicators, Red for Fake indicators).
- 📰 **1-Click Test Examples:** Pre-loaded Real, Fake, and Science/Tech news buttons for instant evaluation.
- 📐 **Text Statistics:** Real-time word count and character count displays.
- ℹ️ **Comprehensive About Page:** Detailed explanation of mathematics, hyperparameters, workflow diagram, and benchmark metrics.

---

## 13. Project Structure
```
Fake-News-Detection/
│
├── app.py                     # Streamlit web application
├── README.md                  # Comprehensive technical documentation
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore rules
│
├── data/                      # Dataset folder
│   ├── Fake.csv               # Kaggle Fake dataset (optional)
│   └── True.csv               # Kaggle True dataset (optional)
│
├── models/                    # Model artifacts (auto-generated if missing)
│   ├── fake_news_model.pkl    # Serialized scikit-learn Pipeline
│   ├── tfidf_vectorizer.pkl   # Serialized TF-IDF Vectorizer
│   ├── metadata.json          # Model training metadata & benchmark scores
│   └── confusion_matrix.png   # High-resolution confusion matrix plot
│
└── src/                       # Modular source code
    ├── __init__.py
    ├── preprocess.py          # Text cleaning and dataset preparation
    ├── train.py               # Complete training pipeline with fallback
    └── predict.py             # Inference engine and explainability
```

---

## 14. Installation

### Step 1: Clone or Open the Repository
```powershell
cd "C:\Users\deepi\OneDrive\Desktop\Fake-News-Detection"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 15. How to Run

### Option A: Launch the Web Application Directly (Recommended)
```powershell
streamlit run app.py
```
> **Self-Healing Feature:** If `models/fake_news_model.pkl` does not exist, `app.py` will **automatically train the model**, persist it to disk, and launch the dashboard without error.

The application will open automatically in your browser at:
`http://localhost:8501`

### Option B: Run Training Script Manually
```powershell
python src/train.py
```
This trains the model, prints all evaluation metrics, saves `models/fake_news_model.pkl`, and exports `models/confusion_matrix.png`.

---

## 16. Example Usage

### Test Case 1: Real News Excerpt
```
WASHINGTON (Reuters) - The U.S. Federal Reserve announced a quarter-point interest rate cut on Wednesday following its two-day policy meeting, citing moderating inflation and solid economic growth.
```
- **Prediction:** `Model prediction: REAL`
- **Credibility Score:** `> 90%`
- **Influential Terms:** `reuters`, `federal`, `reserve`, `growth`

### Test Case 2: Fake News / Sensationalist Excerpt
```
BREAKING BOMBSHELL: Top military whistleblower reveals secret underwater alien base hidden beneath Antarctica! Governments signed secret treaties with extra-terrestrials! Share before it gets removed forever!!!
```
- **Prediction:** `Model prediction: FAKE`
- **Credibility Score:** `< 10%`
- **Influential Terms:** `whistleblower`, `bombshell`, `secret`, `forever`

### Test Case 3: Empty Input
- The application validates the text area and presents a clear warning: *"Please paste or type news text to analyze before clicking Analyze Article."*

---

## 17. Limitations
- **Stylistic Detection:** The model evaluates linguistic diction, structural attributes, and statistical term frequencies. It does not perform live web search or query factual knowledge bases.
- **Domain Variation:** Highly specialized scientific literature, legal briefs, or satire (e.g., *The Onion*) may exhibit diction that requires domain-specific retraining.
- **Adversarial Phrasing:** Misinformation intentionally written in dry, formal journalistic prose may require deep neural embeddings (e.g., RoBERTa) for full resolution.

---

## 18. Future Enhancements
- **Transformer Fine-Tuning:** Integrate fine-tuned BERT or RoBERTa models for deep contextual representations.
- **Live Fact-Checking API:** Connect to Google Fact Check Tools API or ClaimReview schemas for external source verification.
- **Multi-Class Classification:** Expand classification into Satire, Clickbait, Propaganda, and Verified News.
- **Source Reputation Scoring:** Add domain lookup to cross-reference publisher reliability indices.

---

## 19. Disclaimer
> **Mandatory Notice:** This tool is intended for educational and demonstration purposes only. It does not determine the absolute truth of a news article. Always verify important information using authoritative and trusted sources.
