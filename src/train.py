"""
train.py
--------
Training pipeline for the AI-Powered Fake News Detection System.

Features:
  1. Locates project root dynamically regardless of execution directory.
  2. Detects external datasets (data/Fake.csv & data/True.csv) if present.
  3. Seamlessly falls back to an extensive built-in curated benchmark dataset
     with realistic Real and Fake news examples if external files are missing.
  4. Performs NLP cleaning, tokenization, and stopword removal.
  5. Trains a scikit-learn Pipeline combining TF-IDF Vectorizer (1-2 n-grams)
     and Logistic Regression classifier.
  6. Evaluates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  7. Persists models/fake_news_model.pkl, models/tfidf_vectorizer.pkl,
     models/metadata.json, and models/confusion_matrix.png.
  8. Prints the exact model path after successful training.
  9. Never crashes due to missing dataset files.
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all platforms
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─── Dynamic Paths Resolution ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path to enable clean imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import clean_text, load_dataset, prepare_text_column, preprocess_dataframe

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Target Artifact Paths ─────────────────────────────────────────────────────
FAKE_CSV     = DATA_DIR / "Fake.csv"
REAL_CSV     = DATA_DIR / "True.csv"
MODEL_OUT    = MODELS_DIR / "fake_news_model.pkl"
TFIDF_OUT    = MODELS_DIR / "tfidf_vectorizer.pkl"
METADATA_OUT = MODELS_DIR / "metadata.json"
CM_IMG_OUT   = MODELS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42

# ─── Built-in Curated Fallback Dataset ──────────────────────────────────────────
# Realistic, balanced collection of 50 journalistic and sensationalist articles
FALLBACK_DATA = [
    # ── REAL NEWS SAMPLES (Label: 1) ──────────────────────────────────────────
    {"text": "WASHINGTON (Reuters) - The U.S. Federal Reserve announced a quarter-point interest rate cut on Wednesday following its two-day policy meeting, citing moderating inflation and solid economic growth. Fed Chairman Jerome Powell noted that the central bank remains committed to sustaining economic expansion while keeping labor markets balanced and price stability on track.", "label": 1},
    {"text": "GENEVA (AP) - The World Health Organization confirmed that international health authorities have approved a new vaccine for malaria prevention in sub-Saharan Africa. Clinical trials conducted across five countries demonstrated significant efficacy in reducing severe illness and child hospitalizations, public health officials announced.", "label": 1},
    {"text": "CAPE CANAVERAL, Fla. (Reuters) - NASA and SpaceX successfully launched four astronauts to the International Space Station on Sunday aboard a Falcon 9 rocket and Crew Dragon capsule from Kennedy Space Center. The six-month research mission will conduct experiments in biology, material science, and Earth observations.", "label": 1},
    {"text": "LONDON (BBC) - The Bank of England held its benchmark interest rate steady today at 5.0 percent, signaling cautious optimism over falling inflation figures. Governor Andrew Bailey stated that borrowing costs will remain restricted until domestic services inflation shows consistent decelerating trends.", "label": 1},
    {"text": "TOKYO (Reuters) - Japan's economy expanded at an annualized rate of 2.9 percent in the second quarter, revised government data showed on Monday. Strong capital expenditure and recovering domestic auto production fueled the upgrade from initial preliminary estimates.", "label": 1},
    {"text": "PARIS (AFP) - European climate monitoring agency Copernicus reported that global temperatures over the past twelve months were among the highest recorded in modern history. Scientists reiterated the urgent requirement for global emission reductions to meet Paris Climate Agreement thresholds.", "label": 1},
    {"text": "BRUSSELS (Reuters) - European Union finance ministers reached agreement on comprehensive digital market regulations targeting anti-competitive tech monopolies. The framework establishes compliance protocols and consumer privacy protections across all 27 member states.", "label": 1},
    {"text": "NEW YORK (The Wall Street Journal) - Major U.S. stock indices ended higher on Friday as tech sector gains and robust corporate quarterly earnings buoyed investor sentiment. The S&P 500 rose 0.7 percent while the Nasdaq Composite gained 1.1 percent.", "label": 1},
    {"text": "BERLIN (Deutsche Welle) - Germany's statistical office revealed that renewable energy accounted for over 58 percent of nationwide electricity consumption during the first half of the year, driven by record solar and offshore wind installations.", "label": 1},
    {"text": "OTTAWA (CBC) - The Canadian government introduced legislation requiring major digital platforms to share revenue with domestic news publishers. Heritage Minister officials defended the policy as necessary to sustain independent journalism across the country.", "label": 1},
    {"text": "WASHINGTON (AP) - The United States Department of Agriculture updated dietary guidelines, emphasizing nutrient-dense whole grains, legumes, and reduced added sugar consumption for public health improvement based on peer-reviewed clinical nutrition studies.", "label": 1},
    {"text": "NEW DELHI (PTI) - The Indian Space Research Organisation successfully positioned a solar observatory satellite into its designated halo orbit around the Sun-Earth Lagrange point L1, marking an essential milestone in deep-space solar weather research.", "label": 1},
    {"text": "BEIJING (Reuters) - Factory output and retail sales in China grew steadily in October, reflecting government stimulus measures and domestic consumer subsidies according to the National Bureau of Statistics.", "label": 1},
    {"text": "CANBERRA (ABC News Australia) - The Australian federal parliament approved a landmark national environmental protection authority to enforce conservation laws and oversee emissions reduction compliance for large industrial developments.", "label": 1},
    {"text": "STOCKHOLM (Reuters) - The Royal Swedish Academy of Sciences awarded the Nobel Prize in Physics to two researchers for foundational breakthroughs enabling machine learning and artificial neural network algorithms.", "label": 1},
    {"text": "SEOUL (Yonhap) - South Korean semiconductor manufacturers reported strong export growth in memory chips, driven by rising worldwide enterprise demand for generative artificial intelligence server hardware.", "label": 1},
    {"text": "ROME (Reuters) - United Nations Food and Agriculture Organization data indicated international food commodity price indices stabilized throughout the previous quarter, with grain and dairy costs seeing moderate decreases.", "label": 1},
    {"text": "WASHINGTON (Reuters) - The bipartisan Senate infrastructure committee released audit reports confirming over 40,000 public transportation and bridge repair projects have received federal funding allocations since 2021.", "label": 1},
    {"text": "BRASILIA (AFP) - Brazilian satellite monitoring data reported an 18 percent decrease in Amazon rainforest deforestation over the past year following strengthened enforcement operations by environmental regulatory agencies.", "label": 1},
    {"text": "ZURICH (Bloomberg) - Swiss pharmaceutical regulators approved an innovative monoclonal antibody therapy for treating early-stage Alzheimer's disease after international multi-center trials demonstrated measurable slowing of cognitive decline.", "label": 1},
    {"text": "HELSINKI (Reuters) - Nordic telecommunications providers signed collaborative agreements to deploy next-generation resilient undersea fiber cables connecting Scandinavia directly to Western European data hubs.", "label": 1},
    {"text": "SINGAPORE (The Straits Times) - The Monetary Authority of Singapore unveiled revised sustainable finance guidelines requiring commercial banks to perform comprehensive climate stress testing on commercial loan portfolios.", "label": 1},
    {"text": "MADRID (El Pais) - Spain's national health ministry confirmed expanded public funding for childhood immunizations against respiratory syncytial virus following a successful pilot program in regional hospitals.", "label": 1},
    {"text": "TORONTO (The Globe and Mail) - Canadian research institutions received federal research grants to advance superconducting materials research aimed at reducing transmission losses in high-voltage power grids.", "label": 1},
    {"text": "VIENNA (Reuters) - The International Atomic Energy Agency verified that nuclear power generation facilities across Eastern Europe have met upgraded safety benchmarks following independent regulatory inspections.", "label": 1},

    # ── FAKE NEWS SAMPLES (Label: 0) ──────────────────────────────────────────
    {"text": "BREAKING BOMBSHELL: Top military whistleblower reveals secret underwater alien base hidden beneath Antarctica! Governments around the world have signed secret treaties with extra-terrestrials to exchange human DNA for weather control superweapons! MSM and corrupt elites are deleting this everywhere! Share before it gets removed forever!!!", "label": 0},
    {"text": "SHOCKING EXPOSURE: Doctors caught on hidden camera confessing that drinking boiled garlic lemon juice instantly eliminates all cancer cells and dissolves blood clots in 12 hours! Big Pharma executives had this banned from medical schools to protect their trillion dollar chemotherapy scam! Forward to 10 friends now to save a life!", "label": 0},
    {"text": "ALERT: Secret UN documents leaked! Globalist elites are planning mandatory microchip implants disguised as 5G network towers to control human brainwaves and wipe out cash currency by December! The mainstream media refuses to report this terrifying truth! Wake up sheeple!", "label": 0},
    {"text": "PROOF: NASA insider leaks top secret footage showing the moon landing was staged in a Hollywood soundstage directed by Stanley Kubrick! Gravity wires clearly visible in unedited tape released by deep state insiders! Share everywhere before FBI takes down the link!", "label": 0},
    {"text": "EMERGENCY: Celebrity billionaire arrested in secret military tribunal at Guantanamo Bay for treason and cloning political figures! Cloned doubles currently seen on TV while the real criminal awaits execution! Mainstream media silent! Retweet this truth immediately!", "label": 0},
    {"text": "MIRACLE CURE: Ancient Himalayan root discovered that reverses biological aging by 30 years and cures blindness overnight! Corrupt pharmaceutical cartels are raiding health food stores to confiscate supplies! Order now before the deep state bans it completely!", "label": 0},
    {"text": "BOMBSHELL EVIDENCE: Chemtrails sprayed by commercial airplanes contain mind-altering nano-bots designed to enforce obedience to communist global dictators! Former pilot confesses everything on deathbed! The government is poisoning our skies while you sleep!", "label": 0},
    {"text": "CONFIRMED: Government weather modification HAARP weapon intentionally triggered recent earthquakes to manipulate real estate prices and destroy patriotic communities! Secret seismic frequency recordings prove artificial manipulation! Share before censorship hits!", "label": 0},
    {"text": "EXPOSED: Secret underground cloning facility discovered beneath Denver airport where political leaders are replaced with reptilian humanoids! Whistleblower security guard found dead after recording shocking video proof! The establishment wants you blind!", "label": 0},
    {"text": "SHOCKING TRUTH: Drinking distilled saltwater under direct moonlight will instantly purge all toxins, cure diabetes, and rebuild damaged organs! Western medicine wants you sick and enslaved to prescription pills! Wake up and reject the pharmaceutical mafia!", "label": 0},
    {"text": "BREAKING REPORT: Secret world bank ledger reveals all national debt is fake and imaginary created by central bankers to enslave humanity! Every citizen is owed five million dollars in gold backed currency starting next Tuesday! Don't let politicians hide your money!", "label": 0},
    {"text": "MASSIVE SCANDAL: Smart meters installed on homes are actually covert laser scanning devices that broadcast your private bedroom conversations to foreign surveillance agencies! Remove your meter now with this simple coat hanger trick before they arrest you!", "label": 0},
    {"text": "LEAKED AUDIO: Famous pop star caught admitting that popular music contains satanic backward frequencies that hypnotize teenagers into buying corporate junk! Deep state record labels threaten anyone who exposes the dark secret! Spread the truth now!", "label": 0},
    {"text": "UNBELIEVABLE: Scientist invents free infinite energy engine made from everyday copper wire and magnets, gets kidnapped by oil tycoons hours before press conference! His blueprints have been leaked online! Build your own generator before government blocks access!", "label": 0},
    {"text": "PANIC: Bill Gates patents technology to block out the sun and freeze global agriculture so you are forced to eat lab-grown synthetic insect burgers! Leaked patents prove the catastrophic depopulation agenda! Forward this message to everyone you love!", "label": 0},
    {"text": "BOMBSHELL: Secret tunnel network found linking major tech headquarters directly to offshore pirate islands where elite pedophiles conduct occult ceremonies! Corrupt politicians caught on drone footage! The reckoning is finally here!", "label": 0},
    {"text": "WARNING: Your tap water contains mind-numbing chemicals designed to suppress your natural telepathic abilities and third eye! Ancient civilizations drank only rainwater and lived for 900 years! Boycott public municipal water immediately!", "label": 0},
    {"text": "REVEALED: Deep state judges secretly replaced with artificial intelligence androids dressed in robes to deliver unconstitutional rulings! Court clerk leaks photographic evidence of silicon robotic skin under wigs! The justice system is an AI simulation!", "label": 0},
    {"text": "TERRIFYING DISCOVERY: Fast food burgers contain synthetic micro-sensors that track your physical location without your cellphone! Whistleblower kitchen worker exposes secret boxes shipped from defense contractors! Never eat fast food again!", "label": 0},
    {"text": "BREAKING: Global elite meeting in Switzerland votes to abolish physical cash and confiscate private savings accounts by midnight this weekend! Transfer your wealth into physical silver immediately before the banks freeze all accounts forever!", "label": 0},
    {"text": "SHOCKING BOMBSHELL: Leaked classified satellite photos prove gravity is an illusion generated by deep-sea electromagnetic generators operated by the United Nations! Schools teach lies to hide the true flat horizon! Re-post before YouTube bans this!", "label": 0},
    {"text": "MIRACLE DISCOVERY: Backyard gardener finds common dandelion weed juice eliminates high blood pressure in 20 minutes! Big Pharma sends hired goons to burn down the greenhouse! Save this recipe now before it gets wiped from Google!", "label": 0},
    {"text": "EXCLUSIVE LEAK: Electric vehicle batteries contain hidden frequency emitters that cause memory loss and obedience in suburban drivers! Secret oil company whistleblowers release damning technical blueprints! Do not charge your car at night!", "label": 0},
    {"text": "UNBELIEVABLE SCANDAL: Hollywood studio caught cloning deceased actors to replace elected senators during televised debates! Unedited studio security recordings show green screens in congressional chambers! Wake up America!", "label": 0},
    {"text": "CRITICAL WARNING: Atmospheric moisture is being secretly laced with nanoscopic tracking dust by weather control satellites! Drinking vinegar every morning is the only verified countermeasure against the deep state cloud seeding network!", "label": 0},
]


def train_model():
    """
    Execute the end-to-end training pipeline:
    - Loads external Kaggle datasets if present, or defaults safely to curated fallback.
    - Preprocesses text using NLP cleaning routines.
    - Fits a TF-IDF Vectorizer (ngram_range=(1,2)) + Logistic Regression Classifier.
    - Evaluates performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
    - Persists fake_news_model.pkl, tfidf_vectorizer.pkl, and metadata.json.
    - Prints evaluation metrics and exact model destination.
    """
    logger.info("=" * 65)
    logger.info("  AI-POWERED FAKE NEWS DETECTION SYSTEM — TRAINING PIPELINE")
    logger.info("=" * 65)

    is_fallback = False

    # 1. Dataset Discovery & Loading
    if FAKE_CSV.exists() and REAL_CSV.exists():
        logger.info(f"External Kaggle dataset detected at {DATA_DIR}.")
        try:
            df = load_dataset(str(FAKE_CSV), str(REAL_CSV))
            df = prepare_text_column(df)
            df = df.dropna(subset=["text", "label"])
            dataset_name = "Kaggle Fake and Real News Dataset (Full)"
        except Exception as e:
            logger.warning(f"Error reading external CSV files: {e}. Falling back to curated benchmark data.")
            df = pd.DataFrame(FALLBACK_DATA)
            dataset_name = "Curated High-Quality News Benchmark Dataset (Fallback)"
            is_fallback = True
    else:
        logger.info("data/Fake.csv and data/True.csv not detected.")
        logger.info("Automatically utilizing high-quality curated benchmark dataset (zero failure).")
        df = pd.DataFrame(FALLBACK_DATA)
        dataset_name = "Curated High-Quality News Benchmark Dataset (Fallback)"
        is_fallback = True

    logger.info(f"Dataset active: {dataset_name}")
    logger.info(f"Total raw records: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Text Preprocessing
    df = preprocess_dataframe(df, text_col="text")

    # 3. Train/Test Split (Stratified)
    test_size = 0.20
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"].astype(int),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"].astype(int),
    )

    # 4. Feature Extraction & Classifier Architecture
    max_features = 50000 if not is_fallback else 5000
    min_df = 2 if not is_fallback else 1

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=min_df,
        max_df=0.98,
        strip_accents="unicode",
        analyzer="word",
    )

    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=RANDOM_STATE,
    )

    # Create unified Scikit-learn Pipeline
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    logger.info(f"Fitting Pipeline [TF-IDF (1-2 n-grams) + Logistic Regression] on {len(X_train_raw)} training samples...")
    pipeline.fit(X_train_raw, y_train)

    # 5. Model Evaluation
    y_pred = pipeline.predict(X_test_raw)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("  MODEL EVALUATION BENCHMARK METRICS")
    print("=" * 55)
    print(f"  Dataset Name : {dataset_name}")
    print(f"  Train Size   : {len(X_train_raw)} articles")
    print(f"  Test Size    : {len(X_test_raw)} articles")
    print(f"  Accuracy     : {acc * 100:.2f}%")
    print(f"  Precision    : {prec * 100:.2f}%")
    print(f"  Recall       : {rec * 100:.2f}%")
    print(f"  F1-Score     : {f1 * 100:.2f}%")
    print("=" * 55)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], zero_division=0))

    # 6. Confusion Matrix Computation & Visualization
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["FAKE", "REAL"],
            yticklabels=["FAKE", "REAL"],
            linewidths=0.5,
            linecolor="gray",
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_title("Confusion Matrix — Fake News Detection", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(CM_IMG_OUT, dpi=150)
        plt.close(fig)
        logger.info(f"Confusion matrix plot generated: {CM_IMG_OUT}")
    except Exception as e:
        logger.warning(f"Could not render confusion matrix figure: {e}")

    # 7. Model Persistence
    # Save the full unified Pipeline as models/fake_news_model.pkl
    joblib.dump(pipeline, MODEL_OUT)
    # Also save the fitted vectorizer separately for direct inspection
    joblib.dump(pipeline.named_steps["tfidf"], TFIDF_OUT)

    fitted_vocab_size = int(len(pipeline.named_steps["tfidf"].vocabulary_))

    metadata = {
        "dataset_name": dataset_name,
        "is_fallback": is_fallback,
        "total_samples": len(df),
        "train_samples": int(len(X_train_raw)),
        "test_samples": int(len(X_test_raw)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "model_type": "Logistic Regression (L-BFGS solver)",
        "features": "TF-IDF (Unigrams + Bigrams, Sublinear TF)",
        "vocabulary_size": fitted_vocab_size,
    }

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 55)
    print("  MODEL PERSISTENCE COMPLETE")
    print("=" * 55)
    print(f"  Exact Model Path      : {MODEL_OUT.resolve()}")
    print(f"  Exact Vectorizer Path : {TFIDF_OUT.resolve()}")
    print(f"  Exact Metadata Path   : {METADATA_OUT.resolve()}")
    print("=" * 55 + "\n")

    return pipeline, pipeline.named_steps["tfidf"], metadata


if __name__ == "__main__":
    train_model()
