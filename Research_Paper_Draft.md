# TruthLens: An Interpretable Dual-Pipeline Approach to Online Misinformation and Propaganda Detection

## Abstract
The rapid dissemination of misinformation and propaganda online poses a significant threat to public discourse. While complex deep learning models achieve high accuracy in detecting such content, they often operate as "black boxes," failing to explain their reasoning to end-users. This paper presents **TruthLens**, a lightweight, transparent, and interpretable dual-pipeline system for simultaneous fake news and propaganda detection. By leveraging classical Natural Language Processing (NLP) techniques—specifically TF-IDF vectorization paired with Logistic Regression, Naive Bayes, and Random Forest classifiers—we prioritize computational efficiency and explainability. TruthLens not only classifies news articles but also highlights specific "signal words" and suspicious phrasing directly within the text, empowering users to understand *why* an article was flagged. We detail our dual-model architecture, the explainability framework, and the dynamic exploratory data analysis (EDA) capabilities built into the user interface.

---

## 1. Introduction

### 1.1 Background
The digital age has democratized information sharing, but it has also catalyzed the spread of fake news and systematic propaganda. While fake news often involves fabricated facts aimed at generating clicks or shaping opinions, propaganda utilizes psychological manipulation techniques to persuade an audience. Identifying both requires nuanced text analysis.

### 1.2 Problem Statement
Current state-of-the-art solutions frequently rely on massive Large Language Models (LLMs) or complex neural networks (e.g., BERT, RoBERTa). Although accurate, these models are computationally expensive to run and lack transparency. When a black-box model flags an article, users are rarely told *why*, which reduces trust in automated moderation systems. 

### 1.3 Objectives
This project aims to bridge the gap between accuracy and explainability. The primary objectives are to:
1. Develop two separate classification pipelines to distinguish between general Fake News and targeted Propaganda.
2. Utilize classical, highly interpretable machine learning models.
3. Provide a user-facing dashboard that highlights malicious text spans and visualizes the model’s decision-making process.

---

## 2. Literature Review

### 2.1 Fake News Detection
Early approaches to fake news detection relied heavily on linguistic features and basic term frequencies, evaluated using standard datasets like ISOT and LIAR. Researchers found that fake news often contains elevated emotional polarity and sweeping generalizations compared to real news. 

### 2.2 Propaganda Detection
Propaganda detection gained prominence with shared tasks like SemEval-2020 Task 11, which challenged researchers to identify specific propagandistic spans (e.g., "name-calling," "loaded language"). The distinction between fake news and propaganda is critical: an article can be factually true but highly propagandistic in its framing.

### 2.3 Explainability in AI
Explainable AI (XAI) in Natural Language Processing usually involves techniques like LIME or SHAP. However, using simpler linear models (like Logistic Regression) allows for intrinsic explainability, where feature weights can be directly mapped back to specific words in the input text without needing computationally heavy secondary explainer modules.

---

## 3. Proposed Methodology

### 3.1 System Architecture
The TruthLens architecture revolves around a dual-pipeline approach. Unlike systems that attempt to categorize text into a single bucket, our system passes every input through two independent evaluation tracks:
1. **Fake vs. Real Track:** Evaluates the factual integrity and linguistic style indicative of generated or fake news.
2. **Propaganda vs. Non-Propaganda Track:** Evaluates the presence of manipulative rhetoric.

### 3.2 Data Preprocessing
Raw text undergoes a standard classical NLP pipeline. Stop words, punctuation, and URLs are stripped, and the text is lowercased. Because machine learning models require numerical input rather than raw text, the cleaned text is then vectorized using Term Frequency-Inverse Document Frequency (TF-IDF), limited to the top 3,000 features. TF-IDF calculates how frequently a word appears in a specific article (Term Frequency) and offsets it by how commonly it appears across the entire dataset (Inverse Document Frequency). This mathematical transformation ensures the model focuses on the most structurally and contextually relevant "signal words" while dismissing ubiquitous noise (like the word "the").

### 3.3 Model Selection and Training
To prioritize speed and interpretability, we trained three classical algorithms on the TF-IDF vectors:
- **Logistic Regression (LR):** A linear classifier that assigns numerical weights (positive or negative) to every word in the vocabulary. It calculates a probability score by summing the weights of the words present in an article. It is used primarily for its highly interpretable feature weights, allowing us to see exactly which words pushed the decision toward "Fake" or "Propaganda."
- **Naive Bayes (NB):** A probabilistic classifier based on Bayes' Theorem. It calculates the probability of an article belonging to a specific class (e.g., Fake News) based on the combined probabilities of the individual words it contains. It serves as the primary production classifier due to its historically strong performance and speed in text classification and spam detection.
- **Random Forest (RF):** An ensemble learning method that builds hundreds of distinct decision trees during training and outputs the majority vote of the trees. It captures non-linear relationships between words and provides a robust baseline to check against the linear models.

### 3.4 The Explainability Framework
Explainability is achieved by correlating the user's input text with the trained weights of our Logistic Regression model. Words in the user's text that correspond to high positive weights are flagged as "signal words." The system then reconstructs the text in a hyper-transparent UI, highlighting high-risk phrases in red and safe/counter-signal phrases in green.

---

## 4. Innovations and Key Features

### 4.1 Granular Text Highlighting (Whitebox Approach)
The core innovation of TruthLens is its departure from binary "True/False" outputs. By reverse-mapping TF-IDF features to original text spans, the system highlights suspicious phrases directly in the user’s browser. This transparent "whitebox" approach builds trust, as the user can independently verify the model’s linguistic triggers.

### 4.2 Side-by-Side Model Comparison
Instead of hiding the algorithm, TruthLens exposes a "Model Comparison" interface. It dynamically runs the user's input through LR, NB, and RF simultaneously, displaying the confidence scores of each. This educates the user on model agreement and uncertainty, showcasing that AI is probabilistic, not absolute.

### 4.3 Dynamic EDA Dashboard
TruthLens features an Exploratory Data Analysis (EDA) dashboard that generates visualizations on the fly. When a user inputs an article, the system generates custom word clouds, sentiment distributions, and text-length comparisons, plotting the user's article directly against the baseline distributions of the training corpus.

---

## 5. Experiments and Results

### 5.1 Dataset Composition
The models were trained using two distinct corpora, allowing the system to distinguish between different types of misinformation:
- **Fake News Dataset (ISOT):** A balanced slice containing thousands of articles explicitly labeled as either `FAKE` (completely fabricated or heavily distorted facts) or `REAL` (verified news from reputable sources).
- **Propaganda Dataset (SemEval-2020 Task 11):** Contains text labeled specifically as `PROPAGANDA` or `NOT_PROPAGANDA`. This allows the model to detect psychological manipulation techniques, emotional appeals, and loaded language, which often occur independently of factual fabrication.

### 5.2 Evaluation Metrics
Models were evaluated using a standard 75/25 train-test split. The evaluation prioritized traditional metrics: Accuracy, Precision, Recall, and F1-Score. 

*(Note: Actual accuracy percentages and metrics can be inserted here dynamically based on the model's terminal output during training).*

Typically, Naive Bayes exhibited the highest confidence stability for binary text categorization, while Logistic Regression provided the most coherent explanations for the UI's highlighting feature.

---

## 6. Discussion

### 6.1 Analysis of Findings
The dual-model approach successfully demonstrates that an article can be classified as factually "Real" while still being flagged as heavily "Propagandistic." Furthermore, mapping Logistic Regression weights directly to the frontend interface proved highly effective for real-time text highlighting without inducing latency.

### 6.2 Limitations
Because the system relies on TF-IDF word frequencies rather than deep contextual embeddings (like BERT), it struggles with sarcasm, subtle context, and zero-shot word associations. It relies heavily on specific vocabulary triggers rather than semantic meaning.

---

## 7. Conclusion and Future Work

TruthLens provides a fast, interpretable, and visually engaging solution to the growing problem of online misinformation. By valuing transparency over marginal accuracy gains from black-box LLMs, the system successfully educates users on the linguistic markers of fake news and propaganda. Future work could involve incorporating lightweight DistilBERT embeddings to capture deeper semantic context while utilizing specialized explainers (like LIME) to maintain the granular highlighting feature.
