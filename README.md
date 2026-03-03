# SVM Sentiment Analysis

A sentiment analysis system for English text classification using Support Vector Machine (SVM). This project implements a complete text classification pipeline with preprocessing, feature extraction, and model training.

## Overview

This project uses a Linear Support Vector Machine (LinearSVC) to classify text sentiment. It processes raw text data through a comprehensive NLP pipeline including tokenization, lemmatization, and TF-IDF feature extraction to build an accurate sentiment classifier.

## Features

- **Text Preprocessing Pipeline**
  - Lowercase normalization
  - Word tokenization using NLTK
  - Part-of-speech (POS) tagging
  - WordNet lemmatization
  - Stop words removal
  - Non-alphabetic token filtering

- **Feature Engineering**
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
  - Configurable maximum features (default: 4000)

- **Machine Learning Model**
  - Linear Support Vector Machine (LinearSVC)
  - Automatic label encoding
  - Train/test split validation

## Project Structure

```
SVM_Sentiment/
├── main.py          # Main script with training pipeline
├── corpus.csv       # Training dataset (text and labels)
├── model.pkl        # Trained SVM model (generated after running)
└── README.md        # Project documentation
```

## Requirements

### Dependencies

| Library | Purpose |
|---------|---------|
| pandas | Data manipulation and CSV handling |
| numpy | Numerical operations |
| scikit-learn | Machine learning algorithms and metrics |
| nltk | Natural language processing tasks |
| joblib | Model serialization |

### Installation

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn nltk joblib
```

Or using conda:

```bash
conda install pandas numpy scikit-learn nltk joblib
```

### NLTK Data Setup

After installing NLTK, download the required data packages:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## Dataset Format

The `corpus.csv` file should contain at least two columns:

| Column | Description |
|--------|-------------|
| `text` | The text content to analyze |
| `label` | Sentiment label (e.g., `__label__1` for negative, `__label__2` for positive) |

Example:
```csv
text,label
"This product is amazing!",__label__2
"Terrible experience, would not recommend.",__label__1
```

## Usage

### Training the Model

Simply run the main script:

```bash
python main.py
```

The script will:
1. Load the corpus from `corpus.csv`
2. Preprocess the text data
3. Split into training (70%) and testing (30%) sets
4. Extract TF-IDF features
5. Train the LinearSVC model
6. Evaluate and display accuracy
7. Save the trained model to `model.pkl`

### Model Performance

After training, the model displays:
- Classification accuracy percentage
- Training time in seconds

## Algorithm Details

### Preprocessing Pipeline

1. **Tokenization**: Splits text into individual words using NLTK's word tokenizer
2. **POS Tagging**: Assigns part-of-speech tags for accurate lemmatization
3. **Lemmatization**: Converts words to their base form using WordNet lemmatizer
4. **Filtering**: Removes English stop words and non-alphabetic tokens

### Feature Extraction

The TF-IDF vectorizer transforms text into numerical features:
- **Term Frequency (TF)**: Measures how frequently a term appears in a document
- **Inverse Document Frequency (IDF)**: Measures importance of the term across all documents
- **Max Features**: Limited to 4000 most important features

### Classification Algorithm

**Linear Support Vector Machine (LinearSVC)**:
- Optimized for high-dimensional sparse data
- Efficient for text classification tasks
- Uses linear kernel for faster training
- Well-suited for binary sentiment classification

## Model Serialization

The trained model is automatically saved as `model.pkl` using joblib for later use:

```python
import joblib

# Load the saved model
model = joblib.load('model.pkl')

# Make predictions on new data
predictions = model.predict(new_features)
```

## Customization

### Adjusting Train/Test Split

Modify the split ratio in `main.py`:

```python
train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(
    corpus['text_final'], 
    corpus['label'],
    test_size=0.2  # Change to desired ratio
)
```

### Modifying TF-IDF Parameters

Adjust feature extraction settings:

```python
tfidf = sk.feature_extraction.text.TfidfVectorizer(
    max_features=5000,      # Increase features
    ngram_range=(1, 2),     # Include bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.95             # Maximum document frequency
)
```

### Using Different SVM Parameters

Customize the LinearSVC model:

```python
svm_model = sk.svm.LinearSVC(
    C=1.0,              # Regularization parameter
    max_iter=1000,      # Maximum iterations
    class_weight='balanced'  # Handle imbalanced classes
)
```

## Performance Optimization

For large datasets, consider:
- Increasing `max_features` in TF-IDF vectorizer
- Using `ngram_range=(1, 2)` for better context capture
- Implementing cross-validation for robust evaluation
- Using `class_weight='balanced'` for imbalanced datasets

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| NLTK data not found | Run the NLTK download commands above |
| Memory error with large corpus | Reduce `max_features` or process in batches |
| Low accuracy | Try adjusting TF-IDF parameters or try different models |
| Encoding errors | Ensure corpus uses compatible encoding (default: latin-1) |

## License

This project is open source. Feel free to use and modify for your own purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- NLTK for comprehensive NLP tools
- scikit-learn for efficient machine learning implementations
- The open source community for continuous support
