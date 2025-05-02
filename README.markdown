# Spam Detection Project

This project develops a machine learning model to classify SMS or email messages as spam or ham (non-spam) using a Multinomial Naive Bayes classifier. The model is deployed as a Streamlit web application for real-time predictions. The dataset used is `spam_dataset.csv`, containing labeled messages.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal is to build an accurate spam detection system that minimizes false positives (ham messages classified as spam). The project includes:
- Data cleaning and exploratory data analysis (EDA).
- Text preprocessing (lowercasing, tokenization, stemming, etc.).
- Model training with Multinomial Naive Bayes.
- Deployment as a Streamlit app for user interaction.

## Features
- Classifies messages as spam or ham with high precision (100% for spam).
- Displays spam and ham probabilities.
- Interactive Streamlit app with a simple text input interface.
- Visualizations (word clouds, histograms) for data insights.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`:
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - streamlit
  - matplotlib
  - seaborn
  - wordcloud
  - swifter

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spam-detection.git
   cd spam-detection
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   Run the following in a Python shell:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage
1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   This opens the app in your default browser (typically at `http://localhost:8501`).

2. **Interact with the app**:
   - Enter a message in the text area.
   - Click "Predict" to classify the message as spam or ham.
   - View the result and probabilities.

3. **Explore the notebook**:
   Open `spamDetection.ipynb` in Jupyter Notebook to review the data analysis, preprocessing, and model training steps:
   ```bash
   jupyter notebook spamDetection.ipynb
   ```

## Project Structure
```
spam-detection/
├── spamDetection.ipynb    # Jupyter Notebook with project code
├── app.py                # Streamlit app for deployment
├── vectorizer.pkl        # Saved TF-IDF vectorizer
├── model.pkl             # Saved Multinomial Naive Bayes model
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── report.tex            # LaTeX project report
└── .gitignore            # Git ignore file
```

**Note**: The `spam_dataset.csv` file is not included in the repository due to its size. You can obtain it from [source, e.g., Kaggle] or contact the repository owner.

## Results
- **Model Performance**:
  - Accuracy: 97.8%
  - Precision (spam): 100%
- **Key Insights**:
  - Spam messages often contain words like "free," "call," and "win."
  - Ham messages are longer and contain conversational words like "go," "know."
- **Deployment**: The Streamlit app provides a user-friendly interface for real-time spam detection.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes relevant tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.