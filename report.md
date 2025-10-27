<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<a id="top"></a>

[Back to Home](index.md)

# Using Twitter Sentiment Analysis to Predict Stock Price Movements of Major Companies via Support Vector Machines

Authors: Chris Charo, Hunter Clayton, and Camille Esteves

## 1. Introduction

### 1.1 Overview of Sentiment Analysis
Sentiment analysis is the process of analyzing text to derive the underlying emotions being expressed in a collection of data. It has a wide range of applications, including understanding customer satisfaction metrics, brand reputation, and marketing campaign responses. One area of sentiment analysis that has received significant attention is its relationship with public sentiment and the valuation of publicly traded companies and broader stock market indexes. With the S&P 500 Index currently boasting an aggregate market capitalization of $57.4 trillion, an advantage in predicting the price movements of publicly traded stocks would be highly valuable for investors [(S&P Dow Jones Indices 2025)](#ref9). One of the most popular outlets for real-time public sentiment is the social media platform Twitter(X), which has over 586 million active users in 2025 [(Statista 2025)](#ref10). Twitter(X) differentiates itself from other platforms through short-form content of 280 characters or fewer, allowing rapid communication of opinions, news, and humor [(Chaires 2024)](#ref10). This concise format provides a valuable source of global sentiment data across generations, cultures, and regions.
Recent studies have shown that machine learning models such as SVM, LSTM, and ensemble methods like XGBoost are increasingly used in financial sentiment analysis, particularly on Twitter(X) datasets. Across the literature, SVM has repeatedly demonstrated superior performance in classifying sentiment polarity for short-text data such as tweets, due to its robustness in high-dimensional feature spaces. Recent research also highlights how combining these models with deep learning architectures can improve predictive accuracy when dealing with social media text data [(Financial Sentiment Analysis: Techniques and Applications 2024)](#ref7).

### 1.2 Importance of Twitter Data in Financial Forecasting
Every day, millions of users share their perspectives on nearly every conceivable topic, including brands and companies. This constant flow of opinionated data makes Twitter(X) an invaluable resource for analyzing trends in public sentiment. In financial markets, these sentiments can serve as indicators of investor confidence, public reaction to corporate events, and potential short-term price fluctuations. Prior studies demonstrate that analyzing sentiment from Twitter posts can effectively reflect investor sentiment and broader market trends [(Twitter Sentiment Analysis and Bitcoin Price Forecasting 2023)](#ref11).

### 1.3 Support Vector Machines (SVM) and Research Rationale
One of the most widely used models for sentiment analysis is Support Vector Machines (SVM). SVMs are a class of supervised machine learning models that excel at classification by formulating an optimal decision boundary, or 'hyperplane,' to separate data into distinct classes. For unstructured text data commonly used in sentiment analysis, this function is crucial because SVM effectively handles multi-dimensional data that cannot be classified linearly. By employing kernel methods, the model can transform non-linearly separable data into a higher-dimensional space to improve accuracy and reduce overfitting [(Du et al. 2024)](#ref6). Notably, prior research has shown that SVM provides the highest accuracy for sentiment analysis classification when predicting stock market movement [(Chakraborty et al. 2017; Kolasani and Assaf 2020)](#ref4). Based on these findings, SVM is selected as the primary model for sentiment analysis in this study. Similarly, comparative analyses have confirmed SVM’s consistency and reliability over other algorithms for sentiment classification tasks in finance [(A Comparative Study of Sentiment Analysis on Customer Reviews 2023)](#ref1).
Despite extensive research in financial forecasting, accurately linking real-time public sentiment to stock movements remains a challenge due to data noise, linguistic ambiguity, and market volatility. This study aims to bridge that gap by combining sentiment classification on Twitter(X) data with market performance analysis for major NASDAQ companies. Several studies have addressed these challenges by proposing improved preprocessing and data filtering methods to enhance classification accuracy [(A Scoping Review of Preprocessing Methods 2023)](#ref2).
The goal of this study is to determine whether sentiment polarity from social media can serve as a reliable predictor of stock performance, using Support Vector Machines as the primary analytical model.
### 1.4 Datasets and Model Implementation
The dataset used for sentiment analysis is titled 'Tweets about the Top Companies from 2015 to 2020,' created by Doğan, Metin, Tek, Yumuşak, and Öztoprak for the IEEE International Conference [(Doğan et al. 2020)](#ref5). This dataset comprises over three million tweets collected using a Selenium-based parsing script. Once sentiment classification is completed, the results will be used to predict the stock price movements of four companies: Amazon, Apple, Google, and Microsoft. The stock data comes from 'Values of Top NASDAQ Companies from 2010 to 2020,' sourced directly from the NASDAQ website and hosted on Kaggle [(Doğan et al. 2020)](#ref5). In addition to SVM, models such as XGBoost, Long Short-Term Memory (LSTM), and Support Vector Regression (SVR) will be tested to evaluate their performance in predicting price trends.
A review of recent literature reveals consistent findings that Support Vector Machines remain among the most effective models for sentiment classification, especially in financial and social media contexts. Across multiple studies, SVM outperformed deep learning alternatives such as CNNs and RNNs in accuracy and generalization, particularly when applied to short, unstructured text data from platforms like Twitter. This supports the decision to employ SVM for this study’s sentiment analysis and stock prediction framework.
### 1.5 Paper Roadmap
The remainder of this paper outlines the methodology for data processing and model training, results of the sentiment and stock prediction analyses, and a discussion of findings with implications for financial forecasting. Overall, this study builds upon a growing body of literature that connects natural language processing, financial analytics, and social media data to improve predictive modeling in stock performance forecasting.

## 2. Methods

### 2.1 Data Collection and Integration
Building upon the datasets and analytical framework described in the introduction, this section outlines the procedures for data preparation and model implementation. This study utilizes two primary datasets to investigate the relationship between social media sentiment and stock market performance. The first dataset, titled 'Tweets about the Top Companies from 2015 to 2020,' was created by Doğan, Metin, Tek, Yumuşak, and Öztoprak for the IEEE International Conference [(Doğan et al. 2020)](#ref5). This dataset contains over three million tweets gathered using a Selenium-based web scraping tool designed to collect text data from Twitter(X). Each record includes tweet text, timestamp, company reference, and engagement attributes such as likes and retweets. The dataset serves as the foundation for sentiment analysis, enabling the classification of public opinion related to four major NASDAQ companies: Amazon, Apple, Google, and Microsoft.

The second dataset, titled 'Values of Top NASDAQ Companies from 2010 to 2020,' was sourced directly from the NASDAQ website and hosted on Kaggle [(Doğan et al. 2020)](#ref5).. It includes daily historical stock prices—opening, closing, high, low, and volume data—for the same four companies. To ensure analytical consistency, both datasets were merged based on overlapping date ranges. The merged dataset was further refined by removing extraneous or non-essential columns such as usernames, retweet indicators, and reply metadata. Before integration, data types were standardized, and timestamps were aligned to maintain temporal accuracy between social sentiment and corresponding stock data.

All data processing and integration steps were performed in Python using libraries such as Pandas for data manipulation and NumPy for numerical operations. This ensured efficient handling of large data volumes, consistent with previous studies in financial sentiment analysis [(Chakraborty et al. 2017; Kolasani and Assaf 2020)](#ref4). Merging the sentiment and market datasets into a single framework enables the study to evaluate whether fluctuations in public sentiment correspond with short-term changes in stock price trends.

### 2.2 Data Preprocessing
Preprocessing is a critical step in preparing unstructured text data for machine learning applications, especially when working with social media content that is often noisy, informal, and contextually ambiguous [(A Scoping Review of Preprocessing Methods 2023)](#ref2). The preprocessing phase for this project focused primarily on the textual component of the Twitter(X) dataset. Each tweet underwent a structured series of cleaning and transformation steps. Specifically, all text was first converted to lowercase to ensure consistency across the dataset. Subsequently, URLs, user mentions, hashtags, and stock symbols were removed, followed by tokenization based on whole words. Stop words were eliminated, with the exception of negation words such as 'not' and 'no,' which are critical in determining sentiment polarity. Finally, lemmatization was applied to reduce each word to its base form, allowing the Support Vector Machine (SVM) model to better capture semantic meaning and context [(Du et al. 2024)](#ref6).

Following these transformations, tokenized tweets were reviewed for outliers, duplicates, and entries lacking meaningful textual content. This filtering step ensured that only relevant, high-quality data remained. Lemmatization, performed using the Natural Language Toolkit (NLTK), helped maintain linguistic consistency while preserving grammatical meaning. These preprocessing methods align closely with those recommended in prior research emphasizing the importance of context retention and dimensionality reduction in text classification tasks [(Financial Sentiment Analysis: Techniques and Applications 2024)](#ref7). The finalized clean dataset was stored in a Pandas DataFrame, ready for subsequent feature extraction and sentiment classification.

Python libraries such as NLTK, scikit-learn, and re (regular expressions) were used extensively throughout preprocessing. The process mirrors common best practices identified in previous literature on text-based sentiment analysis [(Financial Sentiment Analysis: Techniques and Applications 2024)](#ref7). This systematic cleaning approach ensures that the subsequent machine learning stages operate on standardized, high-quality input data. Pre-processing the Tweet data prior to training the SVM model is vital to achieving adequate accuracy of true sentiment expressed in each tweet.

### 2.3 Dataset Visualizations

#### Tweet Volume Over Time

```python
plt.figure(figsize=(14, 6))
finished_dataset['date'] = pd.to_datetime(finished_dataset['date'])
tweets_per_month = finished_dataset.groupby(finished_dataset['date'].dt.to_period('M')).size()
tweets_per_month.plot(kind='line', color='skyblue', linewidth=2)
plt.title('Tweet Volume Over Time (2015-2019)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Tweets', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

![Tweet Volume Over Time](<./images/Tweet-volume-over-time.png>)

#### Average Stock Closing Prices Over Time

```python
plt.figure(figsize=(14, 6))
top_companies = finished_dataset['ticker_symbol'].value_counts().head(4).index # Top four to match the last visualization
for company in top_companies:
    company_data = finished_dataset[finished_dataset['ticker_symbol'] == company]
    plt.plot(company_data.groupby('date')['close_value'].mean(), label=company, alpha=0.7)
plt.title('Average Stock Closing Prices Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

![Average Stock Closing Prices Over Time](<./images/Stock-price-over-time.png>)

#### Average Stock Prices vs. Tweet Volume

```python
# Average stock prices at closing overlaid ontop of tweet volume
# I'm looking for correlation between tweet volume and stock price swings

sns.set_style("dark")
plt.style.use("dark_background")

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
axes = axes.flatten()

# Get top 4 companies by tweet volume
top_companies = finished_dataset['ticker_symbol'].value_counts().head(4).index

for idx, company in enumerate(top_companies):
    ax1 = axes[idx]

    # Filter the data for each company
    company_data = finished_dataset[finished_dataset['ticker_symbol'] == company].copy()

    # Group by date for tweet volume and average stock price
    daily_data = company_data.groupby('date').agg({
        'tweet_id': 'count',  # Count the tweets
        'close_value': 'mean'  # Average the closing price
    }).reset_index()

    # Create the dual axis since we are overlaying two graphs
    ax2 = ax1.twinx()

    # Plot tweet volume as bars
    ax1.bar(daily_data['date'], daily_data['tweet_id'], alpha=0.2, color='skyblue', label='Tweet Volume')
    ax1.set_ylabel('Tweet Volume', fontsize=11, color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Plot stock price as line
    ax2.plot(daily_data['date'], daily_data['close_value'], color='darkred', linewidth=2, label='Stock Price')
    ax2.set_ylabel('Closing Price ($)', fontsize=11, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')

    # Formatting
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_title(f'{company}: Tweet Volume vs Stock Price', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.8)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()
```

![Average Stock Prices vs. Tweet Volume](<./images/tweet-volume-stock-price.png>)

### 2.4 Model Development and Training
The cleaned and pre-processed data is now ready for training and testing using SVM. SVM functions by calculating the optimal hyperplane that separates the data into their respective categories. For our linear SVM model, the formula can be seen below in figure 5.

$$
f(x) = w_1 x + w_2 x^2 + \dots + w_n x^n + b
$$

*Figure 5. Polynomial function*

The features ($x_i$) are the individual words from the body of the tweet separated during tokenization. For instance, an example Tweet ‘Apple keeps going up!’ would become ‘Apple’ - ‘keeps’ - ‘going’ - ‘up’ - ‘!’ where each word is designated as an individual input variable. The weights ($w_i$) indicate the contribution of each word to the sentiment prediction: positive values suggest positive sentiment, negative values indicate negative sentiment, and values near zero correspond to neutral sentiment. Finally, the bias or intercept (b) functions as a baseline shifting the hyperplane to optimally separate the classes in the feature space (Montesinos, 2022).

### 2.5 Evaluation Metrics
We will evaluate the performance of the SVM model using a variety of methods: accuracy, precision, recall, and F-1 score. Accuracy score is the percentage of true positives, true negatives, and true neutrals correctly identified by the model.

$$
\text{Accuracy Score} = \frac{\text{True Positives} + \text{True Negatives} + \text{True Neutrals}}{\text{Total Predictions}}
$$

*Figure 6. Accuracy Score formula*

While accuracy is an important metric, additional methods will be required to understand the model's true ability to classify tweet sentiment. As opposed to accuracy, the remaining evaluation metrics will need to be separated by class to ensure that positive, negative, and neutral predictors are equally represented. The scores of each predictor category will then be averaged and weighted to provide an overall evaluation.

Precision score measures the model’s ability to correctly identify true positives. Of all the actual positives, how many were predicted positive?

$$
\text{Precision (Per Class)} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

*Figure 7. Precision formula per class*

Recall score measures the model’s ability to correctly identify a tweet as a true positive, negative, or neutral or a false positive, negative, or neutral. Of all the predictions, how many were actually correct?

$$
\text{Recall (Per Class)} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

*Figure 8. Recall formula per class*

F-1 Score measures the balance between precision and recall. Does the model balance all predictions or does it favor one type?

$$
\text{F1 (Per Class)} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

*Figure 9. F1 Score formula per class*

### 2.7 Making Predictions
Once sentiment analysis using the SVM model is complete, we will progress to making predictions on the stock price of our selected companies: Apple, Amazon, Google, and Microsoft. A variety of models will be tested including Extreme Gradient Boosting (XGBoost), Long Short-Term Memory (LSTM), and Support Vector Regression (SVR) with the ultimate goal of achieving a model that can adequately predict the closing price of a stock the following day based on tweets from the prior day. 

#### Extreme Gradient Boosting
XGBoost is a commonly used, scalable, and powerful gradient boosting-based machine learning algorithm. XGBoost surpasses other decision tree-based models by employing gradient boosting, which functions by starting with one tree and sequentially building additional trees that correct the errors of previous ones. Additionally, Lasso and Ridge regularization techniques are incorporated to balance complexity while minimizing overfitting. While XGBoost has a multitude of formula’s, the core mathematical equation, the Objective Function, is detailed in figure 6 (Chen, 2016).

$$
\text{Obj}(\Theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

*Figure 10. XGBoost objective function*

#### Long Short-Term Memory
LSTM is a Recurrent Neural Network (RNN)-based deep learning algorithm that is widely used in the financial industry for time series forecasting, making it a fitting choice for predicting stock prices. LSTM improves upon traditional RNN algorithms by solving two issues: the vanishing gradient problem,where the model struggles to retain long-term dependencies, and the exploding gradient problem, where the model keeps too much information, causing instability during training. LSTM solves these problems by controlling the memory cell using three gates: The input gate, forget gate, and output gate. This allows LSTM to utilize important long term memory while discarding irrelevant memory to improve efficiency and model performance (Qin, 2023). The structure of the cell using all three gates can be seen below in figure 7.

![LSTM Memory Cell Gates Diagram](<./images/lstm.png>)

*Figure 11. LSTM Memory Cell Gates Diagram*

## References

<a id="ref1"></a> A Comparative Study of Sentiment Analysis on Customer Reviews Using Machine Learning and Deep Learning. 2023. *Computers* 13, no. 12 (340). https://www.mdpi.com/2073-431X/13/12/340 [\[Back to Top\]](#top)  

<a id="ref2"></a> A Scoping Review of Preprocessing Methods for Unstructured Text Data to Assess Data Quality. 2023. *PLOS ONE.* https://pmc.ncbi.nlm.nih.gov/articles/PMC10476151/ [\[Back to Top\]](#top)  

<a id="ref3"></a> Chaires, Rita. 2024. “Ultimate Social Media Cheat Sheet: Character Limits & Best Days/Times to Post”. *American Academy of Estate Planning Attorneys,* February 6, 2024. https://www.aaepa.com/2022/05/ultimate-social-media-cheat-sheet-character-limits-best-days-times-to-post [\[Back to Top\]](#top)  

<a id="ref4"></a> Chakraborty, P., U. S. Pria, M. R. A. H. Rony, and M. A. Majumdar. 2017. “Predicting Stock Movement Using Sentiment Analysis of Twitter Feed.” In *Proceedings of the 2017 6th International Conference on Informatics, Electronics and Vision (ICIEV),* 1–6. Himeji, Japan. https://doi.org/10.1109/ICIEV.2017.8338584 [\[Back to Top\]](#top)  

<a id="ref5"></a> Doğan, M., Ö. Metin, E. Tek, S. Yumuşak, and K. Öztoprak. 2020. “Speculator and Influencer Evaluation in Stock Market by Using Social Media.” In *Proceedings of the 2020 IEEE International Conference on Big Data (Big Data),* 4559–4566. Atlanta, GA. https://doi.org/10.1109/BigData50022.2020.9378170 [\[Back to Top\]](#top)  

<a id="ref6"></a> Du, Ke-Lin, Bingchun Jiang, Jiabin Lu, Jingyu Hua, and M. N. S. Swamy. 2024. “Exploring Kernel Machines and Support Vector Machines: Principles, Techniques, and Future Directions.” *Mathematics* 12, no. 24: 3935. https://doi.org/10.3390/math12243935 [\[Back to Top\]](#top)  

<a id="ref7"></a> Financial Sentiment Analysis: Techniques and Applications. 2024. *ACM Computing Surveys.* https://dl.acm.org/doi/pdf/10.1145/3649451 [\[Back to Top\]](#top)  

<a id="ref8"></a> Kolasani, Sai Vikram, and Rida Assaf. 2020. “Predicting Stock Movement Using Sentiment Analysis of Twitter Feed with Neural Networks.” *Journal of Data Analysis and Information Processing* 8 (4): 309–319. https://doi.org/10.4236/jdaip.2020.84018 [\[Back to Top\]](#top)  

<a id="ref9"></a> S&P Dow Jones Indices. 2025. “S&P 500®.” Accessed October 2, 2025. https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview [\[Back to Top\]](#top)  

<a id="ref10"></a> Statista. 2025. “Most Used Social Networks 2025, by Number of Users.” March 26, 2025. https://www.statista.com/statistics/272014/global-social-networks-ranked-by-number-of-users [\[Back to Top\]](#top)  

<a id="ref11"></a> Twitter Sentiment Analysis and Bitcoin Price Forecasting: Implications for Financial Risk Management. 2023. *ProQuest.* https://www.proquest.com/scholarly-journals/twitter-sentiment-analysis-bitcoin-price/docview/3047039752/se-2 [\[Back to Top\]](#top)  

 
## Glossary
Twitter(X) — In 2022, Twitter was acquired and re-branded to X. The dataset used for this report include data from pre and post acquisition therefore we have chosen to combine the two terms.  

Tweets(Posts) — Post acquisition, Tweets were renamed to Posts on X.  

SVM — Support Vector Machines  

[Back to Home](index.md)
