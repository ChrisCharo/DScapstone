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
