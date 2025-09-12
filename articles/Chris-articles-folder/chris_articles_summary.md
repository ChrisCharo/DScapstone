# Article 1
## A Comparative Study of Sentiment Analysis on Customer Reviews 
## Using Machine Learning and Deep Learning 
## by Ashbaugh and Zhang

This article was actually pretty sparse on SVM and focused more on NNs and other selected ML models. 
However, it did talk about sentiment analysis and I found the bits about the challenges faced to be 
informative. Determining human sentiment (negative or positive in this case) via text has a lot of 
hurdles to overcome. One is the multiple definitions of the same word based on the context. Another 
issue is the need for large amounts of data but also having to figure out how to manage bias and balance 
the data, so these are things our group will have to start looking at how we want to approach them. 

The bulk of the mentions on SVMs as they relate to sentiment analysis was in section 2. Related Work. 
Here the authors describe a number of different efforts at sentiment analysis using a variety of techniques, 
SVM being prominent among them. I will be looking into these other papers in the coming weeks as I select 
articles to read for this project. Noticeably, in all of these other related works, SVM came out a top performer 
on all but one of the efforts mentioned, so it seems pretty obvious it is going to be a good model for our 
application. I think the real work will come into preparing the data and tuning the model so I'm glad we have 
something like three weeks for the group to work on it. I am also very interested in "Obiedat et al. conducted 
an extensive comparison of several models regarding customer review sentiment analysis, including an SVM 
particle swarm optimization + synthetic minority over-sampling technique (SVM-PSO+BSMOTE)". 
I don't know about you, but particle swarm optimization sounds super interesting!


# Article 2
## Sentiment Analysis using Support Vector Machine and Random Forest  
## by Talha Ahmed Khan, Rehan Sadiq, Zeeshan Shahid, Muhammad Mansoor Alam, 
and  Mazliham Bin Mohd Su'udLinks to an external site.

This article was pretty much on point for what I was looking for. Specifically using SVMs for sentiment 
analysis. Since this was the stated premise my group was going after. I know that this particular tasking 
has really shifted to deep learning techniques but still thought it would be interesting to see what people 
were doing before they became mainstream and machine learning sort of ruled the roost. I was also interested 
to see how well it might perform against 1. Other ML algorithms and 2. NNs of today. I think it's worth noting 
that this article was only written in 2024 so it isn't nearly as dated as a lot of other articles I found. This 
article focuses on comparing random forests and SVMs as the title would imply. I was surprised to see that head 
to head, SVM produced a slightly more accurate model across accuracy, precision, F1 and recall. Not by a huge 
margin but over a percentage point in every benchmark. The article goes on to note that SVM is very good at 
handling high dimensional data and complex relationship and for a long time was a go to model for this 
application. It still is a work horse among ML aglorithms and will be good for our team to use since it 
doesn't require tremendous amounts of data or compute power like NNs would. 

What I really got out of this article concerning SVMs was the preprocessing steps taken and the short mention 
of kernel selection. I've only ever used the linear kernel function I believe so I'm excited to see the 
different results when trying the different kernel options. For data preprocessing, I noted some common 
sense steps such as stemming, removing stop words, reducing words to their base word (running becomes run for 
instance) and tokenization. I was also introduced to term frequency and inverse document frequency (which was 
specifically mentioned by one of our team members as what they wanted to do/try). 

# Article 3
## Title 3

# Article 4
## Title 4

# Article 5
## Title 5

# Article 6
## Title 6
