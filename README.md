# Customer-Churn-Analysis



Customer churn is a problem that all companies need to monitor, especially those that depend on subscription-based revenue streams. The simple fact is that most organizations have data that can be used to target these individuals and to understand the key drivers of churn, and we would be using  Keras for Deep Learning, which predicted customer churn with 88% accuracy. 

Using the new keras package to produce an Artificial Neural Network (ANN) model on the IBM Watson Telco Customer Churn Data Set. As with most business problems, it’s equally important to explain what features drive the model, which is why we’ll use the lime package for better explanation of the model.

In addition, we use three new packages to assist with Machine Learning (ML): recipes for preprocessing, rsample for sampling data and yardstick for model metrics.

![image](https://user-images.githubusercontent.com/70890713/122206545-20d77a80-cebf-11eb-91ea-c940e0246bb0.png)
![image](https://user-images.githubusercontent.com/70890713/125238334-b91d1f80-e304-11eb-9e4b-0e82ad06fdd8.png)

All of the previous modeling and evaluation metrics were useful, but they don’t tell us much about the actual impacts on the business To start, I’ll make several assumptions related to cost. Doing a quick search, it looks like the customer acquisition cost in the telecom industry is around 300.  I’ll assume that my customer retention costs are 60.
cost = FN(300) + TP(60) + FP(60) + TN(0)

![image](https://user-images.githubusercontent.com/70890713/125238516-f5508000-e304-11eb-979d-e5fc1931a501.png)

Since the ANN model seemed to perform slightly better, I’ll use that model.
![image](https://user-images.githubusercontent.com/70890713/125238633-1ca74d00-e305-11eb-8c78-626f3e252711.png)
Formula:
![image](https://user-images.githubusercontent.com/70890713/125238639-216c0100-e305-11eb-9a2f-11486b37a7ed.png)
![image](https://user-images.githubusercontent.com/70890713/125238714-3a74b200-e305-11eb-962b-f80b2aed0e31.png)

