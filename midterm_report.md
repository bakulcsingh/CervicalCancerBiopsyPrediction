# Midterm report

**Description of Data**

The dataset we used originally had 36 features and 858 examples of patients. 25 of the features are categorical (binary) whereas the remaining have discrete data regarding the patient. The main behavioral indicators (based on our research and a few interviews with doctors) of cervical cancer that the dataset covers are the number of pregnancies, contraceptive use, sexually transmitted diseases (STDs), smoking history, intra-uterine disease history and hinselmann test that are used by medical practitioners to recommend a biopsy. Our dataset contains a column titled Biopsy, whose value tells us whether a biopsy was recommended for this patient or not. Our goal is to predict the value of this field for new patients, since a biopsy in an expensive and an invasive procedure. Since biopsies are only recommended in cases when cancer is strongly suspected, we plan on using features which are commonly considered risk factors for Cervical Cancer to decide whether to recommend a biopsy or not.

We have included a few visualisations and some descriptive statistics to better understand the dataset in the repository.
Most of the women in the dataset are under the age of 40 and the average age is a little less than 27.

The average number of sexual partners that the women in the dataset have had are 2 and they range from 1 to 14. The range seems large but 97% of the women have had 6 or less partners.The average number of pregnancies in our dataset are 2 and they vary from 0 to 11. About half of the women in the dataset have had more between 2 and 4 pregnancies.

About 60% of the women in the set have reported using hormonal contraceptives. The average number of years of usage are 2 and range from 0 to 30. 60% of the women have used the contraceptives for upto 1 year.

**Ensuring optimal fit**

**Overfitting**

Cervical Cancer affects a very small percentage of the female population and this is clear from our dataset as well because of the 858 patients only 55 have been recommended a biopsy. This presents the challenge of making sure that the test errors are not low as a result of the model having overfit the data. The model would overfit in case we use a model that is more complex than is required. For example we initially tried an SVM with an rbf kernel because we suspected the data would not be linearly separable but after seeing that even a perceptron is fitting well we plan to change the kernel to a linear one and see how that works out. Moreover, we plan to be wary of low training errors and will cross validate our results for all the model classes that perform well on our training set to ensure that we choose the one that generalizes well. We are also looking for datasets for cervical cancer from other geographies like Europe, Africa and Asia with the same or at least similar features. This will help us get a better idea of how the model generalizes to other populations.

Another strategy that we have employed to ensure that the model does not overfit because of the data being small and imbalanced towards biopsies not being recommended is to assign a higher weight to a false negative as compared with a false positive so that the error presents a clearer picture of model performance.

**Underfitting**

Since the dataset that we are using is small this was initially a bigger concern. But after fitting a few different models, cross validating them and comparing the errors we saw that a linear classifier was working well in terms of cross validation error and realised that our initial hypothesis was incorrect and the data was indeed linearly separable. 

However we plan to engineer new features based on the weights each has been assigned and its rank in being selected as a split in case of random forests. We can include higher order transformations of existing features to see if the model shifts weight to them. Features that we feel interact in some way can be transformed using a product or some other function that we feel better defines the relationship. For example, the number of pregnancies most likely increase the risk of cancer non linearly (based on research and the weights of the classifier and random forest splits) so we will be transforming it using a non-linear function like its second or third power.

**Data Cleaning**

The dataset had ‘?’ for a lot of women across attributes. We could ignore these datapoints because our dataset was not very large to start with. Moreover the rows of entries for which biopsies were recommended were already very less and we could not remove any of these. 

We managed to clean the data by replacing the ‘?’ with values that would result in the least bias during classification and be consistent with the labels. For example: in the number of pregnancies we divided the incorrect entries into two categories one for women who were recommended a biopsy and one for those who were not. For either of these we replaces them with the most commonly occurring entry for the classification in the remaining data. This ensured that despite having changed the entry in the field the outcome would not be affected.

We followed a similar approach for other features in order to clean the data replacing with mean, median or mode to keep it consistent with the labels and trying to add a minimal amount of bias to the data.

**Model Effectiveness**

We used the following models on our data:Perceptron, Perceptron with class weights, Random Forest, Random Forests with class weights, SVM with rbf Kernel, SVM with rbf Kernel and class weights

**Insights on Model Analysis:**

We ran the models with different proportions of the dataset.

On Running Perceptron using entire training data: 96%, 10 - fold Cross Validation using entire training data: 94.5%, Perceptron scores on data: 95.7%, 10 - fold Cross Validation on Perceptron using entire training data: 93.19%, Decision Tree using Cancerous Training Data: 16.32%, Decision Tree using Non Cancerous Training Data: 99.57%, Random Forest using using training data: 96%, 10 - fold Cross Validation on Random Forest using training data: 95.73%.Then we modified the dataset by splitting it up into 6/7ths of the cancerous data in one set and 1/7 of the non-cancerous data in one set in order to ensure consistency in proportions in every test run.

We ran Linear SVM Models using RBF Kernel

Linear SVM Model: 94.4%,10 fold Cross Validation on Linear SVM: 93.87%, Linear SVM Model on Proportional Dataset: 97.77%, 10 fold Cross Validation on Linear SVM Model on Proportional Dataset: 93.61%

**Features Description and Feature Elimination:**

Our original dataset consisted of 34 features and 858 data points. These consisted of some missing values, normally indicated by a ‘?’. After cleaning up our data and removing two columns (as described in the Data Missing and corrupted section), we were left with 32 dimensions and 858 data points. Out of 32 features,25 of the features are categorical (binary) whereas the remaining have discrete data regarding the patient. Our features consist of fields such as:

* Age of the patient

* Age of first sexual intercourse

* Number of pregnancies

* Smoking History

* Use of IUDs

* Use of Hormonal Contraceptives

* Presence of various STDs

* Results of several tests

These fields have traditionally been suspected to have a high correlation with the risk of cervical cancer. We are currently using all the features. We plan on narrowing down the number of features we use as described in the going forward section, in order to eliminate features which merely add "noise" to our data and have no correlation with the occurrence of Cervical Cancer.

We eliminated two features during data cleanup - Time since first STD diagnosis and Time since last STD diagnosis. We did this because of sparsity of values.

**Preliminary Data Analysis**

To analyze the underlying structure of our data, we employed a three pronged strategy. We first tried to plot some of our attributes using Tableau. Since our data is high dimensional, we picked some combination of two to four features and plotted them in order to observe trends like linear separability, distance based clustering, variance along different features etc. Based on above analysis, we handpicked some models and ran them on our cleaned up dataset, and used the observations to further refine our models.

One of the primary problems we ran into was the small size of our dataset, and the sparsity of data points of one class. Our dataset size, was only 858 points, but due to the extremely rare nature of a disease like Cervical cancer, we only had 55 cases where a biopsy was recommended. Due to this lopsided nature of our dataset, we discovered that our models will post high accuracy even when it does not work well. For example, say if our model is unable to identify a single cancer patient correctly, and for any given patients it recommends that they do not get a biopsy. Now if we take a test dataset, due to the rarity of cervical cancer, say only 2 out of 100 patients have cervical cancer. Our model will say no to all data points and will post an accuracy of 98% despite being faulty.

To deal with this problem, we used sklearn’s built in functionality of ‘class_weights’ for all of the models we used. We set this parameter to the value ‘balanced’. The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). This adequately accounts for the huge over-presence of non-cancerous data points in our dataset.

To further analyze our data, we decided to run different Machine Learning models on our dataset, and compare their performances with k-fold cross validation. We tried the following models (both balanced and unbalanced versions)  - Perceptron, Linear Support Vector Machine with rbf kernel and Random Forests. Since each of these models are designed to work for a different data distribution, their relative performance tells us something about the underlying data distribution. As a sanity check, we assigned a higher weight to a false negative as compared with a false positive so that the error presents a clearer picture of model performance and then evaluated the accuracy of our models on each in order to make sure that the unbalanced nature of our data had indeed been compensated for by using the ‘balanced’ versions of these models. The balanced models posted a much higher accuracy score, with both Perceptron and kernelized SVMs performing well on our dataset.

**What remains to be done and possible Extensions:**

Going forward, we plan on shifting our metric of accuracy to F-scores, and running more models on our dataset. Some models that we plan on using are:Artificial neural networks, Linear SVMs with kernels other than rbf, OneclassSVM, K-means clustering or k-nearest neighbors, Penalized perceptron, Logistic Regression

Using insights from these experiments, we will select our final model. We also plan on cutting down on the features we are using. We have identified primarily two techniques for this:

1. Principal Component Analysis - this technique identifies features which are most responsible for the variance in labels

2. Analyzing the weight vector obtained from Perceptron/Linear SVM and eliminating features which have low weights assigned to them

We also plan on finding other datasets with Cervical Cancer data to compensate for the small size of our dataset.

