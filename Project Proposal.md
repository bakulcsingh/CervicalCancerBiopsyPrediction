# Cervical Cancer Biopsy Prediction - Proposal

## Introduction

Cervical cancer is the abnormal growth of cells in the cervix. The cervix is the lower part of the uterus that opens up into the vagina

Although Cervical Cancer is one of the most preventable type of cancer it kills about 4,000 women in the US and about 300,000 women globally. About 11,000 new cases of cervical cancer are diagnosed each year.

The dataset we are using for this project has a large number of features like the number of pregnancies, contraceptive use, sexually transmitted diseases (STDs), smoking history, intra-uterine disease history and hinselmann test that are used by medical practitioners to recommend a biopsy. We intend to encapsulate the knowledge and judgement of medical practitioners by building a linear classifier on the dataset based on whether the medical practitioner recommended a biopsy or not based on the patient’s medical profile. We can then use this classifier to recommend a biopsy if a patient’s medical profile matches the ones for which a biopsy was recommended. 

## Project Goals and Datasets
The dataset is extremely rich in features that are crucial in determining the main causes and risk factors responsible for cervical cancer. The only conclusive test to determine whether a patient has cervical cancer is to carry out a biopsy which is an extremely expensive and invasive procedure. Since the main purpose of the project is to create a model about whether a patient should get a biopsy or not, it can be beneficial for people. The main data that we have is on Age, SocioEconomic and Ethnic factors, Sexual Activity, Family History,
Use of Hormonal Contraceptives, whether they have had many children and other factors such as smoking, presence of HPV etc. All these factors could indicate risks that can lead to cervical cancer. 

We are trying to develop an algorithm that recommends whether a patient should get a biopsy or not. Given the features of our dataset, if our algorithm recommends that a patient get a biopsy, that indicates according to our data, he or she has a high probability of having cervical cancer. 

We plan on using a linear classifier to make predictions, where our solution set is {+1,-1}, where +1 indicates we recommend a biopsy while -1 means we do not recommend a biopsy. We will use either a perceptron or a Support Vector Machine to generate the linear classifier boundary. If that does not work, we will use Kernelization to project our data to a higher dimension using an RBF kernel, and with that we should be able to generate a separation boundary.
