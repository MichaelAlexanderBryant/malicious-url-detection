# Malicious URL Detection

This repository is for the analysis and modeling done with the malicious and benign websites dataset. Below you will find an overview of the data, code, and results.

Over 70% of all system intrusion breaches involve malware, and 32% of all malware is distributed via the web. The average cost of a data breach for an organization like IBM is 4.24 million dollars. Given the rise of remote work due to COVID-19, preventing exposure to malicious websites is imperative.

### Project Outcome

The models I chose were due to their potential to handle an imbalanced dataset. The dataset I used consisted of 12.1% malicious URLS and 87.9% benign. I chose models which have the ability to adjust prior probabilities, change class weights, or have a tunable cost parameter. I also tried downsampling and upsampling. The model was deployed as an API that can be used to check websites for safe access. 

### Code Used 

**Python Version:** <br />
**Packages:** <br />
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Malicious URL Dataset

### Variables

## Files

### code/eda_and_cleaning.py

### code/imputation.py

### code/modeling_without_sampling_methods.py

### code/modeling_with_downsampling.py

### code/modeling_with_upsampling.py

### ouput/eda_and_cleaning

### output/imputation

### output/modeling

### dataset.csv

## Results

<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_F1.jpg"><br/>
  <figcaption>Figure 1: F1 scores per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>

<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_Recall.jpg"><br/>
  <figcaption>Figure 2: Recall scores per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>

<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_Precision.jpg"><br/>
  <figcaption>Figure 3: Precision scores per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>


<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_Accuracy.jpg"><br/>
  <figcaption>Figure 4: Accuracy scores per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>

<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_ROC AUC.jpg"><br/>
  <figcaption>Figure 5: ROC AUC per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>

<div align="center">
<figure>
<img src="output/modeling/model_comparison/stripplot_model_Precision-Recall AUC.jpg"><br/>
  <figcaption>Figure 6: Precision-recall AUC per model using non-sampled, downsampled, and upsampled training data.</figcaption>
</figure>
<br/><br/>
</div>

## Resources

1. [Malicious and benign websites dataset](https://www.kaggle.com/datasets/xwolf12/malicious-and-benign-websites)
2. [A stacking model using URL and HTML features for phishing webpage detection](https://www.sciencedirect.com/science/article/abs/pii/S0167739X1830503X)
3. [Malicious URL Detection Based on Associative Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7911559/)
4. [Intelligent phishing url detection using association rule mining](https://hcis-journal.springeropen.com/articles/10.1186/s13673-016-0064-3#:~:text=(4)-,Association%20rule%20mining%20to%20detect%20phishing%20URL,when%20a%20user%20accesses%20it.)
5. Applied Predictive Modeling by Max Kuhn and Kjell Johnson
6. [Upsampling and Downsampling Imbalanced Data in Python](https://wellsr.com/python/upsampling-and-downsampling-imbalanced-data-in-python/)
7. [ROC Curves and Precision-Recall Curves for Imbalanced Classification](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/)
