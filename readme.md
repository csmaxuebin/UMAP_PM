
# This code is the source code implementation for the paper "UMAP_PM: Local Differential Private High-Dimensional Data Release via UMAP"

  


## Abstract
![输入图片说明](/imgs/2024-06-17/uWE3xQUIcEnC5mJD.png)
Protecting the privacy of high-dimensional datasets while releasing representative data has been a focus of much attention in recent years. Local differential privacy (LDP) has emerged as a promising privacy standard for mining and releasing data. However, most existing research focuses on applying

LDP to complex data and/or analysis tasks, leaving the fundamental problem of releasing high-dimensional data under LDP insufficiently addressed. Applying LDP to high-dimensional data poses significant challenges due to the increased risk of perturbation error and computational complexity. Motivated by this, we propose a novel LDP mechanism named UMAP_PM, which leverages manifold and projection techniques to refine complex relationships between attributes and provide appropriate LDP for each result vector. By balancing data utility and privacy, and reducing computational time, UMAP_PM maximizes the utility of data while ensuring privacy. We experimentally evaluate UMAP_PM on real data and demonstrate its superiority over existing solutions in terms of accuracy and time complexity.

  
  

# Experimental Environment

  

```

- scikit-learn==0.24.2

- pandas==1.3.5

- umap-learn==0.5.2

- scipy==1.7.3

- numpy==1.23.2

- matplotlib==3.5.2

```

  

## Datasets

  

`SS13ACS,Adult,Criditcard`

  
  

## Experimental Setup

**Hyperparameters:**

![输入图片说明](/imgs/2024-06-17/EsoTAusWLqGqtr2h.png)



**Privacy-Preserving Methods:** UMAP_PM (Proposed Method)

- Leverages LDP and manifold learning techniques to achieve dimensionality reduction while maintaining complex data dependencies.
- Aims to balance data utility and privacy by applying LDP in an optimal way to the resulting low-dimensional vectors.
- Experimental results show UMAP_PM achieves significantly better accuracy and lower running time compared to other state-of-the-art solutions for generating synthetic data.
  

**Evaluation Metrics:**

  
Model Performance:

 1. Statistical Query Accuracy: The average variation distance (AVD) is used to measure the average change distance between the synthesized and the original dataset. A query set Qu containing 10,000 random linear queries is generated for each dataset.
2. Classification Performance: SVM classification and Logistic regression classification are used to test the effectiveness of the proposed UMAP_PM algorithm.
3. Running Time: The running time of data synthesis is compared between UMAP_PM and existing algorithms to verify whether UMAP_PM can reduce the running time.

Privacy Protection:

1. Attribute Dimensionality: The effect of attribute dimensions on the AVD is tested using different dimensions on the SS13ACS and Creditcard datasets.
2. Number of Users: The effect of the number of users on the AVD is tested using different numbers of users on the Adult and Creditcard datasets.
## Python Files

- **ssc_sample_UMAP.py**:
Serves as a versatile utility for data dimensionality reduction and classification. It employs the UMAP algorithm to project high-dimensional data into a 3D space, and then utilizes the Projected Mechanism (PM) to generate noisy samples that enhance privacy protection. 

- **ssc_UMAP_PM_gailv**:
This script serves as a framework for dimensionality reduction, privacy-preserving data transformation, and semi-supervised classification using the UMAP and Projected Mechanism algorithms, with a focus on evaluating the performance of the classification model.

- **ssc_GUASS_RP_gailv.py**:

  This script serves as a framework for dimensionality reduction, privacy-preserving data transformation, and semi-supervised classification using the Gaussian Random Projection and Gaussian mechanism, with a focus on evaluating the performance of the classification model

- **SVM_Zhexian**:

The purpose of this script is to provide a visual comparison of the classification performance of an SVM model under different privacy-preserving techniques and different values of the privacy budget (epsilon). This type of visualization can be useful for evaluating the trade-off between privacy and model accuracy.
  
  

## Experimental Results

These graphs and tables summarize the effectiveness of different federated learning strategies under various metrics, focusing on how different privacy-preserving methods.



  ![输入图片说明](/imgs/2024-06-17/2yUTGdiWC79hkk1a.png)![输入图片说明](/imgs/2024-06-17/6G7Ab1tyZwi0Blx4.png)[图片上传中...(image-G1izCaoyQPxiRGMZ)]![输入图片说明](/imgs/2024-06-17/9YLzYM0rrVxYQzGr.png)![输入图片说明](/imgs/2024-06-17/HdCuEQBEsAW54tdX.png)![输入图片说明](/imgs/2024-06-17/SAOFZ3uJ57PYljIC.png)![输入图片说明](/imgs/2024-06-17/JZEEIcq9NYUls9KH.png)![输入图片说明](/imgs/2024-06-17/rQZK9k3yu8GuxhuF.png)
  
  



## Update log

- {24.06.13} Uploaded overall framework code and readme file



