# Literature Review: Fake Job Posting Detection

> DRAFT. Items marked [V] were confirmed against a search result (abstract or metadata level only, full texts not read). Items marked [M] are from memory and not looked up. Check all of them against the original paper before submission.

## 1. Problem and motivation

Online recruitment fraud (ORF) is the posting of fake job advertisements, most commonly as an employment scam. It endangers job seekers' privacy, causes financial losses and damages the credibility of the platforms that host the ads. Vidros et al. (2017) [V] were among the first to define ORF as a research problem, to describe its characteristics, and to release a public dataset, the Employment Scam Aegean Dataset (EMSCAD). Most later work, including ours, uses EMSCAD, which is also hosted on Kaggle. Secondary sources report 17,880 postings of which 866 (4.84%) are fraudulent [V, secondary, not checked against the primary paper]. Severe class imbalance is therefore a defining property of the task.

## 2. Classical approaches: text features plus linear and tree models

The original study applied bag-of-words modelling with standard classifiers [V]. Vidros et al. used a smaller balanced subset, so their results do not say how the models behave on the original imbalanced data [V, per Taneja et al.]. Later EMSCAD studies stay in this family:
- TF-IDF text features combined with hand-built meta-features (such as missing company profile or logo) and a logistic regression classifier (University of the UAE study) [V, abstract only].
- Ensemble methods with response coding or bag-of-words featurization (NITK study) [V, abstract only].
- A random forest ensemble approach reported in the 2022 literature [V, snippet only].

Taneja et al. (2025) tabulate ten earlier EMSCAD studies from 2017 to 2024 [V]. The reported best models are mostly RF, JRip, Extra Trees (ETC) or SVM, with accuracies of 91% to 99.9% and F1 between 0.73 and 0.99 where reported. Two of those entries carry caveats worth repeating: Amaar et al. (2022) reached 99.9% accuracy but measured it on a resampled test set, which does not reflect real performance, and Afzal et al. (2024) used feature selection plus SMOTE and reported low recall and AUC. The paper also states that EMSCAD is manually annotated, contains some duplicate and blank entries, and has a small number of misclassified entries.

A recurring critique is that the original work used a balanced subset, so results on the realistic imbalanced distribution were left open [V, snippet from a secondary paper]. A 2022 paper also argues that treating ORF as plain binary classification ignores deeper structure in the problem [V, snippet].

Take-away for our project: TF-IDF with logistic regression or a linear SVM is the standard baseline, and it is the natural counterpart of the handcrafted-feature branch in our earlier image project.

## 3. Deep and pretrained-language-model approaches

- Akram et al. (2024), IEEE Access 12, 109388-109408 [V]: fine-tune BERT and RoBERTa, ten SMOTE variants for imbalance. BERT with the SMOBD variant gave the best balanced accuracy and recall, about 90%.
- Taneja, Vashishtha and Ratnoo (2025), Fraud-BERT, Discover Computing 28:9 [V, full text read]: fine-tune all BERT-base parameters (learning rate 1e-5, max length 510 tokens, batch size 6, 2 epochs) on one text field built by concatenating 12 columns. They drop `telecommuting`, `has_company_logo`, `has_questions`, `job_id` and `salary_range`. With a single 80:20 split and no resampling they report P 0.94, R 0.92, F1 0.93, accuracy 0.99 and AUC 0.99. Their own baselines on the same split were SVM (P 0.98, R 0.84, F1 0.89), RF (R 0.82, F1 0.88), LR (R 0.70, F1 0.78), Bi-LSTM (F1 0.89) and NB (F1 0.49). SVM took over an hour to train against about 20 minutes for BERT on an A100.
  - Limitations as I read the paper (these are my observations, not the authors' claims): one 80:20 split, no seeds or cross-validation reported, no stratification or validation split described, and no deduplication step described even though the authors say the dataset contains duplicate entries. The authors list the 512-token limit, compute cost and inherited bias as limitations.

Frozen pretrained-LM representations are a cheaper alternative to fine-tuning. Sentence-BERT (Reimers and Gurevych, EMNLP-IJCNLP 2019, pp. 3982-3992) [V] adapts BERT with siamese networks so that embeddings can be compared directly, and it avoids running a cross-encoder over every pair. The final project instead uses matched text-only and late-fusion two-logit classifiers for top-two-layer and full DistilBERT fine-tuning, so adaptation budget and structured fusion can be evaluated directly.

The foundational transformer paper is Devlin et al. (2019), BERT, NAACL [M].

## 4. Class imbalance and evaluation

- SMOTE (Chawla et al., JAIR 16, 321-357, 2002) [V] creates synthetic minority examples instead of duplicating existing ones.
- Saito and Rehmsmeier (PLoS ONE, 2015) [V] argue that ROC plots can give a misleadingly optimistic picture on imbalanced data and that precision-recall plots are more informative.

Amaar et al. (2022) [V, per Taneja et al.] is a concrete example: they evaluated on a resampled test set, so their 99.9% accuracy cannot be compared with results on the natural 4.84% fraud rate. Reported numbers across the ORF papers are not directly comparable. Splits, resampling choices and metrics differ (accuracy, F1, recall and balanced accuracy all appear), and some resample before splitting, which would leak. This is the main justification for our design: grouped train/validation/test splits across five saved seeds, preprocessing fitted on training folds only, no resampling, a majority-class baseline, and precision, recall, F1 and AP reported together.

## 5. Explainability

- LIME (Ribeiro, Singh and Guestrin, KDD 2016, pp. 1135-1144) [V] explains one prediction by fitting a simple model locally around it.
- SHAP (Lundberg and Lee, NIPS 2017) [V] assigns each feature an importance value per prediction and unifies several earlier attribution methods.

For a text classifier these show which words drive a "fraudulent" prediction. For a linear model on TF-IDF, the coefficients give a global view that can be compared against the local ones.

## 6. Gap and positioning

From the sources above (mostly abstract-level, except Fraud-BERT, which was read in full):
1. Reported results are hard to compare because evaluation protocols differ.
2. Fine-tuned transformers are reported to do well, but the cost and the gain over a tuned TF-IDF baseline are rarely isolated, and single-split results carry no variance estimate.
3. In the one paper I read in full (Fraud-BERT), the strong baselines (SVM, RF, LR) are already close to the fine-tuned model in accuracy and the gap is mainly in recall, so a tuned classical baseline deserves a fair test.
4. Few papers combine a baseline floor, adaptation and fusion ablations, tuning and explanation under one protocol.

Our contribution is a focused comparison on one protocol: a majority floor, text-only and structured TF-IDF logistic regression, a decision tree and a one-class SVM, plus matched text-only/late-fusion heads for top-two-layer versus full DistilBERT fine-tuning. The main contrasts are structured-minus-text logistic regression, full-minus-top-two adaptation separately for text and fusion, and fused-minus-text within each adaptation budget. Hyperparameters, epochs and thresholds are selected without test labels within each run; evaluation uses imbalance-appropriate metrics and paired differences across five saved splits. Classical models are inspected with coefficients, tree importances and error examples. Results remain exploratory because EDA informed the design, seed-0 test scores were inspected, partitions overlap, and each neural partition has only one optimization seed. I have not run a search for claims (2) and (3) beyond the papers above, so state them cautiously in the report.

## References (to verify)

1. Vidros, S., Kolias, C., Kambourakis, G., Akoglu, L. (2017). Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset. Future Internet 9(1):6. https://mdpi.com/1999-5903/9/1/6
2. Akram, N., Irfan, R., Al-Shamayleh, A.S., Kousar, A., Qaddos, A., Imran, M., Akhunzada, A. (2024). Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches. IEEE Access 12:109388-109408. doi:10.1109/ACCESS.2024.3435670
3. Taneja, K., Vashishtha, J., Ratnoo, S. (2025). Fraud-BERT: transformer based context aware online recruitment fraud detection. Discover Computing 28:9. doi:10.1007/s10791-025-09502-8 [read in full]
3b. Amaar, A., Aljedaani, W., Rustam, F., Ullah, S., Rupapara, V., Ludi, S. (2022). Detection of Fake Job Postings by Utilizing Machine Learning and Natural Language Processing Approaches. Neural Processing Letters. doi:10.1007/s11063-021-10727-z [listing only, not read]
4. Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 3982-3992. arXiv:1908.10084
5. Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16:321-357. doi:10.1613/jair.953 [author list from memory]
6. Saito, T., Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE 10(3):e0118432. [title from memory]
7. Ribeiro, M.T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD, 1135-1144. doi:10.1145/2939672.2939778
8. Lundberg, S.M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NIPS 2017. arXiv:1705.07874
9. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL. [M]
