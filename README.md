# Auto-Validate by-History (AVH)

Python implementation of AVH and baselines reported in paper [Auto-Validate by-History: Auto-Program Data Quality Constraints to Validate Recurring Data Pipelines]. Can follow the steps below to reproduce results.

## Dependencies
- Ubuntu 18.04, [Anaconda 3.5+](https://www.anaconda.com/)
- Tested on Python 3.8.0
- Download and install [arrayfire](https://repo.arrayfire.com/python/wheels/3.8.0/) e.g. pip install arrayfire-3.8.0-cp38-cp38-linux_x86_64.whl
- All other required python packages can be installed using our prepared requirements.txt (run  `pip install -r requirements.txt`)

## Reproduce paper results in notebooks
**Jupyter Notebook that shows and reproduces results reported in the paper**
- The notebook `visualization.ipynb` shows the main comparison results in our paper
- The notebook `sensitivity.ipynb` shows all sensitivity and ablation results in our paper



**Run AVH from the beginning to reproduce results in the paper**
-  Run `python avh_with_stationary.py`
-  Run `python avh_no_stationary.py`
-  AVH result will be stored in `./result` folder (consumed by the Jupyter notebook above)

## Code Overview: Walk through of each .py file
### Utility Tool
- `dists.py`: compute statistical distance between two samples, and the major entry is `comp_dist(sample_p, sample_q, dtype='numeric')`
- `preprocessing.py`: load and preprocesss raw data
- `utils.py`: provide some common utility tools usd by most models
- `gene_sample.py`: generate synthetical data based on our proposed synthesis rules The main entry is `GenSample.gen_save_sample(dir_name)`
- `stationary_checking`: process time-series, which is a part of running AVH-with stationary
- `global_efficiency.py`: measure the runtime of singal and single+two distribution AVH  
   ### Compared methods
- `avh_no_stationary.py`: Auto-Validate-by-History no stationary checking version
- `avh_with_stationary.py`: Auto-Validate-by-History with stationary checking version
- `azure_drift_detector.py`: Azure Drift Detector
- `azureAD.py`: [Azure Anomaly Detection][1]
- `deequ.py`: [Amazon Deequ Test][2]
- `hypothesis_test.py`: Hypothesis test
- `mad.py`: MAD Test
- `ml_baseline_single_var.py`: extract single-variabe features, and evaluate on supervised ML baselines and unsupervised anomaly detection baselines 
- `tfdv.py`: [Google TFDV Test][3]
- `DQ_test.py`: Avg-KNN, using orig features in the paper [Automating Data Quality Validation for Dynamic Data Ingestion][4]
- `DQ_test_all.py`:Avg-KNN, testing on the AVH features [Automating Data Quality Validation for Dynamic Data Ingestion][4]
- `fast_rule.py`: [Fast rule mining in ontological knowledge bases with AMIE+][5]
- `robust_discovery.py`: [Robust discovery of positive and negative rules in knowledge bases][6]


## Benchmark data
- Unzip data.zip to ./data foler, which has our experiment data
 * **./data**
   * **categorical data**: categorical data used in training and testing 
   * **category_pool**: a large pool of real datasets from which we search for categorical data that are similar to benchmark test data but still statstically different
   * **numerical data**: numerical data used in training and testing 
   * **numeric_pool**: a large pool of real datasets from which we search for numerical data that are similar to benchmark test data but still statstically different
   * **synthesis_0907**
      * **category**: sysnthetical categorical data for training
      * **category_test**: sysnthetical categorical data for testing
      * **numeric**: sysnthetical numerical data for training
      * **numeric_test**:sysnthetical numerical data for testing


  [1]: https://dl.acm.org/doi/pdf/10.1145/3292500.3330680?casa_token=DnN65_1ImG8AAAAA:Ca4a-BMUfcaPHm8nL5x25qhvZgnyTRc5IA0VDMzetcRQX8aGkSzAxbr3W-pNb3faoCQVLHxynuvy
  [2]: https://dl.acm.org/doi/pdf/10.1145/3299869.3320210?casa_token=NWEb7cqiqqYAAAAA:V_bWDiX7gVxLTldVLMFyr-wYrKTk5lPjEz92wYZpfP2XOpeZJw4OLqTRv-DSMH8q9eZ_WWNji-NX
  [3]: https://dl.acm.org/doi/pdf/10.1145/3318464.3384707
  [4]: https://sergred.github.io/files/edbt.reds.pdf
  [5]: https://link.springer.com/article/10.1007/s00778-015-0394-1?email.event.1.SEM.ArticleAuthorContributingOnlineFirst
  [6]: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8509329&casa_token=qqnE6WuNnzIAAAAA:inNe_bNr_abB_yzzdLsxfcgEUCedZqhFtuhStLPt3BRnrzVIBF8J0Tf45f130ci6JQLMtrMpuw&tag=1
