<details open="open">
<summary>Table of Contents</summary>

- [About](##About)
- [Prerequisites](##Prerequisites)
- [Getting Started](##Getting Started)
- [Approach](##Approach)
- [Support](##Support)

</details>

---

## About

<table>
<tr>
<td>
The purpose of this repository is to enhance explainability of sequential recommender models with SHAP values. We currently use 2 packages for this<br>
    - [Rechorus](https://github.com/THUwangcy/ReChorus): A general PyTorch framework for Top-K recommendation with implicit feedback<br>
    - [TimeSHAP](https://github.com/feedzai/timeshap): A model-agnostic, recurrent explainer that builds upon KernelSHAP and extends it to the sequential domain. TimeSHAP computes event/timestamp- feature-, and cell-level attributions.<br>

In this repository, we have chosen GRU4Rec as the sequential recommender system and provided a local level explanations of a few
    interactions on MovieLens 1m dataset.




## Prerequisites

```sh
pip install -r requirements.txt
```
    
## Getting Started
1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository
```bash
git clone https://github.com/maneelusf/extpersonalization
```
3. Install requirements and step into the `src` folder

```bash
cd src
```
4. Run model with the build-in dataset
```bash
python argcorpus.py --model_name GRU4Rec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
```
```bash
python main.py --model_name GRU4Rec --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
```

5. (optional) Run jupyter notebook in `data` folder to download and build new datasets, or prepare your own datasets according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/data/README.md) in `data`

6. (optional) Implement your own models according to [Guideline](https://github.com/THUwangcy/ReChorus/tree/master/src/README.md) in `src`

7. Then move into the notebooks folder
```bash
cd ../Notebooks
```
## Approach
1. The first step is to generate a list of the top K recommended items, and to calculate their scores using a matrix multiplication of the output vector from the model's forward loop with each item. These scores are expected to be highly positive numbers.<br>
2. When perturbing the sequence with a baseline item, the output vector from the forward loop is calculated using the original top K recommended items, rather than calculating it with each individual item in the sequence.<br>
3. This approach is motivated by the hypothesis that the initial list of top K items should not result in a higher positive score when perturbing the sequence. The objective is to evaluate the effect of the perturbed items on the recommendation score relative to the original recommended items.<br>
4. For example, it is expected that the SHAPLEY values for events -1 to -5 will be the highest. Therefore, perturbing these events should result in a vector that is less similar to the recommended items. Conversely, the last items in a sequence (e.g. events -20 and onwards) should output a vector that is similar to the recommended items, as these events have less impact on the subsequent items in the sequence.<br>

### Calculating SHAP values
After training a model, we can run the following the [notebook] (https://github.com/maneelusf/extpersonalization/blob/main/notebooks/Notebook%20to%20generate%20top%20K%20recommendations.ipynb)to generate SHAP values. 

## Support

Reach out to the maintainer at one of the following places:
- [GitHub discussions](https://github.com/maneelusf)


