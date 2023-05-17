<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Prerequisites](#getting-started)
- [Getting Started](##Getting Started)
- [Support](#support)

</details>

---

# About

<table>
<tr>
<td>
The purpose of this repository is to enhance explainability of sequential recommender models with SHAP values. We currently use 2 packages for this
    - [Rechorus](https://github.com/THUwangcy/ReChorus): A general PyTorch framework for Top-K recommendation with implicit feedback
    - [TimeSHAP](https://github.com/feedzai/timeshap): A model-agnostic, recurrent explainer that builds upon KernelSHAP and extends it to the sequential domain. TimeSHAP computes event/timestamp- feature-, and cell-level attributions. 

In this repository, we have chosen GRU4Rec as the sequential recommender system and provided a local level explainations of 




## **Prerequisites**

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
    
### Calculating SHAP values
After training a SHAP model, we can run the following the [notebook](https://github.com/maneelusf/extpersonalization/blob/main/notebooks/Notebook%20to%20generate%20top%20K%20recommendations.ipynb).

## Support

Reach out to the maintainer at one of the following places:
- [GitHub discussions](https://github.com/maneelusf)


