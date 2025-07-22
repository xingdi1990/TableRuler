# Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models (ECCV'24)

This is the implementation for paper: [Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models](https://www.arxiv.org/pdf/2407.10299).

## Description

![](pipe.svg)

The AnomalyRuler pipeline consists of two main stages: induction and deduction. The induction stage involves: i) visual perception transfers normal reference frames to text descriptions; ii) rule generation derives rules based on these descriptions to determine normality and anomaly; iii) rule aggregation employs a voting mechanism to mitigate errors in rules. The deduction stage involves: i) visual perception transfers continuous frames to descriptions; ii) perception smoothing adjusts these descriptions considering temporal consistency to ensure neighboring frames share similar characteristics; iii) robust reasoning rechecks the previous dummy answers and outputs reasoning.

## Dependencies

```
conda create --name tableruler python=3.10 -y
conda activate tableruler
pip install torch==2.1.0 torchvision==0.16.0 transformers==4.35.0 accelerate==0.24.1 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.22.post7 triton==2.1.0
```

```angular2html
pip install pandas pillow openai scikit-learn protobuf
```

## Dataset

### Command Line Download (Recommended)
All datasets can be downloaded via command line. Create a datasets directory and run these commands:

```bash
# Setup environment and install dependencies
conda activate tableruler
pip install gdown

# Create datasets directory
mkdir -p datasets && cd datasets

# Download all datasets (total ~28GB)
gdown 1PO5BCMHUnmyb4NRSBFu28squcDv5VWTR -O ped2_dataset.zip
gdown 1b1q0kuc88rD5qDf5FksMwcJ43js4opbe -O avenue_dataset.zip
gdown 1KbfdyasribAMbbKoBU1iywAhtoAt9QI0 -O ubnormal_dataset.zip

# SHTech (SharePoint download with browser user agent)
wget --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
"https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yyang179_jh_edu/EZQzXSY1XfZNm7gEh1zFS7IB4RA484KQD-BGEb-H_kAtVA?e=49KJ9h&download=1" \
-O shtech_dataset.zip

# Extract all datasets
unzip ped2_dataset.zip -d ped2/
unzip avenue_dataset.zip -d avenue/
unzip ubnormal_dataset.zip -d UBNormal/
unzip shtech_dataset.zip -d SHTech/

# Cleanup downloaded archives and Mac metadata
rm *.zip
rm -rf SHTech/__MACOSX
```

### Expected Directory Structure
After download and extraction, your datasets directory should look like:

```
datasets/
├── ped2/ped2/               # 105MB - UCSD Ped2 dataset
│   ├── training/
│   └── testing/
├── avenue/avenue/           # 2.2GB - CUHK Avenue dataset  
│   ├── training/
│   └── testing/
├── UBNormal/               # 15GB - UBNormal dataset (29 scenes)
│   ├── Scene1/
│   ├── Scene2/
│   └── ... (Scene29)
└── SHTech/SHTech/          # 10.75GB - ShanghaiTech dataset
    ├── train/
    └── test/
        ├── 01_0014/
        │   ├── 000.jpg
        │   ├── 001.jpg
        │   └── ...
        ├── 01_0015/
        └── ... (94 test sequences)
```

### Manual Download Links (Alternative)
If command line download fails, use these manual links:
* [SHTech](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yyang179_jh_edu/EZQzXSY1XfZNm7gEh1zFS7IB4RA484KQD-BGEb-H_kAtVA?e=49KJ9h)
* [ped2 and avenue](https://github.com/feiyuhuahuo/Anomaly_Prediction?tab=readme-ov-file)
* [UBNormal](https://github.com/lilygeorgescu/UBnormal)

## Run

### Step 1: Visual Perception
```
python image2text.py --data='SHTech'
```

### Step 2: Rule Generation + Rule Aggregation
```angular2html
python main.py --data='SHTech' --induct --b=1 --bs=10
```

### Step 3: Perception Smoothing
```angular2html
python majority_smooth.py --data='SHTech'
```
PS: You can also start from Step 3 to reuse the rules and simply reproduce the results.

### Step 4: Robust Reasoning
```angular2html
python main.py --data='SHTech' --deduct
```


## Citation

```angular2html
@inproceedings{yang2024anomalyruler,
    title={Follow the Rules: Reasoning for Video Anomaly Detection with Large Language Models},
    author={Yuchen Yang and Kwonjoon Lee and Behzad Dariush and Yinzhi Cao and Shao-Yuan Lo},
    year={2024},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)}
}
```
