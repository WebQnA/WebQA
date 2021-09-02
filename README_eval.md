### Build Environment

<br>

```
conda create --name WebQA_eval python=3.8.3
conda activate WebQA_eval
```

https://spacy.io/usage
```
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```

```
pip install word2number
```

https://github.com/google-research/bleurt
```
pip install --upgrade pip
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

https://pytorch.org/get-started/previous-versions/
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

https://pypi.org/project/transformers/4.9.2/
```
pip install transformers==4.9.2
```

<br>

### Run

<br>

Download BART checkpoint on ParaBank: https://drive.google.com/file/d/1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m/view?usp=sharing

Clone BARTScore repo https://github.com/neulab/BARTScore

Eval on all question categories
```
CUDA_VISIBLE_DEVICES=1 python eval.py --file <path_to_output_file> 
```

Eval on chosen categories

Provide Qcate_breakdown argument as a json loadable string. Available categories for image-based queries: `color, shape, number, choose, YesNo, Others`.
```
CUDA_VISIBLE_DEVICES=1 python eval.py --file <path_to_output_file> --Qcate_breakdown '["color", "shape", "number"]' 
```




