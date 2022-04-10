Please find baseline predictions on the Val set in this directory.

#### Performance Scores (Predicted sources):

**VLP+x101fpn**

| Query Type  | Retrieval F1 |   FL   |   Acc  |  FL\*Acc |
|-------------|:------------:|:------:|:------:|:--------:|
| Img Queries |    0.6799    | 0.4654 | 0.4511 | 0.2422 |
| Txt Queries |    0.7007    | 0.2614 |  ----  |  ----  |

#### Performance Scores (Oracle sources):

**VLP+VinVL**

| Query Type  |   FL   |   Acc  |  FL\*Acc |
|-------------|:------:|:------:|:--------:|
| Img Queries | 0.4759 | 0.4961 | 0.2753 |
| Txt Queries | 0.3072 |  ----  |  ----  |

**VLP+x101fpn**

| Query Type  |   FL   |   Acc  |  FL\*Acc |
|-------------|:------:|:------:|:--------:|
| Img Queries | 0.4692 | 0.4429 | 0.2377 |
| Txt Queries | 0.3039 |  ----  |  ----  |

Acc scores on txt queries are not available because we didn't annotate Keyword answers on Val samples. 
To approximate keyword annotations on Val, we would suggest you try named entity detection and noun-chunk extraction tools with Spacy.
