Please find baseline predictions on the Val set in this directory.

Performance Scores:

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
