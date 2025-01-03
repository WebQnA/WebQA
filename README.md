### News

**Oct 15 Update**: We decided to release the output files of our baseline models in case they will be helpful for future investigations. Feel free to [check it out](https://github.com/WebQnA/WebQA/tree/main/baseline_output_files)!

**Oct 9 Update**: Please note that we've updated the image reading method from `cv2` to `PIL` in the demo notebook. `ImageFile.LOAD_TRUNCATED_IMAGES = True
` is the key to avoid "Image NoneType error".

<br>

### Download Data
- [Main Data](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ)

The main data is split into two files. One for train+val (36,766+4,966 samples) and the other for test (7,540 samples).

- Images

The large img file is compressed and split into 51 chunks of 1GB. Download all chunks before moving to next step.

To unzip and merge all chunks, run ` 7z x imgs.7z.001 `

We also provide google drive download [links](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ?usp=sharing) 

You are good when you have `WebQA_train_val.json`, `WebQA_test.json`, `imgs.lineidx` and `imgs.tsv`.

<br>

### [Explore Data](https://github.com/WebQnA/WebQA/blob/main/demo/Take_a_look_WebQA.ipynb)

### Output Format (A json file with guids as keys)
```
{<guid>: {'sources': [<image_id>/<snippet_id>, ..., ],
          'answer': "xxxxxxx" },
 <guid>: {...},
 <guid>: {...},

}
```
