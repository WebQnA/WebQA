### Download Data
- [Main Data](http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_data_first_release.7z)

- Images

The large img file is compressed and split into 51 chunks of 1GB. You can download all chunks at once by running [this script](https://github.com/WebQnA/WebQA/blob/main/download_imgs.sh).

To unzip and merge all chunks, run ` 7z x imgs.7z.001 `

We also provide google drive download [links](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ?usp=sharing) 

You are good when you have `WebQA_train_val.json`, `WebQA_test.json`, `imgs.lineidx` and `imgs.tsv`.

### [Explore Data](https://github.com/WebQnA/WebQA/blob/main/demo/Take_a_look_WebQA.ipynb)

### Output Format (A json file with guids as keys)
```
{<guid>: {'sources': [<image_id>/<snippet_id>, ..., ],
          'answer': "xxxxxxx" },
 <guid>: {...},
 <guid>: {...},

}
```
