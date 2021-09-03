Clone repo https://github.com/LuoweiZhou/VLP
```
cd VLP
conda env create -f misc/vlp.yml --prefix /home/<username>/miniconda3/envs/vlp
conda activate vlp
```

Clone repo https://github.com/NVIDIA/apex
```
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --cuda_ext --cpp_ext
```

```
pip install datasets==1.7.0
```