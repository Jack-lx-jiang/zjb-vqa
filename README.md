# ZhejiangLab Cup Global AI Competition 2018 - Video Recognition and Q&A
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/Jack-lx-jiang/zjb-vqa.svg?token=DgfD5ypVBPJzsnZ7L2Rm&branch=master)](https://travis-ci.com/Jack-lx-jiang/zjb-vqa)
## Introduction
This is an implement of our model for ZhejiangLab Cup Global AI Competition 2018 - Video Recognition and Q&A.
 
Contest website: [https://tianchi.aliyun.com/competition/introduction.htm?raceId=231676](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231676)

## Requirement
- One gtx 1080ti is enough to train the model
- CUDA 8.0
- CUDNN6

## Installation
```bash
pip install -r requirements.txt  # install requirements
imageio_download_bin ffmpeg      # install ffmpeg dependency
```

## Train and Test
```bash
./main.py
```
This will save submit.txt in `../submit` folder

You may also need to change `data_dir` in `main.py`

## Difference performance from different datasets
If only use DatasetA, you will get a result around 0.34.<br/>
If you want to achieve the accuracy on ranking board(0.45), you need to merge DatasetA with DatasetB and set the 
corresponding path in main.py.
