# ZhejiangLab Cup Global AI Competition 2018 - Video Recognition and Q&A
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/Jack-lx-jiang/zjb-vqa.svg?token=DgfD5ypVBPJzsnZ7L2Rm&branch=master)](https://travis-ci.com/Jack-lx-jiang/zjb-vqa)
## Introduction
This is an implement of our model for ZhejiangLab Cup Global AI Competition 2018 - Video Recognition and Q&A.
 
Contest website: [https://tianchi.aliyun.com/competition/introduction.htm?raceId=231676](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231676)
## requirement
Ubuntu 14<br/>
Python 3.5<br/>
One gtx 1080ti is enough to train the model<br/>
CUDA 8.0<br/>CUDNN6<br/>
Other dependencies see the requirements.txt<br/>
## Difference performance from different datasets
If only use DatasetA, you will get a result around 0.34.<br/>
If you want to achieve the accuracy on ranking board(0.45), you need to merge DatasetA with DatasetB and set the 
corresponding path in main.py.