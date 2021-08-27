# AdelaiDet-distill
This repo is a distillation tool for AdelaiDet.

AdelaiDet is an open source toolbox for multiple instance-level recognition tasks on top of [Detectron2](https://github.com/facebookresearch/detectron2).
All instance-level recognition works from our group are open-sourced here.

To date, AdelaiDet implements the following algorithms:

* [FCOS](configs/FCOS-Detection/README.md)
* [BlendMask](configs/BlendMask/README.md)
* [MEInst](configs/MEInst-InstanceSegmentation/README.md)
* [ABCNet](configs/BAText/README.md)
* [CondInst](configs/CondInst/README.md)
* [SOLO](https://arxiv.org/abs/1912.04488) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [SOLOv2](https://arxiv.org/abs/2003.10152) _to be released_ ([mmdet version](https://github.com/WXinlong/SOLO))
* [DirectPose](https://arxiv.org/abs/1911.07451) _to be released_

Quantized AdelaiDet can be found [here](https://github.com/aim-uofa/AdelaiDet-Quantization).


## Installation

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) for now. The incompatibility with the latest one will be fixed soon.

Then build AdelaiDet with:

```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```

Some projects may require special setup, please follow their own `README.md` in [configs](configs).

## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

```BibTeX
@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}

@inproceedings{chen2020blendmask,
  title     =  {{BlendMask}: Top-Down Meets Bottom-Up for Instance Segmentation},
  author    =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{zhang2020MEInst,
  title     =  {Mask Encoding for Single Shot Instance Segmentation},
  author    =  {Zhang, Rufeng and Tian, Zhi and Shen, Chunhua and You, Mingyu and Yan, Youliang},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{liu2020abcnet,
  title     =  {{ABCNet}: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network},
  author    =  {Liu, Yuliang and Chen, Hao and Shen, Chunhua and He, Tong and Jin, Lianwen and Wang, Liangwei},
  booktitle =  {Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year      =  {2020}
}

@inproceedings{wang2020solo,
  title     =  {{SOLO}: Segmenting Objects by Locations},
  author    =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}

@article{wang2020solov2,
  title   =  {{SOLOv2}: Dynamic, Faster and Stronger},
  author  =  {Wang, Xinlong and Zhang, Rufeng and Kong, Tao and Li, Lei and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:2003.10152},
  year    =  {2020}
}

@article{tian2019directpose,
  title   =  {{DirectPose}: Direct End-to-End Multi-Person Pose Estimation},
  author  =  {Tian, Zhi and Chen, Hao and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1911.07451},
  year    =  {2019}
}

@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](https://cs.adelaide.edu.au/~chhshen/).
