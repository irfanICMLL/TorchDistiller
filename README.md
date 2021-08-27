
#TorchDistiller

This project is a collection of the open source pytorch code for knowledge distillation, especially for the perception tasks, including semantic segmentation, depth estimation, object detection and instance segmentation.

## Collection papers and codebase

### Semantic Segmentation

- structured knowledge distillation for semantic segmentation, CVPR2019 [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) [[code]](https://github.com/irfanICMLL/structure_knowledge_distillation/)
- Intra-class Feature Variation Distillation for Semantic Segmentation, ECCV2020 [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520341.pdf) [[code]](https://github.com/YukangWang/IFVD)
- Channel-wise Knowledge Distillation for Dense Prediction, ICCV2021 [[paper]](https://arxiv.org/abs/2011.13256) [[code]](./SemSeg-distill)
- Knowledge distillation based on MMsegmentation  [[code]](https://github.com/pppppM/mmsegmentation-distiller)
### Object Detection and Instance Segmentation
- Knowledge distillation based on MMdetection [[code]](https://github.com/pppppM/mmdetection)
- Knowledge distillation based on adet [[code]](./adet-distill)

## Update History

- 2021.08.20 Release the code for channel-wise distillation for semantic segmentation

We are integrating more of our work and other great studies into this project. 
## TO DO LIST
- Distillation on FCOS
- Distillation on CondInst


## Contribute

To contribute, PR is appreciated and suggestions are welcome to discuss with.

## License

For academic use, this project is licensed under the 2-clause BSD License. See LICENSE file. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com) and [Peng Chen](mailto:blueardour@gmail.com).

