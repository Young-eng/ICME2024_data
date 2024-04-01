### Track #6, Animal Action Recognition

### Dataset

1. the dataset comes from Animal Kingdom(https://sutdcv.github.io/Animal-Kingdom/Animal_Kingdom/action_recognition/README_action_recognition.html)

2. While action classification labels in this dataset are multi-label, here I only take one label for each clip to make it adapted for the action classification algorithm, like Kinetics-400 dataset format.I will provide the train.csv and val.csv which I used for training and validation. I only used **10,000 video clips**, which is approximately**one-third of clips in dataset, 8000 clips from training set and 2000 clips from test set set respectively**. I did not used any images for training and validation. I will provide the python script which is used for generating these two csv files.

### Algorithm

1. Here I take **VideoMAE, proposed by team from Nanjing University, Tencent AI Lab and Shanghai AI Lab**, which takes Kinetcis-400 as raining dataset. here are their codes(https://github.com/MCG-NJU/VideoMAE), you can find their paper(Tong, Zhan et al. “VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.” ArXiv abs/2203.12602 (2022): n. pag.).You can follow this official guide in code repository to install packages and run codes. 

2. I changed some parameters during training. like learning rate is 5e-4  and batch size is 4 respectively for 100 epochs. The GPU I used is a RTX 3090. The pre-trained model is vit-base-224, you can find pretrained model and script in their code [Repository](https://github.com/MCG-NJU/VideoMAE/blob/main/scripts/kinetics/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_800/finetune.sh)

3. When I knew this MMVRAC Challenge, there are only five days left. So I did not have enough time to make some imporvements on the algorithm model, what I did is just processing the data and fine-tuning the model.

4. Here metrics are top1 and top5 accuracy. While I used one label for each clip, original clips are multi-label, so top5 accuracy may be more credible.
The top1 Accuracy is more than 45%, and top5 max accuracy is more than 80%

5. There are many things to do for future improvements. For example, modify the model structure to adapt for Animal Kingdom dataset or other downstream works if needed. And the metrics in this algorithm is little different with metrics in Animal Kingdom dataset. 

6. I will provide all scripts, files and results. My work for this challenge is not perfect. I think it is a good start for the future, because I did not use all images and video data in this dataset.

7. Fine-tuned model link is here, [checkpoint_best](https://drive.google.com/file/d/1P5WYJbCvgFrB4_B6JuwU5LcA5-ciBAxF/view?usp=sharing)

8. After finishing final epoch of fine-tuning, it seems there is a bug when mering final results while there is no any other .txt files in output directory.


### Citation of authors' Paper:

@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}