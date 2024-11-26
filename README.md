# MV-Match: Multi-View Matching for Domain-Adaptive Identification of Plant Nutrient Deficiencies


Official pytorch implementation of [MV-Match](https://arxiv.org/abs/2409.00903) (BMVC24). 

## 📜 News (2024-11-26)
- TODO: ckpt, dataset
- The source code is released!

## 💡 Introduction
Taks: Unsupervised domain adaptation for classification of plant nutrient deficiencies (PND).

Insights: (1) Acquiring labeled data for more plants is extremely expensive, while collecting multi-view data given limited plants is straightforward; (2) Existing PND approaches do not generalize to unseen genotypes / cultivars.

MV-Match: We propose a framework that leverages multiple camera views in the source and target domain for unsupervised domain adaptation for PND.

Datasets: We show that our MV-Match achieves state-of-the-art results on two proposed nutrient deficiency datasets: MiPlo-B and MiPlo-WW.

![Framework](images/MV-Match.png)
## Install
```bash
conda install pip==22.3.1 python==3.8.16
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install opencv-python portalocker tqdm webcolors scikit-learn matplotlib numba timm numpy==1.23.5 tensorboardX prettytable timm wandb scikit-image line_profiler seaborn pandas
pip install -i https://test.pypi.org/simple/ tllib==0.4
```

## Data
TODO

## Train and Evaluation
```bash
bash scripts/B2H.sh
```

## ❤️ Acknowledgments

The codes are adapted from [Transfer-Learning-library](https://github.com/thuml/Transfer-Learning-Library/tree/master). Thanks for their wonderful works and code!

## ✒️ Citation

If MV-Match is helpful for your research, please consider star ⭐ and citation 📝 :
```
@article{yi2024MV-Match,
  title={MV-Match: Multi-View Matching for Domain-Adaptive Identification of Plant Nutrient Deficiencies},
  author={Yi, Jinhui and Luo, Yanan and Deichmann, Marion and Schaaf, Gabriel and Gall, Juergen},
  journal={arXiv preprint arXiv:TODO},
  year={2024}
}
```
## 📄 License

- The content of this project is released under the MIT license license as found in the [LICENSE](https://github.com/jh-yi/MV-Match/blob/main/LICENSE) file.
- This dataset follows Creative Commons Attribution Non Commercial Share Alike 4.0 Internation License.
