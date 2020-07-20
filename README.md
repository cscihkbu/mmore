# Medical Concept Embedding with Multiple Ontological Representations (IJCAI-19)

This repo contains the PyTorch implementation of the paper `Medical Concept Embedding with Multiple Ontological Representations` in IJCAI-19. [[paper]](https://www.ijcai.org/Proceedings/2019/641) [[dataset]](https://mimic.physionet.org/)

## Requirements
The codes have been tested with the following packages:
- Python 3.5
- PyTorch 0.4.1

## Quick Demo
To run the model, clone the repo and decompress the demo data archive by executing the following commands:
```bash
git clone git@github.com:cscihkbu/mmore.git
cd mmore
python mmore_dxrx.py
```

Or you can train the model with diagnoses only by:
```bash
python mmore_dx.py
```

## Data Format and Organization
We follow the same input data organization used by GRAM. Please refer to the section `STEP 4: How to prepare your own dataset` of [the GRAM repository](https://github.com/mp2893/gram) for more details.


## Citation
If you find the paper or the implementation helpful, please cite the following paper:
```bib
@inproceedings{song2019learning,
  title={Medical concept embedding with multiple ontological representations},
  author={Song, Lihong and Cheong, Chin Wang and Yin, Kejing and Cheung, William K. and Fung, Benjamin C. M. and Poon, Jonathan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence ({IJCAI-19})},
  pages={4613--4619},
  year={2019},
  organization={AAAI Press}
}
```


