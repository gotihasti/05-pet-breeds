# Pet Breed Image Classification (fastai)

A deep learning notebook that classifies **37 breeds of cats and dogs** from photos, using transfer learning with fastai and PyTorch. It works through the image classification chapter of [*Deep Learning for Coders with fastai & PyTorch*](https://github.com/fastai/fastbook) and goes beyond a basic training run: it covers data augmentation, loss functions, learning rate selection, and fine-tuning strategy.

## Results

| Experiment | Model | Validation error rate |
|------------|-------|----------------------|
| Baseline, `fine_tune(2)` | ResNet34 | 7.4% |
| Tuned base learning rate (`3e-3`), `fine_tune(2)` | ResNet34 | 7.6% |
| `fit_one_cycle(3)`, then unfreeze and train 12 epochs with discriminative learning rates (`slice(1e-6, 1e-4)`) | ResNet34 | 7.0% |
| 6 more epochs at `1e-5` | ResNet34 | 6.5% |
| Mixed precision (fp16), `fine_tune(6, freeze_epochs=3)` | ResNet50 | **5.8%** |

The best model reaches about **94% accuracy** on a held-out validation set of 1,478 images. These numbers are from a single run with a fixed seed, so expect small variation if you rerun it.

## What the notebook covers

- **Data pipeline**: builds a fastai `DataBlock`, extracts each label from the filename with a regex, and applies a seeded random 80/20 train/validation split
- **Presizing**: resizes images to 460px on the CPU, then applies augmentations on the GPU and crops to 224px
- **Data augmentation**: rotation, zoom, and warp, shown visually
- **Loss functions from first principles**: sigmoid, softmax, negative log likelihood, and cross-entropy. The manual calculations are checked against PyTorch's `nn.CrossEntropyLoss`.
- **Model interpretation**: confusion matrix and most-confused pairs. The model mostly mixes up visually similar breeds, such as Egyptian Mau and Bengal, Siamese and Birman, and American Pit Bull Terrier and Staffordshire Bull Terrier.
- **Learning rate experiments**: a deliberately bad learning rate (`0.1`) to show instability, then the learning rate finder to pick a good one
- **Fine-tuning strategies**: freezing and unfreezing layers, and discriminative learning rates
- **Bigger model and mixed precision**: ResNet50 trained with fp16

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/): about 7,400 images across 37 breeds (12 cat and 25 dog). It is downloaded automatically by `untar_data(URLs.PETS)`, so you don't need to add data to this repo.

## Tech stack

Python, fastai, PyTorch, torchvision (pretrained ResNet34 and ResNet50), pandas, NumPy, and Jupyter.

## Getting started

A GPU is strongly recommended. The notebook was run on a Kaggle GPU, with each epoch taking under a minute.

### Run on Kaggle or Colab

Upload `05-pet-breeds.ipynb`, enable a GPU, and run all cells. The first code cell is Kaggle's default boilerplate and can be skipped or deleted.

### Run locally

```bash
git clone https://github.com/<your-username>/05-pet-breeds.git
cd 05-pet-breeds

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -U fastai fastbook jupyter
jupyter notebook 05-pet-breeds.ipynb
```

On a local machine without a GPU, training will be much slower. Reduce the epoch counts to try it out.

## Project structure

```
05-pet-breeds/
├── 05-pet-breeds.ipynb    # the full notebook, with outputs
└── README.md
```

## Possible extensions

- [ ] Export the trained model with `learn.export()` and serve predictions through a small web app
- [ ] Try a different architecture (for example ConvNeXt or EfficientNet) and compare error rates
- [ ] Add test-time augmentation
- [ ] Explore the confused breed pairs with `plot_top_losses`


Add a license of your choice (for example MIT) before publishing. The original fastbook material is licensed separately, so check its terms if you reuse text from it.
