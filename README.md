# Fine-Tuning Stable Diffusion on Digimon Dataset

This repository contains the fine-tuning of Stable Diffusion on a dataset of Digimon images, enabling the model to generate Digimon-style images with improved coherence and specificity.

## Dataset

The dataset was crawled from [Wikimon's Visual List of Digimon](https://wikimon.net/Visual_List_of_Digimon) and is available for download:
[Google Drive Link](https://drive.google.com/drive/folders/1tmcdsoX67NvmAgtmGJgo6kb3N6SlJeLu)

## Base Model

Was fine-tuned from Stable Diffusion v1.4:
[Stable Diffusion v1.4 (Hugging Face)](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4-full-ema.ckpt)

## Training Details

- **Hardware:** 2x A6000 GPUs
- **Epochs:** 171 (~7-8 hours of training)
- **Training Process:**
  - Cleaned and preprocessed dataset.
  - Fine-tuned the model while balancing visual coherence and retention of original Stable Diffusion knowledge.

## Model Checkpoints

The trained model is available on Hugging Face:
[Fine-Tuned Digimon Model](https://huggingface.co/tmar5)

## Results

A tradeoff in model performance:
- **Epoch >142:** Digimon images are more visually coherent but may start overfitting (catastrophic forgetting).
- **Epoch <99:** Model retains more general knowledge but produces less refined Digimon visuals.

### Example Outputs:
(Include images here)

## Conclusion
Fine-tuning Stable Diffusion on a Digimon dataset enables better Digimon-specific generations while requiring careful balancing to prevent overfitting. Further improvements can be made by adjusting dataset diversity and fine-tuning strategies.

## References
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Original Digimon Dataset Source](https://wikimon.net/Visual_List_of_Digimon)

