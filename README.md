# Fine-Tuning Stable Diffusion on Digimon Dataset

This repository contains the fine-tuning of Stable Diffusion v1.4 on a dataset of Digimon images, enabling the model to generate Digimon-style images with improved coherence and specificity.

## Dataset

The dataset was crawled from [Wikimon's Visual List of Digimon](https://wikimon.net/Visual_List_of_Digimon) and is available for download:
[Google Drive Link](https://drive.google.com/drive/folders/1tmcdsoX67NvmAgtmGJgo6kb3N6SlJeLu)

## Base Model

We fine-tuned Stable Diffusion v1.4 using the following base model:
[Stable Diffusion v1.4 (Hugging Face)](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4-full-ema.ckpt)

## Training Details

- **Batch Size:** 4 (if trained on 1xA6000 GPU, we recommend using gradient accumulation of 2 to avoid noisy loss, stabilizing training and preventing memory overflow.)
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

| Epoch Range | Observations |
|-------------|-------------|
| **Epoch >=142** | Digimon images are more visually coherent but may start overfitting and forgetting information. |
| **Epoch <=99** | Model retains more general knowledge but produces less refined Digimon visuals. |

### Example Outputs:
#### "A strong Digimon"
| Epoch 99 | Epoch 128 | Epoch 142 | Epoch 171 |
|----------|----------|----------|----------|
| ![download](https://github.com/user-attachments/assets/46f68d9e-a845-42bb-bde6-22a0dfbd2609) | ![download](https://github.com/user-attachments/assets/431d1618-9953-4eb9-ba60-5103837a31d0) | ![download](https://github.com/user-attachments/assets/782c125d-c6cf-4a4a-80d9-946a838f97bc) | ![download](https://github.com/user-attachments/assets/133dea61-7052-4d97-a91c-18a8f6b73522) |

#### "Bob Marley"
| Epoch 99 | Epoch 128 | Epoch 142 | Epoch 171 |
|----------|----------|----------|----------|
| ![download](https://github.com/user-attachments/assets/af451a30-51c9-443d-a080-2ef3019eb13e) | ![download](https://github.com/user-attachments/assets/603e22d9-a6bd-42a1-8ae2-ada262b3d2ab) | ![download](https://github.com/user-attachments/assets/645c0037-9fff-4f6b-ab7b-1a6f8d329f04) | ![download](https://github.com/user-attachments/assets/ab96393e-b17a-4baf-8cc3-e1fbc38c5d53) |

#### Other Examples (Epoch 142 as a good compromise)
| Prompt | Image |
|--------|--------|
| "A robot humanoid" | ![download](https://github.com/user-attachments/assets/1017b51b-1c6c-4df5-9d92-79709cd538c9) |
| "A tiger Digimon" | ![download](https://github.com/user-attachments/assets/481cb062-abd0-45f5-9f84-87174634b695) |
| "A Robocop Digimon" | ![download](https://github.com/user-attachments/assets/c1e38fb2-fe6b-4eb2-a3c6-084b1ded6c04) |

### Generation Settings
- **Prompt Modifications:** prompt + ", high quality, 4k"
- **Negative Prompt:** "deformed, poor art, bad quality"
- **CFG Scale:** 7

## Conclusion
Fine-tuning Stable Diffusion on a Digimon dataset enables better Digimon-specific generations while requiring careful balancing to prevent overfitting. Further improvements can be made by adjusting dataset diversity and fine-tuning strategies.

## References
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Original Digimon Dataset Source](https://wikimon.net/Visual_List_of_Digimon)






















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
- **Batch Size:** 4 (if trained on 1xA6000 GPU, we recommend using gradient accumulation of 2 to avoid noisy loss, maintaining training stability while preventing memory overflow)

## Model Checkpoints

The trained model is available on Hugging Face:
[Fine-Tuned Digimon Model](https://huggingface.co/tmar5)

## Results



A tradeoff in model performance:
- **Epoch >142:** Digimon images are more visually coherent but may start overfitting (catastrophic forgetting).
- **Epoch <99:** Model retains more general knowledge but produces less refined Digimon visuals.

"a strong digimon"

Epoch 99:

![download](https://github.com/user-attachments/assets/46f68d9e-a845-42bb-bde6-22a0dfbd2609)

Epoch 128:
![download](https://github.com/user-attachments/assets/431d1618-9953-4eb9-ba60-5103837a31d0)


Epoch 142:
![download](https://github.com/user-attachments/assets/782c125d-c6cf-4a4a-80d9-946a838f97bc)

Epoch 171:
![download](https://github.com/user-attachments/assets/133dea61-7052-4d97-a91c-18a8f6b73522)




"bob marley"

Epoch 99:
![download](https://github.com/user-attachments/assets/af451a30-51c9-443d-a080-2ef3019eb13e)

Epoch 128:
![download](https://github.com/user-attachments/assets/603e22d9-a6bd-42a1-8ae2-ada262b3d2ab)


Epoch 142:
![download](https://github.com/user-attachments/assets/645c0037-9fff-4f6b-ab7b-1a6f8d329f04)


Epoch 171:
![download](https://github.com/user-attachments/assets/ab96393e-b17a-4baf-8cc3-e1fbc38c5d53)



Epoch 142 seem to be a good compromise

Other examples Epoch 142:
"a robot humanoid"
![download](https://github.com/user-attachments/assets/1017b51b-1c6c-4df5-9d92-79709cd538c9)

"a tiger digimon":
![download](https://github.com/user-attachments/assets/481cb062-abd0-45f5-9f84-87174634b695)

"a robocop digimon"
![download](https://github.com/user-attachments/assets/c1e38fb2-fe6b-4eb2-a3c6-084b1ded6c04)





Examples generated with
- prompt += ", high quality, 4k"
- neg_prompt = "deformed, poor art, bad quality"
- scale = 7 #CFG Scale


## Conclusion
Fine-tuning Stable Diffusion on a Digimon dataset enables better Digimon-specific generations while requiring careful balancing to prevent overfitting. Further improvements can be made by adjusting dataset diversity and fine-tuning strategies (ex: increasing the resolution).

## References
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Original Digimon Dataset Source](https://wikimon.net/Visual_List_of_Digimon)

