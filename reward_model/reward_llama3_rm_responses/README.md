---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: meta-llama/Meta-Llama-3-8B
metrics:
- accuracy
model-index:
- name: reward_llama3_rm_responses
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# reward_llama3_rm_responses

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on the mut_pref_data_responses dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6331
- Accuracy: 0.6964

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Accuracy |
|:-------------:|:------:|:-----:|:---------------:|:--------:|
| 0.6174        | 0.3703 | 2000  | 0.5997          | 0.6922   |
| 0.6025        | 0.7406 | 4000  | 0.5836          | 0.6993   |
| 0.5963        | 1.1110 | 6000  | 0.5818          | 0.6985   |
| 0.5816        | 1.4813 | 8000  | 0.5936          | 0.6922   |
| 0.5867        | 1.8516 | 10000 | 0.6069          | 0.7026   |
| 0.5598        | 2.2219 | 12000 | 0.6337          | 0.6960   |
| 0.5665        | 2.5922 | 14000 | 0.6236          | 0.6976   |
| 0.5624        | 2.9626 | 16000 | 0.6329          | 0.6960   |


### Framework versions

- PEFT 0.11.1
- Transformers 4.41.2
- Pytorch 2.3.0
- Datasets 2.19.2
- Tokenizers 0.19.1