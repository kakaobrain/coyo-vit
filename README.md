# VisionTransformer

This repository attempted to reproduce the ViT from the COYO-Labeled-300M dataset.

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [COYO-700M: Image-Text Pair Dataset](https://github.com/kakaobrain/coyo-dataset)
- [COYO-Labeled-300M: Image-labeled Dataset](https://github.com/kakaobrain/coyo-dataset/subset/coyo-labeled-300m)

The model was pre-trained on the labeled COYO-Labeled-300M dataset, which is the largest number of published
classification ViT.

We provide the code for pretraining and finetuning in Tensorflow2.

We will also work with HuggingFace to provide the weights file and make it usable in pytorch and jax through the
HuggingFace platform as well.

## Training

- We have trained and evaluated using tpu-v3 with bfloat16.

- The pretraining weight we provide is last_checkpoint trained with COYO-Labeled-300M.

- The finetuing weights we provide are best_checkpoint trained with Imagenet.

- We used the hyperparameter search below to explore the best_weight files in finetuing.
    ```
    learning_rate:  [0.06, 0.03, 0.01]
    steps:          [20_000, 40_000]
    ```

- We provide a weight file trained in bfloat16 and we have confirmed that there is a performance change when evaluating
  with float32. (But imagenet-real was evaluated in float32)

- The code in this repository can be reproduced on gpu as well as tpu.
    ```
    # configs/trainer.yaml
    ---
    runtime:
      strategy: 'tpu' # one of ['cpu', 'tpu', 'gpu', 'gpu_multinode', 'gpu_multinode_async']
      use_mixed_precision: true
      tpu:
        version: 2.8.0
        name: ???
        zone: 'europe-west4-a'
        type: 'v3-32'
    ---
    change to
    ---
    runtime:
      strategy: 'gpu'
      use_mixed_precision: true
    ---
    ```
- To train, you need to set the path to your dataset in here.
    ```
    # configs/dataset/coyo300m.yaml
    train:
      cache: false
      supervised_key: 'labels'
      builder:
        - tfds_name: null
          tfds_data_dir: {your dir}
          tfds_split: 'train'
    
    validation:
      cache: false
      supervised_key: 'labels'
      builder:
        - tfds_name: null
          tfds_data_dir: {your dir}
          tfds_split: 'validation[:50000]' # We performed validation as part of the Imagenet21k dataset. Or you can use subset of COYO-Labeled-300M
    ```

## Results

| Model    	| Upstream Dataset  	| Resolution 	| ImageNet (downstream) 	| ImageNet-ReaL (downstream) 	| Public 	|
|----------	|-------------------	|------------	|-----------------------	|----------------------------	|--------	|
| ViT-L/16 	| JFT-300M          	| 512        	| 87.76                 	| 90.54                      	| X      	|
| ViT-L/16 	| COYO-Labeled-300M 	| 512        	| 87.24 (-0.52)         	| 90.03 (-0.51)              	| O      	|
| ViT-L/16 	| JFT-300M          	| 384        	| 87.12                 	| 89.99                      	| X      	|
| ViT-L/16 	| COYO-Labeled-300M 	| 384        	| 86.72 (-0.40)         	| 89.84 (-0.15)              	| O      	|

## Checkpoints

| Model    	| Upstream Dataset  	| Downstream Dataset 	| Resolution 	| link                                                                          	|
|----------	|-------------------	|--------------------	|------------	|-------------------------------------------------------------------------------	|
| ViT-L/16 	| COYO-Labeled-300M 	| -                  	| 224        	| [ link ]( https://huggingface.co/kakaobrain/vit-l16-coyo-labeled-300m )        	|
| ViT-L/16 	| COYO-Labeled-300M 	| ImageNet           	| 384        	| [ link ]( https://huggingface.co/kakaobrain/vit-l16-coyo-labeled-300m-i1k384 ) 	|
| ViT-L/16 	| COYO-Labeled-300M 	| ImageNet           	| 512        	| [ link ]( https://huggingface.co/kakaobrain/vit-l16-coyo-labeled-300m-i1k512 ) 	|

## Requirements

- We have tested our codes on the environment below
- `python==3.7.3` / `tensorflow==2.8.0` / `tensorflow-datasets==4.5.0`
- Please run the following command to install the necessary dependencies
    ```
    pip install -r requirements.txt
    ```

## Commands

We have used hydra to manage the configuration. For detailed usage, see [here](https://hydra.cc/).

### Pretraining

```bash
python3 -m trainer trainer=vit_l16_coyo300m \
runtime.tpu.name={your_tpu_name} \
runtime.tpu.type={your_tpu_type} \
experiment.debug=false experiment.save_dir={your_save_dir}
```

### Finetuning

```bash
python3 -m trainer trainer=vit_l16_i1k_downstream \
  runtime.tpu.name={your_tpu_name} \
  runtime.tpu.type={your_tpu_type} \
  experiment.debug=false \
  experiment.save_dir={your_save_dir} \
  trainer.backbone.pretrained={your_pretrained_weight} 
```

Also, you can experiment by changing the configuration as follows.

```bash
python3 -m trainer trainer=vit_l16_i1k_downstream \
  runtime.tpu.name={your_tpu_name} \
  runtime.tpu.type={your_tpu_type} \
  experiment.debug=false experiment.save_dir={your_save_dir} \
  trainer.backbone.pretrained={your_pretrained_weight} \
  trainer.epochs=16 \
  trainer.learning_rate.base_lr=3e-2
```

### Evaluation

```bash
python3 -m trainer trainer=vit_l16_i1k_downstream \
  runtime.tpu.name={your_tpu_name} \
  runtime.tpu.type={your_tpu_type} \
  experiment.debug=false \
  experiment.save_dir={your_weight_path} \
  experiment.mode='eval'
```

## Citation
```bibtex
@misc{kakaobrain2022coyo-vit,
  title         = {COYO-ViT},
  author        = {Lee, Sungjun and Park, Beomhee},
  year          = {2022},
  howpublished  = {\url{https://github.com/kakaobrain/coyo-vit}},
}
```
```bibtex
@misc{kakaobrain2022coyo-700m,
  title         = {COYO-700M: Image-Text Pair Dataset},
  author        = {Byeon, Minwoo and Park, Beomhee and Kim, Haecheon and Lee, Sungjun and Baek, Woonhyuk and Kim, Saehoon},
  year          = {2022},
  howpublished  = {\url{https://github.com/kakaobrain/coyo-dataset}},
}
```
```bibtex
@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## People
  - Sungjun Lee ([@justhungryman](https://github.com/justHungryMan))
  - Beomhee Park ([@beomheepark](https://github.com/beomheepark))

## Contact

This is released as an open source in the hope that it will be helpful to many research institutes and startups for research purposes. 

[jun.untitled@kakaobrain.com](mailto:jun.untitled@kakaobrain.com)

## License

The source codes are licensed under Apache 2.0 License.