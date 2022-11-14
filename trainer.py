import hydra
import logging
import os
import time
from typing import List, Tuple, Dict, AnyStr, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from omegaconf import DictConfig, OmegaConf, ListConfig
from dataloaders.dataset import TfdsInputLoader
from common.distribute_utils import get_distribution_strategy
from optimizers import ExponentialMovingAverage, AdamWeightDecay
from losses import BCELoss
from metrics import F1Score, MeanAveragePrecision, TopKPrecision
from callbacks.moving_average import MovingAverageCallback
from callbacks.checkpoint import CheckpointManagerCallback
from models.classification import ClassificationModel
from backbones import build_backbone
from schedulers import get_schedule
from utils import set_mixed_precision_policy, set_pretrained_pos_embed_for_vit

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
                 trainer_config: DictConfig,
                 experiment_config: DictConfig,
                 strategy: tf.distribute.Strategy,
                 **kwargs):
        self.trainer_config = trainer_config
        self.experiment_config = experiment_config
        self.strategy = strategy

        self.num_workers = self.strategy.num_replicas_in_sync
        if self.trainer_config.get('global_batch_size', None) is not None:
            self.global_batch_size = self.trainer_config.global_batch_size
            self.local_batch_size = self.global_batch_size // self.num_workers
        else:
            self.local_batch_size = self.trainer_config.local_batch_size
            self.global_batch_size = self.local_batch_size * self.num_workers

        self.debug = self.experiment_config.debug

        self.save_dir = self.experiment_config.save_dir
        self.log_dir = os.path.join(self.save_dir, 'logs')
        self.ckpt_dir = os.path.join(self.save_dir, 'ckpt')

        logger.info(f'strategy: {self.strategy}')
        logger.info(f'num_workers: {self.num_workers}')
        logger.info(f'local_batch_size: {self.local_batch_size}, global_batch_size: {self.global_batch_size}')

    def build_model(self, num_classes) -> ClassificationModel:
        model_config = self.trainer_config.backbone
        backbone_name = model_config.backbone_name
        logger.info(f'Build backbone (name={backbone_name})')
        backbone = build_backbone(backbone_name,
                                  OmegaConf.to_object(model_config.get('backbone_params', DictConfig({}))))
        kernel_init = model_config.cls_kernel_init
        bias_init = model_config.cls_bias_init

        model = ClassificationModel(
            backbone=backbone,
            dropout_rate=model_config.dropout_rate,
            num_classes=num_classes,
            weight_decay=self.trainer_config.loss.l2_weight_decay,  # TODO; move to losses
            kernel_initializer=getattr(tf.keras.initializers, kernel_init.type)(**kernel_init.get('params', {})),
            bias_initializer=getattr(tf.keras.initializers, kernel_init.type)(**kernel_init.get('params', {})),
            clip_grad_norm=self.trainer_config.optimizer.get('clip_norm_grad', 0.)
        )

        pretrained = model_config.get('pretrained', None)
        if pretrained:
            logger.info(f'load pretrained: weights from {pretrained}')
            checkpoint = tf.train.Checkpoint(backbone=model.backbone)
            try:
                checkpoint.restore(pretrained).expect_partial()
            except ValueError as e:
                logger.info(f'load pretrained: restore: {e}')
            if backbone_name.startswith('vit'):
                set_pretrained_pos_embed_for_vit(backbone, ckpt_path=pretrained)

        model.backbone.summary(expand_nested=True)

        return model

    @property
    def use_moving_average(self):
        optimizer_config: DictConfig = self.trainer_config.optimizer
        return optimizer_config.moving_average_decay > 0.

    def build_optimizer(self,
                        model: tf.keras.Model,
                        scheduler: tf.keras.optimizers.schedules.LearningRateSchedule = None
                        ) -> tf.keras.optimizers.Optimizer:
        optimizer_config: DictConfig = self.trainer_config.optimizer
        optimizer_dict = OmegaConf.to_object(optimizer_config)

        if scheduler is not None:
            optimizer_dict['config'].update(learning_rate=scheduler)

        if optimizer_dict['class_name'].lower() == 'adamw':
            optimizer = AdamWeightDecay(**optimizer_dict['config'])
        else:
            optimizer = tf.keras.optimizers.get(optimizer_dict)

        if self.use_moving_average:
            optimizer = ExponentialMovingAverage(optimizer,
                                                 trainable_weights_only=False,
                                                 average_decay=optimizer_config.moving_average_decay)
            optimizer.shadow_copy(model)

        logger.info(f'optimizer: {type(optimizer)}')
        for k, v in optimizer.get_config().items():
            logger.info(f'    {k}: {v}')
        return optimizer

    def build_loss(self) -> tf.keras.losses.Loss:
        loss_config: DictConfig = self.trainer_config.loss
        loss_dict = OmegaConf.to_object(loss_config)
        if loss_dict['class_name'] == 'BCELoss':
            loss = BCELoss(**loss_dict['config'])
        else:
            loss = tf.keras.losses.get(loss_dict)
        logger.info(f'Build loss: {type(loss)}')
        return loss

    def build_metrics(self, num_classes) -> List[tf.keras.metrics.Metric]:
        metrics_config: DictConfig = self.trainer_config.metrics
        metrics_dict = OmegaConf.to_object(metrics_config)

        logger.info(f'Build metrics...')
        metrics_list = metrics_dict['metrics_list']
        metrics = list()
        for m in metrics_list:
            m['config'] = m.get('config', {})

            m['config'] = m.get('config', {})
            if m['class_name'] == 'F1Score':
                m['config']['num_classes'] = num_classes
                metric = F1Score(**m['config'])
            elif m['class_name'] == 'MeanAveragePrecision':
                m['config']['num_classes'] = num_classes
                metric = MeanAveragePrecision(**m['config'])
            elif m['class_name'] == 'TopKPrecision':
                metric = TopKPrecision(**m['config'])
            else:
                metric = tf.keras.metrics.get(m)
            metrics.append(metric)
        return metrics

    def build_dataset(self, data_config: DictConfig, is_training: bool) -> Tuple[
        tf.distribute.DistributedDataset, Dict]:
        logger.info(f'Build dataset (is_training={is_training})')
        logger.info(f'   {data_config.builder}')

        train_input = TfdsInputLoader(is_training=is_training,
                                      tfds_build_list=data_config.builder,
                                      preprocess_config=data_config.get('preprocess', []),
                                      mixup_alpha=data_config.get('mixup_alpha', 0.),
                                      cutmix_alpha=data_config.get('cutmix_alpha', 0.),
                                      image_dtype=data_config.dtype,
                                      image_size=data_config.image_size,
                                      supervised_key=data_config.supervised_key,
                                      normalize_label=data_config.get('normalize_label', False),
                                      cache=data_config.cache
                                      )
        dataset = self.strategy.distribute_datasets_from_function(
            train_input.distribute_dataset_fn(self.global_batch_size)
        )
        info = train_input.info
        return dataset, info

    def build_callbacks(self, checkpoint_manager: tf.train.CheckpointManager = None) -> List[
        tf.keras.callbacks.Callback]:
        logger.info('Build callbacks...')

        callbacks = list()
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                        write_steps_per_second=True,
                                                        update_freq='epoch',
                                                        profile_batch=0))
        if checkpoint_manager:
            callbacks.append(CheckpointManagerCallback(checkpoint_manager))

        if self.use_moving_average:
            callbacks.append(MovingAverageCallback(overwrite_weights_on_train_end=True))

        return callbacks

    def build_checkpoint_manager(self, model, checkpoint_interval_steps: int, max_to_keep: int = 16):
        checkpoint = tf.train.Checkpoint(model=model,
                                         backbone=model.backbone,
                                         optimizer=model.optimizer,
                                         ckpt_saved_iterations=model.optimizer.iterations
                                         )

        checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                        directory=self.ckpt_dir,
                                                        step_counter=model.optimizer.iterations,
                                                        checkpoint_interval=checkpoint_interval_steps,
                                                        max_to_keep=max_to_keep)
        return checkpoint_manager

    def build_scheduler(self,
                        steps_per_epoch,
                        global_batch_size) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        learning_rate_config: DictConfig = self.trainer_config.learning_rate

        if self.trainer_config.get('steps', None) is not None:
            self.trainer_config.epochs = self.trainer_config.steps // steps_per_epoch

        end_epochs = self.trainer_config.epochs
        schedule = get_schedule(learning_rate_config=learning_rate_config,
                                global_batch_size=global_batch_size,
                                steps_per_epoch=steps_per_epoch,
                                end_epochs=end_epochs)

        return schedule

    def _train_stage(self, model=None, skip_validation=True):
        logging.info(OmegaConf.to_yaml(self.trainer_config))
        with self.strategy.scope():
            train_data, train_info = self.build_dataset(self.trainer_config.dataset.train, is_training=True)
            steps_per_epoch = max(1, train_info['num_examples'] // self.global_batch_size)

            logger.info('Build the model...')
            model = model or self.build_model(num_classes=train_info['num_classes'])
            scheduler = self.build_scheduler(steps_per_epoch, self.global_batch_size)
            optimizer = self.build_optimizer(model, scheduler)
            loss = self.build_loss()
            metrics = self.build_metrics(num_classes=train_info['num_classes'])

            logger.info('Compile the model...')
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                steps_per_execution=1 if self.debug else steps_per_epoch // 2,
            )

            if skip_validation:
                validation_args = dict()
            else:
                logger.info('Build validation dataset...')
                val_data, val_info = self.build_dataset(self.trainer_config.dataset.validation, is_training=False)
                steps_per_val = val_info['num_examples'] // self.global_batch_size
                validation_args = dict(
                    validation_data=val_data,
                    validation_steps=steps_per_val,
                    validation_batch_size=self.global_batch_size,
                    validation_freq=1,
                )

            checkpoint_manager = self.build_checkpoint_manager(model, steps_per_epoch)
            if checkpoint_manager.latest_checkpoint:
                logger.info(f'Restore or initialize the model from {checkpoint_manager.latest_checkpoint}')
                checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint)
            initial_epoch = optimizer.iterations // steps_per_epoch
            if initial_epoch >= self.trainer_config.epochs:
                return
            callbacks = self.build_callbacks(checkpoint_manager)

            logger.info('Train the model...')
            model.fit(
                train_data,
                callbacks=callbacks,
                initial_epoch=initial_epoch,
                epochs=self.trainer_config.epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=2 if not self.debug and isinstance(self.strategy, tf.distribute.TPUStrategy) else 1,
                **validation_args
            )
            model.save_weights(os.path.join(self.save_dir, 'model-weights'))
            model.backbone.save_weights(os.path.join(self.save_dir, 'backbone-weights'))

    def train(self, skip_validation=True):
        self._train_stage(skip_validation=skip_validation)

    def eval(self, ckpt=None):
        if ckpt is None:
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
            if ckpt is None:
                raise ValueError(f'cannot found the latest checkpoint in {self.ckpt_dir}')

        with self.strategy.scope():
            logging.info(f'evaluate checkpoint: {ckpt}')

            val_data, val_info = self.build_dataset(self.trainer_config.dataset.validation, is_training=False)
            steps_per_val = val_info['num_examples'] // self.global_batch_size

            model = self.build_model(num_classes=val_info['num_classes'])

            logger.info('Compile the model...')
            optimizer = self.build_optimizer(model)
            loss = self.build_loss()
            metrics = self.build_metrics(num_classes=val_info['num_classes'])
            callbacks = self.build_callbacks()

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                steps_per_execution=1 if self.debug else steps_per_val,
            )

            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(ckpt)
            logger.info('Evaluate the model...')
            eval_results = model.evaluate(
                val_data,
                steps=steps_per_val,
                callbacks=callbacks,
                verbose=2 if not self.debug and isinstance(self.strategy, tf.distribute.TPUStrategy) else 1
            )
            logging.info(eval_results)
        return eval_results


@hydra.main(config_path="configs", config_name="trainer")
def train_main(config: DictConfig):
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")

    strategy = get_distribution_strategy(device=config.runtime.strategy,
                                         tpu_address=config.runtime.get('tpu', {}).get('name'))

    set_mixed_precision_policy(strategy, config.runtime.use_mixed_precision)

    trainer = Trainer(config.trainer, config.experiment, strategy)

    experiment_mode = config.experiment.mode
    if experiment_mode == 'train':
        trainer.train()
    elif experiment_mode == 'train_eval':
        trainer.train(skip_validation=False)
    elif experiment_mode == 'eval':
        trainer.eval(config.experiment.save_dir)
    else:
        raise ValueError('invalid config.mode: {}'.format(experiment_mode))


if __name__ == "__main__":
    train_main()
