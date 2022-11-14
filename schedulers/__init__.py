import math
import tensorflow as tf

from .warmup import WarmupDecaySchedule
from .warmup_cosinedecay import WarmupCosineDecay
from .warmup_polydecay import WarmupPolynomialDecay


def get_schedule(learning_rate_config,
                 global_batch_size,
                 steps_per_epoch,
                 end_epochs):
    # TODO; beautify.
    schedule_name = learning_rate_config.schedule_name

    if schedule_name.startswith('vit'):
        init_lr = learning_rate_config.init_lr
        base_lr = learning_rate_config.base_lr

        if schedule_name.endswith('/cosine'):
            total_steps = int(end_epochs * steps_per_epoch)
            end_learning_rate = learning_rate_config.end_learning_rate
            warmup_steps = int(learning_rate_config.warmup_steps)

            schedule = WarmupCosineDecay(
                base_lr=base_lr,
                init_lr=init_lr,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_learning_rate=end_learning_rate
            )

            return schedule

        elif schedule_name.endswith('/linear'):
            total_steps = int(end_epochs * steps_per_epoch)
            end_learning_rate = learning_rate_config.end_learning_rate
            warmup_steps = int(learning_rate_config.warmup_steps)
            power = learning_rate_config.power

            schedule = WarmupPolynomialDecay(
                base_lr=base_lr,
                decay_steps=total_steps,
                end_learning_rate=end_learning_rate,
                power=power,
                warmup_steps=warmup_steps,
                init_lr=init_lr
            )

            return schedule
        else:
            raise ValueError(f'Invalid value for schedule name: {schedule_name}')
    else:
        raise ValueError(f'Invalid value for schedule name: {schedule_name}')
