"""
Train a model on Omniglot.
"""

import random
from pathlib import Path

import tensorflow.compat.v1 as tf
# The usage of tensorflow in this code needs eager execution disabled
tf.disable_eager_execution()

from pytorch_lightning.loggers import TensorBoardLogger

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/omniglot'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    RUN_DIR = Path(__file__).resolve(strict=True).parent.parent / 'mlmi-federated-learning' / 'run'
    experiment_path = RUN_DIR / 'supervised-reptile' / (
        f"seed{args.seed};{args.classes}-way{args.shots}-shot;"
        f"train_shots{args.train_shots}ib{args.inner_batch}ii{args.inner_iters}"
        f"lr{str(args.learning_rate).replace('.', '')}"
        f"ms{str(args.meta_step).replace('.', '')}"
        f"msf{str(args.meta_step_final).replace('.', '')}"
        f"mb{args.meta_batch}eb{args.eval_batch}"
        f"{'sgd' if args.sgd else ''}"
    )
    tensorboard_logger = TensorBoardLogger(experiment_path.absolute())

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(tensorboard_logger, sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)

        for label, dataset in zip(['Train', 'Test'], [train_set, test_set]):
            accuracy = evaluate(sess, model, dataset, **eval_kwargs)
            tensorboard_logger.experiment.add_scalar(
                f'final_{label}_acc',
                accuracy,
                global_step=0
            )
            print(f'{label} accuracy: {accuracy}')

if __name__ == '__main__':
    main()
