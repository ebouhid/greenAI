import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
import mlflow
from dataset.dataset import XinguDataset
from datetime import datetime
import glob
import time
import numpy as np
import os
import threading
from gpuprofiling import track_gpu
import argparse

parser = argparse.ArgumentParser(description="Your script description here")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
parser.add_argument("--stride_size", type=int, default=64, help="Stride size")
parser.add_argument("--encoder", type=str, default='resnet34', help="Encoder type")
parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
parser.add_argument("--tracking_interval", type=float, default=5.0, help="Tracking time interval in seconds")
parser.add_argument("--image_dir", type=str, default="./dataset/scenes_allbands_ndvi", help="Image directory")
parser.add_argument("--mask_dir", type=str, default="./dataset/truth_masks", help="Mask directory")
parser.add_argument("--train_regions", nargs='+', type=int, default=[1, 2, 6, 7, 8, 9, 10], help="Train regions")
parser.add_argument("--test_regions", nargs='+', type=int, default=[3, 4], help="Test regions")
parser.add_argument("--composition", nargs='+', type=int, default=[4, 3, 2], help="Composition as a list of integers")
parser.add_argument("--composition_name", type=str, default="RGB", help="Name of the composition")
parser.add_argument("--experiment_name", type=str, default="Default", help="Name of the experiment")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
PATCH_SIZE = args.patch_size
STRIDE_SIZE = args.stride_size
encoder = args.encoder
INFO = args.experiment_name
NUM_CLASSES = args.num_classes
TRACKING_INTERVAL = args.tracking_interval
COMPOSITION = args.composition
composition_name = args.composition_name

train_regions = args.train_regions
test_regions = args.test_regions



# TODO: Decouple compositions

os.environ['MLFLOW_EXPERIMENT_NAME'] = INFO

TRACKING_FNAME = f'./results/{INFO}-{encoder}-{composition_name}-power.csv'
# Start GPU power draw tracking
stop_event = threading.Event()
tracking_thread = threading.Thread(target=track_gpu,
                                    args=(TRACKING_INTERVAL, TRACKING_FNAME,
                                            stop_event))
tracking_thread.start()

CHANNELS = len(COMPOSITION)
# (model, loss, lr)
configs = [
    (smp.DeepLabV3Plus(
        in_channels=CHANNELS,
        classes=NUM_CLASSES,
        activation='sigmoid',
        encoder_name=encoder,
        encoder_weights=None,
    ), smp.utils.losses.JaccardLoss(), 5e-4),
]

for (model, loss, lr) in configs:
    with mlflow.start_run():
        best_epoch = 0
        max_f1 = 0
        max_precision = 0
        max_iou = 0
        max_accuracy = 0
        max_recall = 0

        print(f"{10 * '#'} {model.__class__.__name__} {10*'#'}")
        # instantiating datasets
        train_ds = XinguDataset(args.image_dir,
                                args.mask_dir,
                                COMPOSITION,
                                train_regions,
                                PATCH_SIZE,
                                STRIDE_SIZE,
                                transforms=True)
        test_ds = XinguDataset(args.image_dir,
                                args.mask_dir,
                                COMPOSITION,
                                test_regions,
                                PATCH_SIZE,
                                PATCH_SIZE,
                                transforms=False)

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=lr),
        ])

        metrics = [
            smp.utils.metrics.IoU(),
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.Precision(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall()
        ]

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device='cuda',
            verbose=True,
        )
        test_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device='cuda',
            verbose=True,
        )

        # dataloaders for this fold
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=True,
            num_workers=8)

        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=8)

        # logging parameters
        mlflow.log_params({
            "model": model.__class__.__name__,
            "loss": loss.__class__.__name__,
            "lr": lr,
            "composition": composition_name,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patch_size": PATCH_SIZE,
            "stride_size": STRIDE_SIZE,
            "Description": INFO,
            "train_regions": train_regions,
            "test_regions": test_regions,
            "encoder": encoder
        })

        start = time.time()
        torch.cuda.reset_max_memory_allocated()

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f'\nEpoch: {epoch}')
            train_logs = train_epoch.run(train_dataloader)
            test_logs = test_epoch.run(test_dataloader)

            if max_iou < test_logs['iou_score']:
                torch.save(
                    model,
                    f'./models/{INFO}-{encoder}-{model.__class__.__name__}-{composition_name}.pth'
                )
                torch.save(
                    model.state_dict(),
                    f'./models/{INFO}-{encoder}-{model.__class__.__name__}-{composition_name}-StateDict.pth'
                )

                max_iou = test_logs['iou_score']
                max_precision = test_logs['precision']
                max_f1 = test_logs['fscore']
                max_accuracy = test_logs['accuracy']
                max_recall = test_logs['recall']
                best_epoch = epoch
                print('Model saved!')

            # gathering data
            loss_train = next(iter(train_logs.values()))
            iou_score_train = train_logs['iou_score']

            precision_test = test_logs['precision']
            f1_test = test_logs['fscore']
            iou_score_test = test_logs['iou_score']
            accuracy_test = test_logs['accuracy']
            recall_test = test_logs['recall']

            # logging to mlflow
            mlflow.log_metric('train_loss', loss_train, epoch)
            mlflow.log_metric('train_iou', iou_score_train, epoch)
            mlflow.log_metric('test_precision', precision_test, epoch)
            mlflow.log_metric('test_f1', f1_test, epoch)
            mlflow.log_metric('test_iou', iou_score_test, epoch)
            mlflow.log_metric('test_accuracy', accuracy_test, epoch)
            mlflow.log_metric('test_recall', recall_test, epoch)

        end = time.time()
        execution_time = end - start
        stop_event.set()
        tracking_thread.join()

        # Convert execution time to minutes and seconds
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)

        max_memory_usage = torch.cuda.max_memory_reserved() / (1024**2)
        with open(
                f'results/results-{INFO}-{encoder}-{model.__class__.__name__}-{lr :.0e}-{composition_name}.txt',
                'w') as f:
            f.write("TEST RESULTS\n")
            f.write(f'{model.__class__.__name__}-{composition_name}\n')
            f.write(f'Precision: {max_precision :.4f}\n')
            f.write(f'F1 Score: {max_f1 :.4f}\n')
            f.write(f'IoU: {max_iou :.4f}\n')
            f.write(f'Accuracy: {max_accuracy :.4f}\n')
            f.write(f'Recall: {max_recall :.4f}\n')
            f.write(f'On epoch: {best_epoch}\n')
            f.write(f'Time: {minutes}m {seconds}s\n')