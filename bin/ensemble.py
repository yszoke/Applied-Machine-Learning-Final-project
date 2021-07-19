#!/usr/bin/python
"""
Run an ensemble experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under GPLv2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import os
import argparse
import toupee as tp
import dill
import logging
import wandb

from toupee.parameters import Parameters

#1
def create_model(params, epochs=1, use_wandb=False):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logging.info(("using toupee version {0}".format(tp.version)))
    params.epochs = epochs
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    wandb_params = None
    if use_wandb:
        dataset_name = os.path.basename(os.path.normpath(params.dataset))
        wandb_project = f"toupee-{dataset_name}"
        group_id = wandb.util.generate_id()
        wandb_group = f"{dataset_name}-{params.ensemble_method['class_name']}-{group_id}"
        wandb_params = {"project": wandb_project, "group": wandb_group}

    return tp.ensembles.create(params=params, data=data, wandb=wandb_params), params


def fit_method(method, params, wandb_params=None, save_file_dir=None):
    metrics = method.fit()

    # print this to csv
    logging.info('\n{:*^40}'.format(' Ensemble trained in %.2fm ' % (metrics['time'] / 60.)))
    logging.info(metrics['ensemble']['classification_report'])
    tp.utils.pretty_print_confusion_matrix(metrics['ensemble']['confusion_matrix'])

    if wandb_params:
        run = wandb.init(project=wandb_params["project"], reinit=True,
                    config={"type": "ensemble", "params": params.__dict__},
                    group=wandb_params["group"],
                    name="finished-ensemble")
        for i, member_metrics in enumerate(metrics['members'].to_dict('records')):
            wandb.log({k: v for k, v in member_metrics.items() if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({f"ensemble_{k}": v for k, v in metrics['round_cumulative'][i].items()
                       if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({'member': i, 'step': i, 'epoch': i})
        for metric, value in metrics['ensemble'].items():
            wandb.run.summary[metric] = value
        wandb.run.summary['total time'] = metrics['time']
        run.finish()

    logging.info('\n{:*^40}'.format(" Member Metrics "))
    for metric_name in tp.PRINTABLE_METRICS:
        logging.info(f"{metric_name}: {metrics['members'][metric_name].tolist()}")
    logging.info('\n{:*^40}'.format(" Aggregate Metrics "))
    # print this to csv -> final model results
    tp.log_metrics(metrics["ensemble"])

    if save_file_dir:
        method.save(save_file_dir)
        dill.dump(metrics, save_file_dir + '.metrics')

    return metrics






def ensemble_main(args=None, params=None, params_file=None, epochs=0):
    """ Train a base model as specified """
    # parser = argparse.ArgumentParser(description='Train a single Base Model')
    # parser.add_argument('params_file', help='the parameters file')
    # parser.add_argument('save_file', nargs='?',
    #                     help='the file where the trained MLP is to be saved')
    # parser.add_argument('--epochs', type=int, nargs='?',
    #                     help='number of epochs to run')
    # parser.add_argument('--adversarial-testing', action="store_true",
    #                     help="Test for adversarial robustness")
    # parser.add_argument('--wandb', action="store_true",
    #                     help="Send results to Weights and Biases")
    # parser.add_argument('--wandb-project', type=str, help="Weights and Biases project name")
    # parser.add_argument('--wandb-group', type=str, help="Weights and Biases group name")
    # args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logging.info(("using toupee version {0}".format(tp.version)))
    if not params and params_file:
        params = tp.config.load_parameters(params_file)
    elif not params and not params_file:
        params = tp.config.load_parameters(args.params_file)
    if args and args.epochs:
        params.epochs = args.epochs
    elif epochs > 0:
        params.epochs = epochs
    data = tp.data.Dataset(src_dir=params.dataset, **params.__dict__)
    wandb_params = None
    if args and args.wandb:
        dataset_name = os.path.basename(os.path.normpath(params.dataset))
        wandb_project = args.wandb_project or f"toupee-{dataset_name}"
        group_id = wandb.util.generate_id()
        wandb_group = args.wandb_group or f"{dataset_name}-{params.ensemble_method['class_name']}-{group_id}"
        wandb_params = {"project": wandb_project, "group": wandb_group}
    if args:
        method = tp.ensembles.create(params=params, data=data, wandb=wandb_params, adversarial_testing=args.adversarial_testing)
    else:
        method = tp.ensembles.create(params=params, data=data, wandb=wandb_params)
    metrics = method.fit()
    # print this to csv
    logging.info('\n{:*^40}'.format(' Ensemble trained in %.2fm ' % (metrics['time'] / 60.)))
    logging.info(metrics['ensemble']['classification_report'])
    tp.utils.pretty_print_confusion_matrix(metrics['ensemble']['confusion_matrix'])
    if args and args.wandb:
        run = wandb.init(project=wandb_project, reinit=True,
                    config={"type": "ensemble", "params": params.__dict__},
                    group=wandb_group,
                    name="finished-ensemble")
        for i, member_metrics in enumerate(metrics['members'].to_dict('records')):
            wandb.log({k: v for k, v in member_metrics.items() if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({f"ensemble_{k}": v for k, v in metrics['round_cumulative'][i].items()
                       if k in tp.PRINTABLE_METRICS}, commit=False)
            wandb.log({'member': i, 'step': i, 'epoch': i})
        for metric, value in metrics['ensemble'].items():
            wandb.run.summary[metric] = value
        wandb.run.summary['total time'] = metrics['time']
        run.finish()
    logging.info('\n{:*^40}'.format(" Member Metrics "))
    for metric_name in tp.PRINTABLE_METRICS:
        logging.info(f"{metric_name}: {metrics['members'][metric_name].tolist()}")
    logging.info('\n{:*^40}'.format(" Aggregate Metrics "))
    # print this to csv -> final model results
    tp.log_metrics(metrics["ensemble"])

    if args and args.save_file:
        method.save(args.save_file)
        dill.dump(metrics, args.save_file + '.metrics')
    #save_metadata_etc


if __name__ == '__main__':
    ensemble_main()