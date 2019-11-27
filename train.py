import argparse
import h5py
import logging
import os
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.utils import AverageMeter
from nli.NLIDataset import NliDataset
from nli.nli_models.PpoModel import PpoModel


def get_logger(exp_name):
    log_path = "nli/logs/log"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger("general_logger")
    handler = logging.FileHandler(f"{log_path}/exp-{exp_name}.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    print(f"Logger path: {log_path}/exp-{exp_name}.log")
    logger.info(f"args: {str(args)}")
    return logger


def get_summary_writer(exp_name):
    tensorboard_path = "nli/logs/tensorboard"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary_writer = dict()
    summary_writer["train"] = SummaryWriter(log_dir=os.path.join(tensorboard_path, 'exp-' + exp_name, 'train'))
    summary_writer["valid"] = SummaryWriter(log_dir=os.path.join(tensorboard_path, 'exp-' + exp_name, 'valid'))
    return summary_writer


def get_dataloader(args):
    train_data = NliDataset.load_data(f"data/nli/train_lower={args.lower}.pckl")
    valid_data = NliDataset.load_data(f"data/nli/valid_lower={args.lower}.pckl")
    test_data = NliDataset.load_data(f"data/nli/test_lower={args.lower}.pckl")
    print(f"train len: {len(train_data)}")
    print(f"valid len: {len(valid_data)}")

    train_dataset = NliDataset(train_data, max_len=args.max_len)
    valid_dataset = NliDataset(valid_data)
    test_dataset = NliDataset(test_data)

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                            collate_fn=NliDataset.collate_fn, pin_memory=True)
    valid_data = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                            collate_fn=NliDataset.collate_fn, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                           collate_fn=NliDataset.collate_fn, pin_memory=True)

    with h5py.File(f"data/nli/glove_lower={args.lower}.h5", 'r') as f:
        id_to_glove = f["glove"][...]

    args.vocab_size = id_to_glove.shape[0]
    args.label_size = NliDataset.label_size

    return train_data, valid_data, test_data, id_to_glove


def get_optimizer(args, policy_parameters, env_parameters):
    if args.pol_optimizer == "adam":
        pol_optimizer_class = torch.optim.Adam
    elif args.pol_optimizer == "adadelta":
        pol_optimizer_class = torch.optim.Adadelta
    else:
        pol_optimizer_class = torch.optim.SGD

    if args.env_optimizer == "adam":
        env_optimizer_class = torch.optim.Adam
    elif args.env_optimizer == "adadelta":
        env_optimizer_class = torch.optim.Adadelta
    else:
        env_optimizer_class = torch.optim.SGD

    optimizer = {"policy": pol_optimizer_class(params=policy_parameters, lr=args.pol_lr, weight_decay=args.l2_weight),
                 "environment": env_optimizer_class(params=env_parameters, lr=args.env_lr, weight_decay=args.l2_weight)}
    scheduler = {
        "policy": lr_scheduler.ReduceLROnPlateau(optimizer=optimizer["policy"], mode="max", factor=0.5, patience=10,
                                                 verbose=True),
        "environment": lr_scheduler.ReduceLROnPlateau(optimizer=optimizer["environment"], mode="max", factor=0.5,
                                                      patience=10, verbose=True)}
    return optimizer, scheduler


def test(test_data, model, epoch, device, logger):
    model.eval()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    with torch.no_grad():
        for labels, premises, p_mask, hypothese, h_mask in test_data:
            if torch.cuda.is_available():
                labels = labels.to(device=device)
                premises = premises.to(device=device)
                p_mask = p_mask.to(device=device)
                hypotheses = hypotheses.to(device=device)
                h_mask = h_mask.to(device=device)
            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = model(premises,
                                                                                                          p_mask,
                                                                                                          hypotheses,
                                                                                                          h_mask,
                                                                                                          labels)
            entropy = entropy.mean()
            normalized_entropy = normalized_entropy.mean()
            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            n = p_mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            entropy_meter.update(entropy.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)

    logger.info(f"Test: ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"entropy: {entropy_meter.avg:.4f} n_entropy: {n_entropy_meter.avg:.4f} ")
    return accuracy_meter.avg


def validate(valid_data, model, epoch, device, logger, summary_writer):
    model.eval()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    with torch.no_grad():
        for labels, premises, p_mask, hypotheses, h_mask in valid_data:
            if torch.cuda.is_available():
                labels = labels.to(device=device)
                premises = premises.to(device=device)
                p_mask = p_mask.to(device=device)
                hypotheses = hypotheses.to(device=device)
                h_mask = h_mask.to(device=device)

            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = model(premises,
                                                                                                          p_mask,
                                                                                                          hypotheses,
                                                                                                          h_mask,
                                                                                                          labels)
            entropy = entropy.mean()
            normalized_entropy = normalized_entropy.mean()
            n = p_mask.shape[0]
            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            entropy_meter.update(entropy.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"entropy: {entropy_meter.avg:.4f} n_entropy: {n_entropy_meter.avg:.4f} ")
    summary_writer["valid"].add_scalar(tag="ce", scalar_value=ce_loss_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="accuracy", scalar_value=accuracy_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="n_entropy", scalar_value=n_entropy_meter.avg, global_step=global_step)

    model.train()
    return accuracy_meter.avg


def train(train_data, valid_data, model, optimizers, schedulers, epoch, args, logger, summary_writer, exp_name):
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    # TODO(siyu) why tracking prob ratio
    prob_ratio_meter = AverageMeter()

    device = args.gpu_id
    model.train()
    # TODO(siyu) handle with loading models
    global best_val_accuracy

    with tqdm(total=len(train_data), desc=f"Train Epoch #{epoch + 1}") as t:
        for batch_idx, (labels, premises, p_mask, hypotheses, h_mask) in enumerate(train_data):
            if torch.cuda.is_available():
                labels = labels.to(device=device)
                premises = premises.to(device=device)
                p_mask = p_mask.to(device=device)
                hypotheses = hypotheses.to(device=device)
                h_mask = h_mask.to(device=device)
            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = model(premises,
                                                                                                          p_mask,
                                                                                                          hypotheses,
                                                                                                          h_mask,
                                                                                                          labels)
            ce_loss.backward()
            optimizers["environment"].step()
            optimizers["environment"].zero_grad()
            for k in range(args.ppo_updates):
                if k == 0:
                    new_normalized_entropy, new_actions_log_prob = normalized_entropy, actions_log_prob
                else:
                    new_normalized_entropy, new_actions_log_prob = model.evaluate_actions(premises, p_mask,
                                                                                          actions["p_actions"],
                                                                                          hypotheses, h_mask,
                                                                                          actions["h_actions"])
                prob_ratio = (new_actions_log_prob - actions_log_prob.detach()).exp()
                clamped_prob_ratio = prob_ratio.clamp(1.0 - args.epsilon, 1.0 + args.epsilon)
                ppo_loss = torch.max(prob_ratio * rewards, clamped_prob_ratio * rewards).mean()
                loss = ppo_loss - args.entropy_weight * new_normalized_entropy.mean()
                loss.backward()
                optimizers["policy"].step()
                optimizers["policy"].zero_grad()

            entropy = entropy.mean()
            normalized_entropy = normalized_entropy.mean()
            n = p_mask.shape[0]
            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            entropy_meter.update(entropy.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            prob_ratio_meter.update((1.0 - prob_ratio.detach()).abs().mean().item(), n)

            global global_step
            summary_writer["train"].add_scalar(tag="ce", scalar_value=ce_loss.item(), global_step=global_step)
            summary_writer["train"].add_scalar(tag="accuracy", scalar_value=accuracy.item(), global_step=global_step)
            summary_writer["train"].add_scalar(tag="n_entropy", scalar_value=normalized_entropy.item(),
                                               global_step=global_step)
            summary_writer["train"].add_scalar(tag="prob_ratio", scalar_value=prob_ratio_meter.value,
                                               global_step=global_step)

            global_step += 1

            if (batch_idx + 1) % (len(train_data) // 10) == 0:

                logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                            f"accuracy: {accuracy_meter.avg:.4f} entropy: {entropy_meter.avg:.4f} "
                            f"n_entropy: {n_entropy_meter.avg:.4f}")
                new_val_accuracy = validate(valid_data, model, epoch, device, logger, summary_writer)
                # print(f"validationg accuracy: {new_val_accuracy:.4f}")
                schedulers["environment"].step(new_val_accuracy)
                schedulers["policy"].step(new_val_accuracy)
                global best_model_path, best_val_accuracy
                if new_val_accuracy > best_val_accuracy:
                    model_dir = f"{args.model_dir}/{exp_name}"
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    best_model_path = f"{args.model_dir}/{exp_name}/{epoch}-{batch_idx}.mdl"
                    logger.info("saving model to" + best_model_path)
                    torch.save({"epoch": epoch, "batch_idx": batch_idx, "state_dict": model.state_dict()},
                               best_model_path)
                    best_val_accuracy = new_val_accuracy
                model.train()

            t.set_postfix({'loss': ce_loss_meter.avg,
                           'accuracy': 100. * accuracy_meter.avg})
            t.update(1)


def main(args):
    exp_name = f"bs{args.batch_size}hd{args.hidden_dim}ppo{args.ppo_updates}ew{args.entropy_weight}"
    logger = get_logger(exp_name)
    summary_writer = get_summary_writer(exp_name)

    # TODO(siyu) checking dataloader
    train_data, valid_data, test_data, vectors = get_dataloader(args)
    # TODO(siyu): adding baseline
    model = PpoModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     mlp_hidden_dim=args.mlp_hidden_dim,
                     label_dim=args.label_size,
                     dropout_prob=args.dropout_prob,
                     trans_hidden_dim=args.trans_hidden_dim,
                     use_batchnorm=args.use_batchnorm)
    if torch.cuda.is_available():
        model = model.cuda(args.gpu_id)
    dtype = model.parser_embedding.weight.data.dtype
    device = model.parser_embedding.weight.data.device
    model.parser_embedding.weight.data = torch.tensor(vectors, dtype=dtype, device=device)
    model.tree_embedding.weight.data = torch.tensor(vectors, dtype=dtype, device=device)
    if args.freeze_embeddings:
        model.parser_embedding.weight.requires_grad = False
        model.tree_embedding.weight.requires_grad = False
        logger.info("Using frozen embedding for parser and tree embedding layers.")

    # TODO(siyu) adding early parsing
    optimizers, schedulers = get_optimizer(args, policy_parameters=model.get_policy_parameters(),
                                           env_parameters=model.get_environment_parameters())

    for epoch in range(args.max_epoch):
        train(train_data, valid_data, model, optimizers, schedulers, epoch, args, logger, summary_writer, exp_name)
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    test(test_data, model, args.gpu_id, logger)


if __name__ == "__main__":
    args = {
        "gpu-id": -1,
        "lower": "True",
        "batch-size": 64,
        "use-batchnorm": "True",
        "dropout-prob": 0.1,
        "max-len": 120,
        "word-dim": 300,
        "hidden-dim": 256,
        "mlp-hidden-dim": 1024,
        "env-optimizer": "adadelta",
        "pol-optimizer": "adadelta",
        "freeze-embeddings": "True",
        "l2-weight": 0.0,
        "max-epoch": 150,
        "ppo-updates": 1,
        "entropy-weight": 0.0,
        "env-lr": 1.0,
        "pol-lr": 1.0,
        "epsilon": 0.2,
        "model-dir": "./nli/checkpoints",
        "trans-hidden-dim": 256,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--lower", default=args["lower"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--use-batchnorm", default=args["use-batchnorm"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--dropout-prob", default=args["dropout-prob"], type=float)
    parser.add_argument("--freeze-embeddings", default=args["freeze-embeddings"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--gpu-id", required=False, default=args["gpu-id"], type=int)
    parser.add_argument("--max-len", default=args["max-len"], type=int)
    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--mlp-hidden-dim", default=args["mlp-hidden-dim"], type=int)
    parser.add_argument("--env-optimizer", required=False, default=args["env-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--pol-optimizer", required=False, default=args["pol-optimizer"],
                        choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
    parser.add_argument("--ppo-updates", required=False, default=args["ppo-updates"], type=int)
    parser.add_argument("--entropy-weight", default=args["entropy-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)
    parser.add_argument("--env-lr", required=False, default=args["env-lr"], type=float)
    parser.add_argument("--pol-lr", required=False, default=args["pol-lr"], type=float)
    parser.add_argument("--epsilon", required=False, default=args["epsilon"], type=float)
    parser.add_argument("--model-dir", required=False, default=args["model-dir"], type=str)
    parser.add_argument("--trans-hidden-dim", required=False, default=args["trans-hidden-dim"], type=int)

    global_step = 0
    best_model_path = None
    best_val_accuracy = 0.
    args = parser.parse_args()
    with torch.cuda.device(args.gpu_id):
        main(args)
