# -*- coding: utf-8 -*-
"""Entry module.
"""
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
import click

from misc import get_mc_scheduler_fn, get_velocity_fn, get_time_grid, prior_sampling, mkdir, pbar, my_logging
from example import get_example
from fflow import fflow_sampler
from mcmc import mcmc_sampler
from network import ResNet


@click.group()
def main():
    pass


@main.command()
@click.option("--tag", "-t", default=None, help="suffix for this run", type=str)
@click.option("--workdir", "-d", default="assets", help="working directory for output", type=str, required=True)
@click.option("--example_id", "-e", help="example distributions", type=int, required=True)
@click.option("--nsample", "-n", help="number of samples", type=int, required=True)
@click.option("--parallel", "-p", is_flag=True, help="use joblib for multithreading on CPU tasks", required=True)
@click.option("--n_job", "-j", default=-1, help="number of threads for joblib, default to max", type=int, required=True)
@click.option("--uniform", "-u", is_flag=True, help="type of temporal grid for ODE", required=True)
@click.option("--time_span", nargs=2, default=[0.0, 1.0], help="span of temporal grid for ODE", type=float, required=True)
@click.option("--time_steps", "-K", default=100, help="size of temporal grids for ODE", type=int, required=True)
@click.option("--velocity_form", "-v", help="form of Föllmer velocity", type=click.Choice(["closed", "mc"]), required=True)
@click.option("--n_mc", "-M", help="number of Monte Carlo simulation", type=int)
@click.option("--mc_scheduler", help="sheduler of number of Monte Carlo samples", type=click.Choice(["static", "dynamic"]))
@click.option("--mu", default=0., help="mean of preconditioner", type=float, required=True)
@click.option("--sigma", default=1., help="std of preconditioner", type=float, required=True)
def follmer_flow(tag, workdir, example_id, nsample, parallel, n_job, uniform, time_span, time_steps, velocity_form, n_mc, mc_scheduler, mu, sigma):
    mkdir(workdir)
    handler = logging.FileHandler(os.path.join(workdir, "log.txt"), mode="a")
    desc = f"ex{example_id}-fflow-{velocity_form}-n{nsample//1000:d}k-K{time_steps}-mu{mu:.1f}-sigma{sigma:.1f}"
    desc += f"-M{n_mc}-{mc_scheduler}" if velocity_form == "mc" else ""
    desc += "-uniform" if uniform else "-ununiform"
    desc += f"-{tag}" if tag is not None else ""
    with my_logging(desc, handler):
        target = get_example(example_id)
        dimension = target.dimension
        mc_scheduler_fn = get_mc_scheduler_fn(mc_scheduler)
        velocity_fn = partial(get_velocity_fn(
            target, velocity_form), preconditioner=dict(offset=mu, deviation=sigma))
        time_grid = get_time_grid(time_span, time_steps, uniform)
        x0 = prior_sampling(dimension, mu, sigma, nsample)
        x1 = fflow_sampler(velocity_fn, x0, time_grid,
                           parallel, n_job, n_mc, mc_scheduler_fn)
        np.savez(os.path.join(workdir, f"{desc}.npz"), x0=x0, x1=x1)


@main.command()
@click.option("--tag", "-t", default=None, help="suffix for this run", type=str)
@click.option("--workdir", "-d", default="assets", help="working directory for output", type=str, required=True)
@click.option("--example_id", "-e", help="example distributions", type=int, required=True)
@click.option("--nsample", "-n", help="number of samples", type=int, required=True)
@click.option("--parallel", "-p", is_flag=True, help="use joblib for multithreading on CPU tasks", required=True)
@click.option("--n_job", "-j", default=-1, help="number of threads for joblib, default to max", type=int, required=True)
@click.option("--method", "-m", help="name of MCMC sampler", type=click.Choice(["RWMH", "ULA", "tULA", "tULAc", "MALA", "tMALA", "tMALAc", "LM", "tLM", "tLMc"]), required=True)
@click.option("--n_burnin", "-b", default=10000, help="number of burn-in samples", type=int, required=True)
@click.option("--n_chain", "-k", default=50, help="number of chains", type=int, required=True)
@click.option("--step_size", "-s", default=0.2, help="step size", type=float, required=True)
@click.option("--hybrid", is_flag=True, help="use predictor-corrector hybrid method", required=True)
@click.option("--prediction_path", help="path to follmer samples npz file", type=str)
def mcmc(tag, workdir, example_id, nsample, parallel, n_job, method, n_burnin, n_chain, step_size, hybrid, prediction_path):
    mkdir(workdir)
    desc = f"ex{example_id}-{method}-chain{n_chain}-n{nsample//1000:d}k-burn{n_burnin//1000:d}k-step{step_size:.1f}"
    desc += "-hybrid" if hybrid else ""
    desc += f"-{tag}" if tag is not None else ""
    handler = logging.FileHandler(os.path.join(workdir, "log.txt"), mode='a')
    with my_logging(desc, handler):
        target = get_example(example_id)
        dimension = target.dimension
        mcmc = mcmc_sampler(target)
        if hybrid:
            x0 = np.load(prediction_path)["x1"][:n_chain, :]
        else:
            x0 = np.random.randn(n_chain, dimension)
        x1 = mcmc.sample(x0, method, nsample, n_burnin,
                         step_size, parallel, n_job)
        np.savez(os.path.join(workdir, f"{desc}.npz"), x1=x1)


@click.group()
def neural_follmer_flow():
    pass


@neural_follmer_flow.command()
@click.option("--example_id", "-e", help="example distributions", type=int, required=True)
@click.option("--device", default="auto", help="torch devices", type=click.Choice(["cuda", "cpu"]))
@click.option("--tag", "-t", default=None, help="suffix for this run", type=str)
@click.option("--workdir", "-d", default="assets", help="working directory for output", type=str, required=True)
@click.option("--dump_freq", default=500, help="dump train states frequency (epoch)", type=int, required=True)
@click.option("--lr", default=1e-4, help="initial learning rate", type=float, required=True)
@click.option("--bsz", default=200, help="batch size of dataloader", type=int, required=True)
@click.option("--n_epoch", default=5000, help="number of total train epoches", type=int, required=True)
@click.option("--trainset", help="path to train set (npz)", type=str, required=True)
def train(example_id, device, tag, workdir, dump_freq, lr, bsz, n_epoch, trainset):
    mkdir(workdir)
    device = torch.device(device)
    handler = logging.FileHandler(os.path.join(workdir, "log.txt"), mode='a')
    desc = f"ex{example_id}-fflow-neural"
    desc += f"-{tag}" if tag is not None else ""
    with my_logging(desc, handler):
        logger = logging.getLogger()
        tmp = np.load(trainset)
        x0 = torch.from_numpy(tmp["x0"].astype(np.float32))
        x1 = torch.from_numpy(tmp["x1"].astype(np.float32))
        dimension = x0.shape[1]
        del tmp
        dataset = torch.utils.data.TensorDataset(x0, x1)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bsz, shuffle=True)
        model = ResNet(dimension, dimension).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        for epoch in pbar(n_epoch, update_freq=n_epoch//100):
            for step, (x0, x1) in enumerate(dataloader):
                x0, x1 = x0.to(device), x1.to(device)
                pred = model(x0)
                loss = (pred - x1).square().sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.info(f"epoch [{epoch+1}/{n_epoch}], step [{step+1}/{len(dataloader)}], loss {loss.item():.4e}")
            if epoch != 0 and (epoch+1) % 500 == 0:
                scheduler.step()
            if epoch != 0 and (epoch+1) % dump_freq == 0 or epoch + 1 == n_epoch:
                torch.save(model.state_dict(), os.path.join(workdir, f"{desc}-epoch{epoch+1}.pth"))


@neural_follmer_flow.command()
@click.option("--example_id", "-e", help="example distributions", type=int, required=True)
@click.option("--tag", "-t", default=None, help="suffix for this run", type=str)
@click.option("--workdir", "-d", default="assets", help="working directory for output", type=str, required=True)
@click.option("--ckpt_path", "-c", help="path of checkpoint", type=str, required=True)
@click.option("--bsz", "-b", default=200, help="batch size of dataloader", type=int, required=True)
@click.option("--nsample", "-n", help="number of samples", type=int, required=True)
@click.option("--mu", default=0., help="mean of preconditioner", type=float, required=True)
@click.option("--sigma", default=1., help="std of preconditioner", type=float, required=True)
def eval(example_id, tag, workdir, ckpt_path, bsz, nsample, mu, sigma):
    mkdir(workdir)
    handler = logging.FileHandler(os.path.join(workdir, "log.txt"), mode='a')
    desc = f"ex{example_id}-fflow-neural-n{nsample//1000:d}k"
    desc += f"-{tag}" if tag is not None else ""
    with my_logging(desc, handler):
        target = get_example(example_id)
        dimension = target.dimension
        model = ResNet(dimension, dimension)
        model.load_state_dict(torch.load(
            ckpt_path, map_location=torch.device("cpu")))
        x0 = torch.from_numpy(prior_sampling(dimension, mu, sigma, nsample).astype(np.float32))
        x1 = torch.empty_like(x0)
        dataset = torch.utils.data.TensorDataset(x0)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=bsz, shuffle=False)
        with torch.no_grad():
            for step, (x0_i, ) in enumerate(dataloader):
                x1[step*bsz: (step+1)*bsz, ...] = model(x0_i)
        np.savez(os.path.join(workdir, f"{desc}.npz"),
                 x0=x0.cpu().numpy(), x1=x1.cpu().numpy())


main.add_command(neural_follmer_flow)


if __name__ == "__main__":
    main()
