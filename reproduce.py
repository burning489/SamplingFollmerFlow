import re
import numpy as np
import pandas as pd
import ot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans

from example import get_example
from misc import sample_gm1d_truth, sample_gmnd_truth, SeabornFig2Grid, plot2d

from metric import mmd, w2
from example import get_example


@click.group()
def main():
    pass


@main.command()
def table1():
    np.random.seed(42)
    n_correction = 5000
    nsample = 10000
    tags = ["fflow-closed-n10k-K100-mu0.0-sigma1.0-uniform",] + [f"{method}-chain50-n10k-burn10k-step0.2" for method in ["RWMH", "tULA", "tMALA"]]
    sampler_extractor = lambda tag: re.search(r"^([^-]+)", tag).group(1)
    rows = []
    for id in range(1, 4):
        target = get_example(id)
        truth_correction = sample_gm1d_truth(target, n_correction)
        truth = sample_gm1d_truth(target, nsample)
        mmd_bench = mmd(truth_correction, truth)
        w2_bench = ot.emd2_1d(truth_correction, truth, metric="euclidean")
        for tag in tags:
            x1 = np.load(f"./assets/ex{id}-{tag}.npz")["x1"]
            w2_adj = ot.emd2_1d(x1, truth_correction, metric="euclidean") - w2_bench
            mmd_adj = mmd(x1, truth_correction) - mmd_bench
            rows.append({
                "example": id,
                "sampler": sampler_extractor(tag),
                "adj. W": w2_adj,
                "adj. MMD": mmd_adj,
            })
    df = pd.DataFrame(rows)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df)
    df.to_csv(f"assets/1d.csv")

@main.command()
def table2():
    np.random.seed(42)
    n_correction = 5000
    nsample = 20000
    numThreads=64
    tags = ["fflow-closed-n20k-K100-mu0.0-sigma1.0-uniform", "fflow-neural-n20k", ] + [f"{method}-chain50-n20k-burn10k-step0.2" for method in ["RWMH", "tULA", "tMALA"]]
    sigmas = {4: 2.0, 5:4.0, 6:1.0, 7:1.7, 8:1.4, 9:1.8, 10: 1.0}
    sampler_extractor = lambda tag: re.search(r'(.*?)-n', tag).group(1)

    rows = []
    for example_id in range(4, 11):
        target = get_example(example_id)
        truth_correction = sample_gmnd_truth(target, n_correction)
        truth = sample_gmnd_truth(target, nsample)
        mmd_bench = mmd(truth_correction, truth)
        w2_bench = w2(truth_correction, truth, numThreads=numThreads)
        for tag in tags + [f"fflow-mc-n20k-K100-mu0.0-sigma{sigmas[example_id]:.1f}-M1000-static-uniform"]:
            print(example_id, tag)
            x1 = np.load(f"./assets/ex{example_id}-{tag}.npz")["x1"]
            x1 = x1[~(np.isnan(x1) | np.isinf(np.abs(x1))).any(axis=1)]
            w2_adj = w2(x1, truth_correction, numThreads=numThreads) - w2_bench
            mmd_adj = mmd(x1, truth_correction) - mmd_bench
            rows.append({
                "example": example_id,
                "sampler": sampler_extractor(tag),
                "adj. W": w2_adj,
                "adj. MMD": mmd_adj,
            })
    df = pd.DataFrame(rows)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df)
    df.to_csv(f"assets/2d.csv")

    # samplers = ["fflow-closed", "fflow", "fflow-mc", "RWMH-chain50", "tULA-chain50", "tMALA-chain50"]
    # for eid in range(4, 11):
    #     print(f"example {eid} & ", end="")
    #     for sampler in samplers:
    #         w2 = float(data[(data["example"] == eid) & (data["sampler"] == sampler)]["adj. W"])
    #         mmd = float(data[(data["example"] == eid) & (data["sampler"] == sampler)]["adj. MMD"])
    #         print(f"{w2:.3f} & {mmd:.3f} &", end=" ")
    #     print()

@main.command()
def fig1():
    dimension = 2
    nsample = 10000
    example_id = 4
    target = get_example(example_id)
    velocity = target.velocity_closed
    T = 100
    h = 1/T
    mean = target.mean_array
    cov = target.cov_array
    kap = mean.shape[0]
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i) for i in np.linspace(0, 1, kap)]
    x = np.empty((T, nsample, dimension))
    x[0, ...] = np.random.randn(nsample, dimension)

    for i in range(T-1):
        t = i*h
        x[i+1, ...] = x[i, ...] + h * velocity(x[i, ...], t)

    k_means = KMeans(n_clusters=kap)
    k_means.fit(x[-1, ...])
    label_pred = k_means.predict(x[-1, ...])
    labels = np.array([label_pred[i] for i in range(nsample)])

    x_traj = np.empty((T, kap, dimension))
    for i in range(kap):
        x_traj[-1, i, ...] = k_means.cluster_centers_[i]
    for i in range(1, T):
        t = (T-i-1)*h
        x_traj[T-i-1, ...] = x_traj[T-i, ...] - h * velocity(x_traj[T-i, ...], t)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    interval = list(range(0, T, 20))
    for i in interval:
        for j in range(kap):
            xi = x[i, labels == j, 0]
            yi = x[i, labels == j, 1]
            ax.scatter(xi, yi, zs=i, zdir='z', c=colors[j], s=0.3, marker="8")
    ax.set_box_aspect(aspect=(1, 1, 1.5))
    ax.view_init(elev=-160, azim=30, roll=90)
    for j in range(kap):
        xs = [x_traj[i, j, 0] for i in interval]
        ys = [x_traj[i, j, 1] for i in interval]
        zs=[i for i in interval]
        ax.plot(xs, ys, zs, c=colors[j], linewidth=2)

    plt.axis('off')
    plt.savefig("./assets/evolution.png", bbox_inches="tight", dpi=300, transparent=True)


@main.command()
def fig2():
    plt.style.use("ggplot")
    linewidth = 1
    def target_density(t, target):
        ret = np.zeros_like(t)
        for i in range(len(target.weights)):
            ret += target.weights[i]/(np.sqrt(2*np.pi*target.var_array[i])) * \
                np.exp(-(t-target.mean_array[i])**2/(2*target.var_array[i]))
        return ret

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8, 4))
    tags = ["fflow-closed-n10k-K100-mu0.0-sigma1.0-uniform"] + [f"{method}-chain50-n10k-burn10k-step0.2" for method in ["RWMH", "tULA", "tMALA"]]
    idx = 0
    for example_id in range(1, 4):
        for tag in tags:
            ax = axes.flat[idx]
            idx += 1
            target = get_example(example_id)
            mean_array = target.mean_array
            t = np.arange(mean_array[0]-2, mean_array[-1]+4, 0.01)
            truth = target_density(t, target)
            x1 = np.load(f"assets/ex{example_id}-{tag}.npz")["x1"]
            sns.kdeplot(x=x1.ravel(), ax=ax, bw_adjust=0.25, label=tag, linewidth=linewidth)
            ax.fill(t, truth, alpha=0.25, color="grey")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if idx % 4 != 1:
                ax.set_yticks([])
            ax.set_xlim(list(map(lambda x: x*1.5, ax.get_xlim())))
            ax.grid(alpha=0.4, linestyle="-.")
    axes.flat[4].set_ylabel("Density", fontsize=16)
    for i in range(3):
        ax = axes.flat[4*i+3].twinx()
        ax.set_yticks([])
        ax.set_ylabel(f"example {i+1}", rotation=270, labelpad=12, fontsize=12)
    titles = ["Föllmer Flow", "MH", "ULA", "MALA"]
    for i in range(4):
        axes.flat[i].set_title(titles[i], fontsize=12)
    plt.tight_layout()
    fig.savefig(f"assets/1d.pdf", bbox_inches="tight", pad_inches=0., dpi=300)

@main.command()
def fig3():
    plt.style.use("default")
    sns.color_palette("Set2")
    tags = ["fflow-closed-n20k-K100-mu0.0-sigma1.0-uniform", "fflow-neural-n20k", ] + [f"{method}-chain50-n20k-burn10k-step0.2" for method in ["RWMH", "tULA", "tMALA"]]
    sigmas = {4: 2.0, 5:4.0, 6:1.0, 7:1.7, 8:1.4, 9:1.8, 10: 1.0}
    examples = list(range(4, 11))
    nrow = len(examples)
    ncol = 6
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(nrow, ncol)
    idx = 0
    for example_id in examples:
        target = get_example(example_id)
        lim = np.max(np.abs(target.mean_array)) + 5
        for tag in tags + [f"fflow-mc-n20k-K100-mu0.0-sigma{sigmas[example_id]:.1f}-M1000-static-uniform"]:
            x1 = np.load(f"assets/ex{example_id}-{tag}.npz")["x1"]
            g = plot2d(x1, lim=lim)
            SeabornFig2Grid(g, fig, gs[idx])
            idx+=1
    for i in range(nrow):
        fig.axes[18*i].set_ylabel(f"example {i+4}", fontsize=12)
    titles = ["Föllmer Flow", "Neural Föllmer Flow", "MH", "ULA", "MALA", "MC Föllmer Flow"]
    for i in range(ncol):
        fig.axes[1+3*i].set_title(titles[i], fontsize=12)
    gs.tight_layout(fig)
    plt.savefig(f"assets/2d.pdf", bbox_inches="tight", pad_inches=0.2, dpi=300, transparent=True)

@main.command()
def fig4():
    plt.style.use("default")
    sns.color_palette("Set2")
    example_id = 7
    sigmas = np.arange(1.0, 3.1, 0.4)
    target = get_example(example_id)
    lim = np.max(np.abs(target.mean_array)) + 5
    nrow = 1
    ncol = len(sigmas)
    fig = plt.figure(figsize=(8, 1.6))
    gs = gridspec.GridSpec(nrow, ncol)
    for i in range(ncol):
        tag = f"ex{example_id}-fflow-mc-n20k-K100-mu0.0-sigma{sigmas[i]:.1f}-M1000-static-uniform"
        x1 = np.load(f"assets/{tag}.npz")["x1"]
        g = plot2d(x1, lim=lim)
        SeabornFig2Grid(g, fig, gs[i])
        fig.axes[1+3*i].set_title(r"$\sigma=$"+f"{sigmas[i]:.1f}", fontsize=12)
    fig.axes[0].set_ylabel("example 7", fontsize=12)
    gs.tight_layout(fig)
    plt.savefig(f"assets/precondition.pdf", bbox_inches="tight", pad_inches=0.2, dpi=300, transparent=True)

@main.command()
def fig5():
    plt.style.use("default")
    sns.color_palette("Set2")
    example_id = 7
    target = get_example(example_id)
    lim = np.max(np.abs(target.mean_array)) + 5
            
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(2, 3)
    idx = 0
    for hybrid in [False, True]:
        for method in ["RWMH", "tULA", "tMALA"]:
            tag = f"./assets/ex{example_id}-{method}-chain50-n20k-burn10k-step0.2"
            tag += f"-hybrid-T10" if hybrid else ""
            x1 = np.load(f"{tag}.npz")["x1"]
            g = plot2d(x1, lim)
            SeabornFig2Grid(g, fig, gs[idx])
            idx += 1
    titles = ["MH", "ULA", "MALA"]
    for i in range(3):
        fig.axes[1+3*i].set_title(titles[i], fontsize=16)
    labels = ["original", "hybrid", "hybrid", "hybrid"]
    for i in range(2):
        fig.axes[9*i].set_ylabel(f"{labels[i]}", fontsize=16, rotation=0, labelpad=40)
    plt.tight_layout()
    fig.savefig(f"assets/hybrid.pdf", bbox_inches="tight", pad_inches=0.2, dpi=300, transparent=True)


@main.command()
def fig6():
    sns.color_palette("Set2")

    def get_moments(x):
        alpha = np.ones((x.shape[-1], ))
        alpha /= np.linalg.norm(alpha)
        h1 = np.mean(x @ alpha, axis=0)
        h2 = np.mean((x @ alpha) ** 2, axis=0)
        h3 = np.log(np.mean(np.exp(x @ alpha), axis=0))
        h4 = 10*np.mean(np.cos(5*x @ alpha), axis=0)
        return h1, h2, h3, h4

    def sampler_extractor(tag):
        if "RWMH" in tag:
            return "MH"
        elif "ULA" in tag:
            return "ULA"
        elif "MALA" in tag:
            return "MALA"
        elif "fflow-mc" in tag:
            return "MC Föllmer flow"
        elif "fflow-closed" in tag:
            return "Föllmer flow"
        
    rows = []
    tags = [f"fflow-closed-n20k-K200-mu0.0-sigma1.0-ununiform"] + [f"{method}-chain50-n20k-burn10k-step0.2" for method in ["RWMH", "tULA", "tMALA"]]
    for example_id in range(11, 21):
        target = get_example(example_id)
        dimension = example_id - 10
        for tag in tags + [f"fflow-mc-n20k-K200-mu0.0-sigma1.0-M{dimension*200:d}-static-ununiform"]:
            x1 = np.load(f"assets/ex{example_id}-{tag}.npz")["x1"]
            h1, h2, h3, h4 = get_moments(x1)
            rows.append({
                "dimension": dimension, "sampler": sampler_extractor(tag), "h1": h1, "h2": h2, "h3": h3, "h4": h4,
            })
        data = pd.DataFrame(rows)
        if dimension > 1:
            truth = sample_gmnd_truth(target, x1.shape[0])
        else:
            truth = sample_gm1d_truth(target, x1.shape[0])
        h1, h2, h3, h4 = get_moments(truth)
        rows.append({
            "dimension": dimension, "sampler": "truth", "h1": h1, "h2": h2, "h3": h3, "h4": h4,
        })

    ylabels = [
        r"$E[\alpha^\top X]$",
        r"$E[(\alpha^\top X)^2]$",
        r"$E[\exp(\alpha^\top X)]$",
        r"$E[10 \cos(5 \alpha^\top X)]$",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    for i in range(4):
        ax = axes.flat[i]
        sns.lineplot(data, x="dimension",
                    y=f"h{i+1}", hue="sampler", ax=ax, style="sampler", markers=True, linewidth=3, markersize=6)
        ax.grid(alpha=0.4, linestyle="-.")
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel(ylabels[i])
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='right', ncols=1, bbox_to_anchor=(0.85, 0.3, 0.5, 0.5), fontsize=16)
    fig.supxlabel('Dimension', fontsize=20)
    fig.supylabel('Moments', fontsize=20)
    plt.tight_layout()
    plt.savefig("assets/nd.pdf", bbox_inches="tight", pad_inches=0., dpi=300)

if __name__ == "__main__":
    main()