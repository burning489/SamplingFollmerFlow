#!/bin/bash

source /opt/data/private/miniconda3/bin/activate fflow

1d() {
    n_burnin=10000
    n_chain=50
    step_size=0.2
    K=100
    nsample=10000

    for id in {1..3}
    do
        python main.py follmer-flow -e ${id} -n ${nsample} -u -K ${K} -v closed -p
        for method in "RWMH" "tULA" "tMALA"
        do
            python main.py mcmc -e ${id} -n ${nsample} -m ${method} --n_burnin ${n_burnin} --n_chain ${n_chain} --step_size ${step_size} -p
        done
    done

    tags=()
    for id in {1..3}
    do
        tags+=("ex${id}-fflow-closed-n10k-K100-mu0.0-sigma1.0-uniform")
        for method in "RWMH" "tULA" "tMALA"
        do
            tags+=("ex${id}-${method}-chain50-n10k-burn10k-step0.2")
        done
    done
    python plot1d.py "${tags[@]}"
}

2d() {
    n_burnin=10000
    n_chain=50
    step_size=0.2
    K=100
    nsample=20000

    for id in {4..10}
    do
        python main.py follmer-flow -e ${id} -n ${nsample} -u -K ${K} -v closed -p
        for method in "RWMH" "tULA" "tMALA"
        do
            python main.py mcmc -e ${id} -n ${nsample} -m ${method} --n_burnin ${n_burnin} --n_chain ${n_chain} --step_size ${step_size} -p
        done
    done
}

2dmc() {
    nsample=20000
    M=1000
    K=100
    ids=(4 5 6 7 8 9 10)
    sigmas=(2.0 4.0 1.0 1.7 1.4 1.8 1.0)

    for ((i=0; i<${#ids[@]}; i++))
    do
        python main.py follmer-flow -e ${ids[i]} -n ${nsample} -u -K ${K} -M ${M} -v mc --mc_scheduler static --sigma ${sigmas[i]} -p
    done
}

precondition() {
    nsample=20000
    M=1000
    K=100
    id=7

    for sigma in 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0
    do
        python main.py follmer-flow -e ${id} -n ${nsample} -u -K ${K} -M ${M} -v mc --mc_scheduler static --sigma ${sigma} -p
    done
}

hybrid() {
    nsample=20000
    id=7
    n_chain=50
    step=0.2
    n_burnin=10000
    sigma=2.0
    T=10

    python main.py follmer-flow -e ${id} -n ${n_chain} -u -K ${T} -M ${T} -v mc --mc_scheduler static --sigma ${sigma} -p
    for method in "RWMH" "tULA" "tMALA"
    do
        python main.py mcmc -e ${id} -n ${nsample} -m ${method} --n_burnin ${n_burnin} --n_chain ${n_chain} --step_size ${step} -p --hybrid --prediction_path assets/ex${id}-fflow-mc-n0k-K${T}-mu0.0-sigma${sigma}-M${T}-static-uniform.npz -t T${T}
    done
}

nd() {
    nsample=20000
    n_burnin=10000
    n_chain=50
    step_size=0.2
    K=200
    time_span="0. 5."

    for id in {11..20}
    do
        python main.py follmer-flow -e ${id} -n ${nsample} -K ${K} -M $(((id-10)*200)) -v mc --mc_scheduler static -p --time_span ${time_span}
        python main.py follmer-flow -e ${id} -n ${nsample} -K ${K} -v closed -p --time_span ${time_span}
        # for method in "RWMH" "tULA" "tMALA"
        # do
        #     python main.py mcmc -e ${id} -n ${nsample} -m ${method} --n_burnin ${n_burnin} --n_chain ${n_chain} --step_size ${step_size} -p
        # done
    done
}

train() {
    ntrain=100000
    K=100
    for id in {4..10}
    do
        python main.py follmer-flow -e ${id} -n ${ntrain} -u -K ${K} -v closed -p
        python main.py neural-follmer-flow train -e ${id} --device cuda --trainset assets/ex${id}-fflow-closed-n100k-K100-mu0.0-sigma1.0-uniform.npz -t closed
    done
}

eval() {
    nsample=20000
    bsz=200
    for id in {4..10}
    do
        python main.py neural-follmer-flow eval -e ${id} -n ${nsample} --bsz ${bsz} --ckpt_path assets/ex${id}-fflow-neural-closed-epoch5000.pth
    done
}


if [ $# -eq 0 ]; then
    echo "No arguments provided."
    exit 1
fi

if [ "$1" == "1d" ]; then
   1d
elif [ "$1" == "2d" ]; then
   2d
elif [ "$1" == "2dmc" ]; then
   2dmc
elif [ "$1" == "precondition" ]; then
   precondition
elif [ "$1" == "hybrid" ]; then
   hybrid
elif [ "$1" == "nd" ]; then
   nd
elif [ "$1" == "train" ]; then
   train
elif [ "$1" == "eval" ]; then
   eval
else
    echo "Invalid argument."
    exit 1
fi
    