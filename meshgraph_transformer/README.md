## Setup

Prepare environment, install dependencies:

    virtualenv --python=python3.6 "${ENV}"
    source ${ENV}/bin/activate
    pip install -r meshgraphnets/requirements.txt

Download a dataset:

    mkdir -p ${DATA}
    bash meshgraphnets/download_dataset.sh flag_simple ${DATA}

## Running the model

Train a model:

    python -m meshgraphnets.run_model --mode=train --model=cloth \
        --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple

Generate some trajectory rollouts:

    python -m meshgraphnets.run_model --mode=eval --model=cloth \
        --checkpoint_dir=${DATA}/chk --dataset_dir=${DATA}/flag_simple \
        --rollout_path=${DATA}/rollout_flag.pkl

Plot a trajectory:

    python -m meshgraphnets.plot_cloth --rollout_path=${DATA}/rollout_flag.pkl

The instructions above train a model for the `flag_simple` domain; for
the `cylinder_flow` dataset, use `--model=cfd` and the `plot_cfd` script.

## Datasets

Datasets can be downloaded using the script `download_dataset.sh`. They contain
a metadata file describing the available fields and their shape, and tfrecord
datasets for train, valid and test splits.
Dataset names match the naming in the paper.
The following datasets are available:

    airfoil
    cylinder_flow
    deforming_plate
    flag_minimal
    flag_simple
    flag_dynamic
    flag_dynamic_sizing
    sphere_simple
    sphere_dynamic
    sphere_dynamic_sizing

`flag_minimal` is a truncated version of flag_simple, and is only used for
integration tests. `flag_dynamic_sizing` and `sphere_dynamic_sizing` can be
used to learn the sizing field. These datasets have the same structure as
the other datasets, but contain the meshes in their state before remeshing,
and define a matching `sizing_field` target for each mesh.
