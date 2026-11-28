#!/bin/bash
# train.sh -g <GPUs> -d <config folder> -c <config> -n <run name> [-r auto] [-o "k=v ..."]
# runs configs/<config folder>/<config>.py in exp/<config folder>/<run name> (Pointcept's
# flags; -e changes the experiment root "exp", -p the python, -w the start weights, -m
# the number of machines).

cd $(dirname "$0") || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

DATASET=hybrid
CONFIG="None"
EXP_NAME=debug
EXP_DIR=exp
WEIGHT="None"
RESUME=false
OPTIONS=""
NUM_GPUS=None
NUM_MACHINES=1

MACHINE_RANK=0
DIST_URL="auto"


while getopts ":p:d:e:c:n:w:g:m:r:o:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG  # the config folder: configs/<DATASET>/, exp/<DATASET>/
      ;;
    e)
      EXP_DIR=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPUS=$OPTARG
      ;;
    m)
      NUM_MACHINES=$OPTARG
      ;;
    o)
      OPTIONS=$OPTARG  # extra config overrides, e.g. "hooks.6.save_freq_iter=1000"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG needs an argument" >&2
      exit 1
      ;;
  esac
done

if [ "${NUM_GPUS}" = 'None' ]
then
  NUM_GPUS=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Config folder: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPUS"
echo "Machine Num: $NUM_MACHINES"

if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
  MACHINE_RANK=${SLURM_NODEID:-0}
fi

echo "Dist URL: $DIST_URL"

EXP_DIR=${EXP_DIR}/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

# Handle auto-resume: check if checkpoint exists
if [ "${RESUME}" = "auto" ]; then
  if [ -f "$MODEL_DIR/model_last.pth" ]; then
    echo "Auto-resume: Found existing checkpoint at $MODEL_DIR/model_last.pth"
    RESUME=true
  else
    echo "Auto-resume: No checkpoint found, starting fresh"
    RESUME=false
  fi
fi

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
  # a checkpoint= option names the fresh run's start weights (config.py keeps them and
  # CheckpointLoader.resume reloads the skipped encoder keys from there); passed after
  # checkpoint=$WEIGHT it would replace model_last.pth, so a resume drops it
  case " $OPTIONS " in
    *" checkpoint="*)
      read -ra OPTION_WORDS <<< "$OPTIONS"
      OPTIONS=""
      for opt in "${OPTION_WORDS[@]}"; do
        case $opt in
          checkpoint=*) echo "Resume: option $opt dropped (resuming from $WEIGHT)" ;;
          *) OPTIONS="${OPTIONS:+$OPTIONS }$opt" ;;
        esac
      done
      ;;
  esac
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  rsync -a ldpt "$CODE_DIR"/
  rsync -a --include='*.py' --include='*.sh' --exclude='*' ./ "$CODE_DIR"/
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
ulimit -n 65535
if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus ${NUM_GPUS} \
    --num-machines ${NUM_MACHINES} \
    --machine-rank ${MACHINE_RANK} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" $OPTIONS
else
    $PYTHON "$CODE_DIR"/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus ${NUM_GPUS} \
    --num-machines ${NUM_MACHINES} \
    --machine-rank ${MACHINE_RANK} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" resume="$RESUME" checkpoint="$WEIGHT" $OPTIONS
fi