#!/usr/bin/env bash

######################################################
##########       INPUTS TO SET       ##########
######################################################

declare -A locations=(
    [20a]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2605_20260821-0949/test/__ALL__'
    [20b]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2606_20260821-0949/test/__ALL__'
    [24api0]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2607_20260821-0950/test/__ALL__'
    [20gpi0]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2608_20260821-0950/test/__ALL__'
    [24aeta]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2609_20260821-0950/test/__ALL__'
    [20geta]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_MC_AOD/2610_20260821-0951/test/__ALL__'
    [data]='http://alitrain.cern.ch/train-workdir/PWGGA/GA_PbPb_AOD/1298_20260821-0952/test/__ALL__'
)

declare -A output_dirs=(
    [20a]='20a'
    [20b]='20b'
    [24api0]='24api0'
    [20gpi0]='20gpi0'
    [24aeta]='24aeta'
    [20geta]='20geta'
    [data]='data'
)

# The train outputs contain only the 3X1X GCo files listed here.
declare -A gco_numbers=(
    [20a]='3310 3311 3312 3313'
    [20b]='3310 3311 3312 3313'
    [24api0]='3410 3411 3412 3413'
    [20gpi0]='3410 3411 3412 3413'
    [24aeta]='3510 3511 3512 3513'
    [20geta]='3510 3511 3512 3513'
    [data]='3210 3211 3212 3213'
)

dataset_names=(20a 20b 24api0 20gpi0 24aeta 20geta data)
NODE=10
TODAY=$(date +%d-%m-%Y)
OUTPUT=${OUTPUT:-/misc/alidata150/alice_u/nath/alitrain/${TODAY}}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLOT_SCRIPT=${SCRIPT_DIR}/test_train_plot.py
ROOT_HELPER=${SCRIPT_DIR}/../AfterburnerQA/run_root.helper
PLOT_ONLY=${PLOT_ONLY:-0}

run_plots() {
    if [[ ! -f "${PLOT_SCRIPT}" ]]; then
        printf 'ERROR: plotting script not found: %s\n' "${PLOT_SCRIPT}" >&2
        return 1
    fi
    if [[ ! -x "${ROOT_HELPER}" ]]; then
        printf 'ERROR: ROOT environment helper not found: %s\n' "${ROOT_HELPER}" >&2
        return 1
    fi

    printf 'Running Photon QA plots for %s\n' "${OUTPUT}"
    if [[ -n "${ALIENVLVL:-}" ]]; then
        MPLCONFIGDIR=/tmp/ga_train_plots_mpl \
            python3 "${PLOT_SCRIPT}" --loc "${OUTPUT}"
    else
        MPLCONFIGDIR=/tmp/ga_train_plots_mpl \
            "${ROOT_HELPER}" python "${PLOT_SCRIPT}" --loc "${OUTPUT}"
    fi
}

if [[ "${PLOT_ONLY}" == 1 ]]; then
    if [[ ! -d "${OUTPUT}" ]]; then
        printf 'ERROR: plot-only directory does not exist: %s\n' "${OUTPUT}" >&2
        exit 1
    fi
    run_plots
    exit $?
fi

if [[ -e "${OUTPUT}" ]]; then
    if [[ ! -d "${OUTPUT}" ]] || [[ -n "$(find "${OUTPUT}" -mindepth 1 -print -quit)" ]]; then
        printf 'WARNING: %s already contains files. Nothing will be downloaded.\n' "${OUTPUT}" >&2
        exit 1
    fi
fi

######################################################
##########       Download       ##########
######################################################

download_file() {
    local name="$1"
    local number="$2"
    local output_dir="${OUTPUT}/${output_dirs[${name}]}"

    mkdir -p "${output_dir}" || return 1
    curl --fail --location --retry 3 --silent --show-error \
        --output "${output_dir}/GCo_${number}.root" \
        "${locations[${name}]}/GCo_${number}.root"
}

wait_for_jobs() {
    local pid

    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    pids=()
}

queue_download() {
    local name="$1"
    local number="$2"

    download_file "${name}" "${number}" &
    pids+=("$!")
    if (( ${#pids[@]} >= NODE )); then
        wait_for_jobs
    fi
}

failed=0
pids=()
for name in "${dataset_names[@]}"; do
    for number in ${gco_numbers[${name}]}; do
        queue_download "${name}" "${number}"
    done
done
wait_for_jobs

if (( failed )); then
    printf 'ERROR: one or more ROOT files could not be downloaded.\n' >&2
    exit 1
fi

if ! run_plots; then
    printf 'ERROR: downloads completed but plotting failed.\n' >&2
    exit 1
fi
