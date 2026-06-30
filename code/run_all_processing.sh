#!/bin/bash
# Keep the screen window open after the script exits (success or error) so logs can be reviewed.
trap 'echo ""; echo "=== Script exited (code $?). Type '\''exit'\'' to close screen. ==="; bash' EXIT

# --------------------------
# User parameters
# --------------------------
# Default values
PATH_DATA="$PATH_DATA" #Defaults from environment
PATH_CODE="$PATH_CODE" #Defaults from environment
PYTHON="${PYTHON:-python}"  # override with e.g. PYTHON=/path/to/env/bin/python
IDs=() # empty  → process all participants
TASKS=() # empty → process all tasks
RUN_PREPROSS=false
RUN_DENOISING=false
RUN_FIRSTLEVEL=false
RUN_SECONDLEVEL=false
RUN_FIGURES=false
RUN_COMPARE=false
REDO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --path-data) PATH_DATA="$2"; shift 2 ;;
        --path-code) PATH_CODE="$2"; shift 2 ;;
        --ids) shift; while [[ $# -gt 0 && "$1" != --* ]]; do IDs+=("$1"); shift; done ;;
        --tasks) shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done ;;
        --preprocess) RUN_PREPROSS=true; shift;;
        --denoising) RUN_DENOISING=true; shift;;
        --firstlevel) RUN_FIRSTLEVEL=true; shift;;
        --secondlevel) RUN_SECONDLEVEL=true; shift;;
        --compare) RUN_COMPARE=true; shift;;
        --redo) REDO=true; shift;;
      *) echo "Unknown argument $1"; exit 1 ;;
    esac
done

if [ "${RUN_PREPROSS}" = false ] && \
   [ "${RUN_DENOISING}" = false ] && \
   [ "${RUN_FIRSTLEVEL}" = false ] && \
   [ "${RUN_SECONDLEVEL}" = false ] && \
   [ "${RUN_FIGURES}" = false ] && \
   [ "${RUN_COMPARE}" = false ]; then
    echo "ERROR: No processing step selected."
    echo "Use --preprocess, --denoising, --firstlevel, --secondlevel and/or --compare"
    exit 1
fi

# Show participants
[ ${#IDs[@]} -eq 0 ] && echo "No specific IDs provided: processing all participants" \
                     || echo "Processing participants: ${IDs[*]}"
[ ${#IDs[@]} -eq 0 ] && IDs=("") # If no IDs were provided, set to empty string


if [ ${#TASKS[@]} -eq 0 ]; then
    echo "No task specified: processing all tasks"
    TASKS_ARG=()        # do not pass --task
else
    echo "Processing tasks: ${TASKS[*]}"
    TASKS_ARG=(--tasks "${TASKS[@]}")
fi

# --------------------------
# Prepare log folder
# --------------------------
cd "${PATH_CODE}" || { echo "ERROR: could not cd to PATH_CODE='${PATH_CODE}'. Pass --path-code /path/to/repo"; exit 1; }
mkdir -p log
cd log || { echo "ERROR: could not cd into log/"; exit 1; }

timestamp=$(date +"%Y%m%d_%H%M%S")

# --------------------------
# Run preprocessing
# --------------------------

run_step() {
    local label="$1"; local logfile="$2"; shift 2
    echo ""
    echo "=== Starting ${label} === (log: log/${logfile})"
    # Run directly (no nohup/&): screen handles disconnection.
    # tee shows output live in screen AND saves to the log file.
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    if [ "${rc}" -ne 0 ]; then
        echo ""
        echo "ERROR: ${label} failed (exit code ${rc}). Stopping pipeline."
        exit "${rc}"
    fi
    echo "=== ${label} done ==="
}

if [ "${RUN_PREPROSS}" = true ]; then
    run_step "Preprocessing" "preprocessing_${timestamp}.txt" \
        ${PYTHON} -u ../code/preprocessing_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}"
fi

# --------------------------
# Run denoising
# --------------------------

if [ "${RUN_DENOISING}" = true ]; then
    run_step "Denoising" "denoising_${timestamp}.txt" \
        ${PYTHON} -u ../code/denoising_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}"
fi

# --------------------------
# Run first level analysis
# --------------------------

if [ "${RUN_FIRSTLEVEL}" = true ]; then
    run_step "First level analysis" "firstlevel_${timestamp}.txt" \
        ${PYTHON} -u ../code/firstlevel_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}"
fi

# --------------------------
# Run second level analysis
# --------------------------
if [ "${RUN_SECONDLEVEL}" = true ]; then
    run_step "Second level analysis" "secondlevel_${timestamp}.txt" \
        ${PYTHON} -u ../code/secondlevel_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}"
fi

# --------------------------
# Run quantitative comparison (1mm vs 3mm)
# --------------------------
if [ "${RUN_COMPARE}" = true ]; then
    run_step "1mm vs 3mm comparison" "compare_${timestamp}.txt" \
        ${PYTHON} -u ../code/compare_workflow.py --path-data "${PATH_DATA}" --ids "${IDs[@]}" "${TASKS_ARG[@]}" --redo "${REDO}"
fi
