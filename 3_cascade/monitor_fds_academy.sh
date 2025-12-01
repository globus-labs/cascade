#!/usr/bin/env bash
# Monitors file descriptor usage for a Python process and its descendants.
# Logs per-PID FD counts over time, FD type breakdown (optional),
# and writes a summary with high-water marks and simple leak flags.

set -euo pipefail

#############################
#  CONFIG: EDIT THIS PART  #
#############################

# Put your full Python command here (no need to pass args on the CLI).
PY_CMD=(
  python
  run_cascade_academy.py
  --initial-structures ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  ../0_setup/final-geometries/packmol-CH4-in-H2O=32-seed=1-mace-medium.vasp
  --chunk-size 5
  --target-length 25
  --retrain-len 15
  --retrain-fraction 0.75
  --n-sample-frames 5
  --accept-rate .5
  --learner mace
  --calc mace
  --dyn-cls velocity-verlet
  --dt_fs 1.0
  --loginterval 1
)

# Sample every N seconds
INTERVAL=2

# Log of time series (for “time charts”)
LOGFILE="fd_tree_monitor.csv"

# Summary CSV at the end
SUMMARY_FILE="fd_summary.csv"

# Whether to classify FD types (files/sockets/pipes/anon).
# This is a bit heavier; set to 0 to disable.
FD_TYPES=1

# Simple heuristic: if a PID ever has >= this many FDs, flag as "suspected_leak"
LEAK_THRESHOLD=1024

#######################################
#  Helper: list all descendants of PID
#######################################

list_descendants() {
    local parent="$1"
    echo "$parent"
    local child
    for child in $(pgrep -P "$parent" 2>/dev/null || true); do
        list_descendants "$child"
    done
}

#########################################
#  Helper: count FD types for a given PID
#########################################
# Returns: "files sockets pipes anon"

fd_type_counts() {
    local pid="$1"
    local fdpath target
    local files=0 sockets=0 pipes=0 anon=0

    # Handle processes that exit mid-loop
    if [[ ! -d "/proc/$pid/fd" ]]; then
        echo "0 0 0 0"
        return
    fi

    for fdpath in /proc/"$pid"/fd/*; do
        [[ -e "$fdpath" ]] || continue
        target=$(readlink "$fdpath" 2>/dev/null || echo "")
        case "$target" in
            socket:*)        ((sockets++)) ;;
            pipe:*)          ((pipes++))   ;;
            anon_inode:*)    ((anon++))    ;;
            *)               ((files++))   ;;
        esac
    done

    echo "$files $sockets $pipes $anon"
}

#########################################
#  Start Python
#########################################

echo "Starting Python command:"
printf '  %q ' "${PY_CMD[@]}"
echo

"${PY_CMD[@]}" &
MAIN_PID=$!
echo "Main Python PID = $MAIN_PID"

# Kill Python if user hits Ctrl-C
trap "echo 'CTRL-C: killing main PID $MAIN_PID'; kill $MAIN_PID 2>/dev/null || true; exit 0" INT

# Require Bash 4+ for associative arrays
if ! declare -A __test 2>/dev/null; then
    echo "Error: this script requires Bash with associative array support (bash 4+)." >&2
    kill "$MAIN_PID" 2>/dev/null || true
    exit 1
fi

#########################################
#  State: per-PID tracking
#########################################

declare -A last_fd
declare -A max_fd
declare -A first_ts
declare -A last_ts
declare -A proc_name_map
declare -A alive

echo "Logging time series to: $LOGFILE"
echo "timestamp_epoch,pid,fd_total,fd_files,fd_sockets,fd_pipes,fd_anon,process,event" > "$LOGFILE"

#########################################
#  Monitoring loop
#########################################

while kill -0 "$MAIN_PID" 2>/dev/null; do
    ts=$(date +%s)

    # Track which PIDs we saw this iteration
    declare -A seen_now
    # Gather tree: main + descendants
    for pid in $(list_descendants "$MAIN_PID" | sort -nu); do
        # Skip if PID vanished mid-iteration
        [[ -d "/proc/$pid/fd" ]] || continue
        seen_now["$pid"]=1

        # Total FD count
        fd_total=$(ls "/proc/$pid/fd" 2>/dev/null | wc -l || echo 0)

        # FD type breakdown (optional)
        fd_files=0; fd_sockets=0; fd_pipes=0; fd_anon=0
        if [[ "$FD_TYPES" -eq 1 ]]; then
            read -r fd_files fd_sockets fd_pipes fd_anon <<<"$(fd_type_counts "$pid")"
        fi

        # Process name (cached)
        if [[ -z "${proc_name_map[$pid]+x}" ]]; then
            if [[ -r "/proc/$pid/cmdline" ]]; then
                raw=$(tr '\0' ' ' < "/proc/$pid/cmdline")
                proc_name_map[$pid]=$(basename "$(echo "$raw" | awk '{print $1}')")
            else
                proc_name_map[$pid]="?"
            fi
        fi
        pname=${proc_name_map[$pid]}

        # First/last timestamps
        if [[ -z "${first_ts[$pid]+x}" ]]; then
            first_ts[$pid]=$ts
        fi
        last_ts[$pid]=$ts

        # High-water mark
        prev_max=${max_fd[$pid]:-0}
        if (( fd_total > prev_max )); then
            max_fd[$pid]=$fd_total
        fi

        last_fd[$pid]=$fd_total
        alive[$pid]=1

        # Log normal sample (no explicit event)
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$ts" "$pid" "$fd_total" "$fd_files" "$fd_sockets" "$fd_pipes" "$fd_anon" "$pname" "" \
            >> "$LOGFILE"
    done

    # Detect processes that exited since last iteration
    for pid in "${!alive[@]}"; do
        if [[ -z "${seen_now[$pid]+x}" ]]; then
            # Consider this PID gone; log explicit EXITED event with fd_total = 0
            exit_ts=$(date +%s)
            pname=${proc_name_map[$pid]:-?}
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "$exit_ts" "$pid" "0" "0" "0" "0" "0" "$pname" "EXITED" \
                >> "$LOGFILE"
            unset alive["$pid"]
        fi
    done

    # Cleanup seen_now for next iteration
    unset seen_now

    sleep "$INTERVAL"
done

echo "Main process $MAIN_PID exited. Building summary…"

#########################################
#  Summary: per-PID high-water + leak flag
#########################################

{
    echo "pid,process_name,first_seen_ts,last_seen_ts,max_fd,last_fd,suspected_leak"
    for pid in "${!max_fd[@]}"; do
        pname=${proc_name_map[$pid]:-?}
        fts=${first_ts[$pid]:-}
        lts=${last_ts[$pid]:-}
        maxv=${max_fd[$pid]:-0}
        lastv=${last_fd[$pid]:-0}
        leak="no"
        if (( maxv >= LEAK_THRESHOLD )); then
            leak="yes"
        fi
        echo "$pid,$pname,$fts,$lts,$maxv,$lastv,$leak"
    done
} > "$SUMMARY_FILE"

echo "Summary written to: $SUMMARY_FILE"
echo "Done."
