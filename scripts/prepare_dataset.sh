#!/usr/bin/env bash
#
# Export RGB/saliency JPEG at H=224 W=1120; normalize in Dataset.
#
#   <root>/C011/001/001/            <root>/C011/001/001/
#       rgb.mp4                         rgb/000001.jpg ...
#       saliency.mp4         ->         saliency/000001.jpg ...
#       sim_gaze_df.csv                 sim_gaze_df.csv
#
# Usage: scripts/prepare_dataset.sh <dataset-root> [jobs]
# CSV coordinates are unchanged. Override GAZE_SOURCE_HEIGHT/WIDTH only if
# the CSV coordinate canvas differs from the author's 720x6400 convention.

set -euo pipefail

ROOT=${1:-./data/raw/drivergaze360/trainval/dataset}
JOBS=${2:-2}

# One recording. Each modality is unpacked into a .partial folder first so an
# interrupted run can just be re-run: finished folders are skipped, half-written
# ones are thrown away.
unpack() {
    local rec=$1 csv_frames src out existing_frames extracted_frames signature kind
    local gaze_height=${GAZE_SOURCE_HEIGHT:-720} gaze_width=${GAZE_SOURCE_WIDTH:-6400}
    [[ $gaze_height =~ ^[1-9][0-9]*$ && $gaze_width =~ ^[1-9][0-9]*$ ]] || {
        echo "Gaze source height/width must be positive integers" >&2; return 1;
    }
    # Never leave stale geometry metadata claiming a partial export is complete.
    rm -f -- "$rec/preprocessing.json"
    csv_frames=$(awk 'END { print NR - 1 }' "$rec/sim_gaze_df.csv")
    if [[ ! $csv_frames =~ ^[1-9][0-9]*$ ]]; then
        echo "FAILED $rec/sim_gaze_df.csv: expected a header and at least one frame row" >&2
        return 1
    fi

    for kind in rgb saliency; do
        src="$rec/$kind.mp4" out="$rec/$kind"
        [[ -f $src ]] || { echo "FAILED: missing source $src" >&2; return 1; }
        signature="jpeg-224x1120-bilinear-q1-v1 $csv_frames $(stat -c '%s %Y' "$src")"
        if [[ -d $out && -f $out/.prepared ]]; then
            existing_frames=$(find "$out" -maxdepth 1 -type f -name '*.jpg' | wc -l)
            if [[ $(cat "$out/.prepared") == "$signature" ]] && (( existing_frames == csv_frames )); then
                continue
            fi
        fi
        rm -rf "$out.partial" && mkdir -p "$out.partial"
        # Frame ids in sim_gaze_df.csv are 1-based, so start numbering at 1.
        ffmpeg -nostdin -loglevel error -y -threads 1 -i "$src" \
            -vf 'scale=1120:224:flags=bilinear' -fps_mode passthrough \
            -threads 1 -frames:v "$csv_frames" -q:v 1 -qmin 1 -start_number 1 \
            "$out.partial/%06d.jpg" </dev/null \
            && extracted_frames=$(find "$out.partial" -maxdepth 1 -type f -name '*.jpg' | wc -l) \
            && (( extracted_frames == csv_frames )) \
            && printf '%s\n' "$signature" > "$out.partial/.prepared" \
            && rm -rf "$out" \
            && mv "$out.partial" "$out" \
            || { echo "FAILED $src; previous output preserved" >&2; return 1; }
    done

    # Distinguish stored JPEG dimensions from the unchanged CSV coordinate canvas.
    printf '{"version":1,"image_size":[224,1120],"gaze_source_size":[%s,%s],"resize_mode":"stretch","csv_coordinates":"unchanged"}\n' \
        "$gaze_height" "$gaze_width" > "$rec/preprocessing.json.partial"
    mv -- "$rec/preprocessing.json.partial" "$rec/preprocessing.json"

    echo "done $rec (JPEG H=224 W=1120)"
}

# xargs re-enters the script to get parallelism across recordings
[[ ${1:-} == --unpack ]] && { unpack "$2"; exit; }
[[ -d $ROOT ]] || { echo "Dataset root does not exist: $ROOT" >&2; exit 1; }
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || { echo "jobs must be positive" >&2; exit 1; }

find "$ROOT" -type f -name sim_gaze_df.csv -printf '%h\0' \
    | sort -z \
    | xargs -0 -r -n1 -P "$JOBS" bash "$(readlink -f "$0")" --unpack
