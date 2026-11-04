#!/usr/bin/env bash
# Usage: ./make_video.sh <scenario.json> <out.mp4> [workers]
# Renders all frames in parallel headless Chromium workers and encodes a 1080x1920 H.264 Short.
set -euo pipefail
SCN=$1; OUT=$2; W=${3:-3}
DIR=$(mktemp -d)
cd "$(dirname "$0")"
N=$(node -e "const s=require(require('path').resolve('$SCN'));import('./engine/sim.mjs').then(m=>console.log(Math.ceil(new m.Sim(s).duration*(s.fps||30))))")
CH=$(( (N + W - 1) / W ))
for k in $(seq 0 $((W-1))); do
  node render.cjs "$SCN" "$DIR" $((k*CH)) $(( (k+1)*CH - 1 )) 1 1 $((8800+k)) > "$DIR/log$k.txt" 2>&1 &
done
wait
ffmpeg -y -loglevel error -framerate 30 -pattern_type glob -i "$DIR/f*.jpg" -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p -movflags +faststart "$OUT"
echo "frames: $(ls $DIR/f*.jpg | wc -l) -> $OUT"
rm -rf "$DIR"
