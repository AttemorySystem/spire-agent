#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
lib_dir=${VA_GAME_LIB_DIR:-"$repo_root/runtime/lib"}
mods_dir=${VA_GAME_MODS_DIR:-"$repo_root/runtime/mods"}
output_dir=${VA_GAME_MOD_OUTPUT_DIR:-"$repo_root/runtime/mods"}
output="$output_dir/AgentVisualizer.jar"

for dependency in \
    "$lib_dir/desktop-1.0.jar" \
    "$lib_dir/ModTheSpire.jar" \
    "$mods_dir/BaseMod.jar"; do
    if [[ ! -f "$dependency" ]]; then
        echo "Missing build dependency: $dependency" >&2
        exit 1
    fi
done

classes_dir=$(mktemp -d)
trap 'rm -rf "$classes_dir"' EXIT

javac_target_args=(-source 8 -target 8)
if javac -help 2>&1 | grep -- '--release' >/dev/null; then
    javac_target_args=(--release 8)
fi

javac "${javac_target_args[@]}" \
    -cp "$lib_dir/desktop-1.0.jar:$lib_dir/ModTheSpire.jar:$mods_dir/BaseMod.jar" \
    -d "$classes_dir" \
    "$script_dir/src/va/visualizer/AgentVisualizerMod.java"

cp "$script_dir/ModTheSpire.json" "$classes_dir/ModTheSpire.json"
cp \
    "$script_dir/resources/translations_zh_CN.json" \
    "$classes_dir/va/visualizer/translations_zh_CN.json"
mkdir -p "$output_dir"
jar cf "$output" -C "$classes_dir" .

echo "Built $output"
