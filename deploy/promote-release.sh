#!/usr/bin/env bash
set -euo pipefail

version=$1
digest=$2
revision=$3
[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-beta\.[0-9]+$ ]]
[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]
[[ "$revision" =~ ^[a-f0-9]{40}$ ]]

repository=$GITHUB_REPOSITORY
source="ghcr.io/$repository@$digest"
target="ghcr.io/$repository-release:$version"

skopeo inspect --config "docker://$source" |
    jq -e --arg revision "$revision" --arg source "$GITHUB_SERVER_URL/$repository" '
        .os == "linux" and .architecture == "amd64" and
        .config.Labels["org.opencontainers.image.revision"] == $revision and
        .config.Labels["org.opencontainers.image.source"] == $source
    ' >/dev/null

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
inspect_target() {
    skopeo inspect --no-tags --format '{{.Digest}}' "docker://$target"
}

if existing=$(inspect_target 2>"$work/inspect.err"); then
    if [[ "$existing" != "$digest" ]]; then
        echo "$target already points to $existing. Use a new version." >&2
        exit 1
    fi
else
    if ! grep -Eiq 'manifest unknown|name unknown' "$work/inspect.err"; then
        cat "$work/inspect.err" >&2
        exit 1
    fi
    skopeo copy --all --preserve-digests --digestfile "$work/copied"         "docker://$source" "docker://$target"
    [[ "$(cat "$work/copied")" == "$digest" ]]
fi
[[ "$(inspect_target)" == "$digest" ]]
printf 'Published %s at %s\n' "$target" "$digest"
