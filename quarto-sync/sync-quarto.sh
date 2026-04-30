#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
sync-quarto.sh

Copies Quarto render outputs (HTML + *_files/) and an images directory into a website repo.
Note: the website lives in a different repo/path; provide it via --dest or DEST.

Usage:
  sync-quarto.sh --src <dir-or-qmd> --dest <website-path> [--stem <name>] [--images <images-dir>]

Args:
  --src     Directory containing Quarto output, OR a .qmd file (we'll infer its directory)
  --dest    Destination directory inside your website repo (must be provided; different repo)
  --stem    (Optional) Basename without extension. Copies:
              <stem>.html and <stem>_files/
            If omitted, we try to infer a single .html file in --src.
  --images  (Optional) Path to images directory to copy.
            If omitted, we search upward from --src for a directory named "images".

Examples:
  ./sync-quarto.sh --src ./post.qmd --dest "/path/to/website/content/posts/bizzaro"
  ./sync-quarto.sh --src . --stem post --images "../../../../images" --dest "/path/to/website/posts/bizzaro"

Environment:
  DEST can be used instead of --dest.
EOF
}

SRC=""
DEST="${DEST:-}"
STEM=""
IMAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --stem) STEM="${2:-}"; shift 2 ;;
    --images) IMAGES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SRC" ]]; then
  echo "--src is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "$DEST" ]]; then
  echo "--dest (or DEST env var) is required" >&2
  usage >&2
  exit 2
fi

if [[ -f "$SRC" ]]; then
  # Accept .qmd or any file; use its directory.
  SRC_DIR="$(cd "$(dirname "$SRC")" && pwd)"
else
  SRC_DIR="$(cd "$SRC" && pwd)"
fi

DEST_DIR="$(mkdir -p "$DEST" && cd "$DEST" && pwd)"

infer_stem_from_html() {
  local d="$1"
  # Prefer a single html file in the directory (excluding hidden files).
  local htmls=()
  while IFS= read -r -d '' f; do htmls+=("$f"); done < <(find "$d" -maxdepth 1 -type f -name "*.html" -print0)
  if [[ ${#htmls[@]} -eq 1 ]]; then
    local base
    base="$(basename "${htmls[0]}")"
    printf '%s' "${base%.html}"
    return 0
  fi
  return 1
}

if [[ -z "$STEM" ]]; then
  if ! STEM="$(infer_stem_from_html "$SRC_DIR")"; then
    echo "Couldn't infer --stem (need exactly one .html in $SRC_DIR). Provide --stem explicitly." >&2
    exit 2
  fi
fi

HTML_SRC="$SRC_DIR/$STEM.html"
FILES_SRC="$SRC_DIR/${STEM}_files"

if [[ ! -f "$HTML_SRC" ]]; then
  echo "Missing HTML output: $HTML_SRC" >&2
  echo "Run: quarto render ${STEM}.qmd (or the right file) first." >&2
  exit 1
fi
if [[ ! -d "$FILES_SRC" ]]; then
  echo "Missing Quarto assets directory: $FILES_SRC" >&2
  echo "Run: quarto render ${STEM}.qmd (or the right file) first." >&2
  exit 1
fi

find_images_dir() {
  local start="$1"
  local cur="$start"
  for _ in {1..8}; do
    if [[ -d "$cur/images" ]]; then
      printf '%s' "$cur/images"
      return 0
    fi
    cur="$(cd "$cur/.." && pwd)"
  done
  return 1
}

if [[ -z "$IMAGES" ]]; then
  if IMAGES="$(find_images_dir "$SRC_DIR")"; then
    :
  else
    echo "Couldn't find an images directory by searching upward from $SRC_DIR." >&2
    echo "Provide it explicitly via --images /path/to/images" >&2
    exit 1
  fi
fi

IMAGES_DIR="$(cd "$IMAGES" && pwd)"
if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Images directory not found: $IMAGES" >&2
  exit 1
fi

echo "Source dir:   $SRC_DIR"
echo "Dest dir:     $DEST_DIR"
echo "Stem:         $STEM"
echo "Images dir:   $IMAGES_DIR"
echo

echo "Copying HTML and Quarto assets..."
rsync -a --delete "$HTML_SRC" "$DEST_DIR/"
rsync -a --delete "$FILES_SRC/" "$DEST_DIR/${STEM}_files/"

echo "Copying images..."
mkdir -p "$DEST_DIR/images"
rsync -a --delete "$IMAGES_DIR/" "$DEST_DIR/images/"

echo
echo "Done."
