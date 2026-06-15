#!/usr/bin/env bash
set -euo pipefail

# ---- YOUR PATHS ----
OLD_HOME="/home/jovyan"
NEW_HOME="/home/jovyan/asr-shared/csiargka"
SHELL_RC="${OLD_HOME}/.bashrc"     # or .zshrc
COPY_EXISTING_DOTFILES_ONCE="true"
# --------------------

# ---- EXCLUDES (relative to OLD_HOME) ----
# Add folders here, one per line, NO leading slash
EXCLUDES=(
  "asr-data-segr/"     # the shared volume path (very important)
  "asr-root/"
  "asr-shared/"
  ".local/share/Trash/"
)
# ----------------------------------------

echo "[1/6] Creating new home dir: ${NEW_HOME}"
mkdir -p "${NEW_HOME}"

echo "[2/6] Creating common subdirs"
mkdir -p "${NEW_HOME}/"{.cache,.config,.local,workspace}

if [[ "${COPY_EXISTING_DOTFILES_ONCE}" == "true" ]]; then
  echo "[3/6] Copying ${OLD_HOME} -> ${NEW_HOME}"
  echo "      Excluding:"
  for e in "${EXCLUDES[@]}"; do echo "        - ${e}"; done

  RSYNC_EXCLUDES=()
  for e in "${EXCLUDES[@]}"; do
    RSYNC_EXCLUDES+=(--exclude "${e}")
  done

  rsync -a \
    --info=progress2 \
    "${RSYNC_EXCLUDES[@]}" \
    "${OLD_HOME}/" "${NEW_HOME}/"
else
  echo "[3/6] Skipping copy"
fi

MARK_BEGIN="# >>> shared-volume-home >>>"
MARK_END="# <<< shared-volume-home <<<"

echo "[4/6] Updating ${SHELL_RC} (idempotent)"
touch "${SHELL_RC}"

tmpfile="$(mktemp)"
awk -v b="${MARK_BEGIN}" -v e="${MARK_END}" '
  $0==b {inblk=1; next}
  $0==e {inblk=0; next}
  !inblk {print}
' "${SHELL_RC}" > "${tmpfile}"
cat "${tmpfile}" > "${SHELL_RC}"
rm -f "${tmpfile}"

cat >> "${SHELL_RC}" <<EOF

${MARK_BEGIN}
# Use shared-volume folder as HOME
export HOME="${NEW_HOME}"
mkdir -p "\$HOME/.cache" "\$HOME/.config" "\$HOME/.local" "\$HOME/workspace" 2>/dev/null || true
case "\$-" in
  *i*) cd "\$HOME" ;;
esac
${MARK_END}
EOF

echo "[5/6] Done."
echo "Open a NEW terminal or run:"
echo "  source \"${SHELL_RC}\""
echo
echo "Verify:"
echo "  echo \$HOME"
echo "  pwd"