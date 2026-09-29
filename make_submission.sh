#!/bin/bash
# Build the STUD.IP submission: 1818211.zip with exactly two PDFs (poster + appendix with the SIGNED declaration).
# Put your signed form at declaration/declaration_signed.pdf first (this file is git-ignored).
# Output goes to ../poster_submission/, outside this repository, so the signature is never committed.
set -e
cd "$(dirname "$0")"
OUT="../poster_submission"

if [ ! -f declaration/declaration_signed.pdf ]; then
  echo "Missing declaration/declaration_signed.pdf (the signed declaration form)."
  exit 1
fi

bash build.sh

# Appendix with the signed declaration, built under a separate job name
pdflatex -interaction=nonstopmode -halt-on-error -jobname=appendix_submission \
  "\def\declfile{declaration/declaration_signed.pdf}\input{appendix}" > /dev/null
bibtex appendix_submission > /dev/null
for i in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=appendix_submission \
    "\def\declfile{declaration/declaration_signed.pdf}\input{appendix}" > /dev/null
done

mkdir -p "$OUT"
cp poster.pdf "$OUT/poster.pdf"
cp appendix_submission.pdf "$OUT/appendix.pdf"
python - "$OUT" <<'PY'
import os, sys, zipfile
out = sys.argv[1]
zpath = os.path.join(out, "1818211.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("poster.pdf", "appendix.pdf"):
        z.write(os.path.join(out, name), arcname=name)
with zipfile.ZipFile(zpath) as z:
    print(zpath, "->", z.namelist())
PY
