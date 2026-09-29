#!/bin/bash
# Build the poster (two passes: overlay positions and QR code), the pre-filled declaration and the appendix.
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null
echo "poster:   errors $(grep -c '^!' poster.log || true), overfull $(grep -c 'Overfull' poster.log || true)"

# two passes: the filled-in fields are placed with an overlay that needs the page position from pass 1
(cd declaration && for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error declaration_prefilled.tex > /dev/null; done)

pdflatex -interaction=nonstopmode -halt-on-error appendix.tex > /dev/null
bibtex appendix > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error appendix.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error appendix.tex > /dev/null
echo "appendix: errors $(grep -c '^!' appendix.log || true), undefined $(grep -ci 'undefined' appendix.log || true), overfull $(grep -c 'Overfull' appendix.log || true)"
