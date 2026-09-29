# Knowing When to Revise: Confidence-Gated Self-Correction in LLMs (Research Poster)

Research poster (DIN A1, portrait) and appendix for the NLP module at the University of Trier.
Fakhr E Alam Khan, Matriculation No. 1818211.

The poster summarises an empirical study of confidence-gated self-correction with Qwen2.5-7B-Instruct on
GSM8K and HotpotQA. Code, data, raw outputs and the term paper are in the experiment repository:
<https://github.com/Fakharalamkhan/confidence-gated-self-correction>.

![Poster preview](poster_preview.png)

## Files

- `poster.tex` / `poster.pdf`: the poster (beamerposter, 594 x 841 mm). All charts and diagrams are vector graphics.
- `appendix.tex` / `appendix.pdf`: references (numbered as on the poster), the list of generative AI tools used, and the declaration form (pre-filled, unsigned in this repository).
- `references.bib`: bibliography.
- `declaration/`: the official declaration form (`declaration_form_blank.pdf`) and `declaration_prefilled.tex`, which types the known fields into it.
- `numbers_sources.md`: every number on the poster and the results file it comes from.
- `verify_poster.py`: checks page size, smallest font size (at least 24 pt) and raster image resolution.
- `build.sh`: builds the poster, the pre-filled declaration and the appendix.
- `make_submission.sh`: builds `1818211.zip` (poster + appendix with the signed declaration) in `../poster_submission/`, outside this repository.

## Build

Requires a LaTeX distribution with beamerposter, tcolorbox, pgfplots, qrcode and pdfpages, and Python with `pdfplumber` for the checks.

```bash
bash build.sh
python verify_poster.py
```
