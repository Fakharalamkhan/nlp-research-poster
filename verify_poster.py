"""Check the poster against the hard requirements: page size, smallest font, raster images."""
import collections

import pdfplumber

with pdfplumber.open("poster.pdf") as pdf:
    assert len(pdf.pages) == 1, f"expected 1 page, got {len(pdf.pages)}"
    page = pdf.pages[0]
    w_pt, h_pt = float(page.width), float(page.height)
    print(f"page size: {w_pt:.2f} x {h_pt:.2f} pt = {w_pt / 72 * 25.4:.1f} x {h_pt / 72 * 25.4:.1f} mm")

    sizes = collections.Counter()
    smallest = []
    for ch in page.chars:
        if not ch["text"].strip():
            continue
        size = round(float(ch["size"]), 2)
        sizes[size] += 1
        smallest.append((size, ch["text"], ch["fontname"]))
    smallest.sort()
    print("characters checked:", sum(sizes.values()))
    print("smallest font size: %.2f pt" % smallest[0][0])
    print("sizes used (pt: count):", dict(sorted(sizes.items())))
    print("10 smallest chars:", [(s, t) for s, t, _ in smallest[:10]])
    below = [c for c in smallest if c[0] < 24]
    print("characters below 24 pt:", len(below))

    print("raster images on page:", len(page.images))
    for im in page.images:
        w_in = (im["x1"] - im["x0"]) / 72
        h_in = (im["bottom"] - im["top"]) / 72
        src_w, src_h = im["srcsize"]
        print(f"  image {src_w}x{src_h}px shown at {w_in:.2f}x{h_in:.2f} in -> {src_w / w_in:.0f} x {src_h / h_in:.0f} PPI")
