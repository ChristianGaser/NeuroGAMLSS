#!/usr/bin/env python3
"""
method_scheme.py - The figures of docs/NeuroGAMLSS Method Scheme.html as standalone SVG
files, for Markdown pages such as docs/NeuroGAMLSS Method Scheme.md.

The HTML page is the source of the figures: they are inline SVG, styled by the CSS of
the page.  This script writes every <svg> of a <figure> to a file of its own, named
after the section of the page, with

  - the styles of the figures in the light colours of the page and a white background,
  - subsets of the fonts embedded as WOFF (IBM Plex Sans Condensed, IBM Plex Mono and
    STIX Two Text, all under the SIL Open Font License 1.1), so that the text fits its
    boxes in every browser, also where these fonts are not installed.

The fonts are downloaded once from Google Fonts into a cache folder (--font-dir).
--no-fonts writes the files without them; the text then uses the installed fonts.

    python3 figures/method_scheme.py      # writes docs/figures/method_scheme_*.svg

Edit a figure in the HTML page, then run the script again.
"""
import argparse
import base64
import hashlib
import html
import io
import logging
import os
import re
import urllib.request

logging.getLogger('fontTools').setLevel(logging.ERROR)
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, '..', 'docs', 'NeuroGAMLSS Method Scheme.html')
OUT = os.path.join(HERE, '..', 'docs', 'figures')
CACHE = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
                     'neurogamlss', 'fonts')

# fonts of the figures; for an unknown user agent, Google Fonts serves static TTF files
FONTS_CSS = ('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@400;500;600'
             '&family=IBM+Plex+Mono:wght@400'
             '&family=STIX+Two+Text:ital,wght@0,400;0,600;1,400')
FEATURES = ['kern', 'liga', 'ccmp', 'locl', 'mark', 'mkmk']    # OpenType features kept

# font stacks instead of those of the page; STIX Two Text also provides the symbols that
# IBM Plex lacks, such as the vertical ellipsis
STACKS = {
    'font-cond': '"IBM Plex Sans Condensed", "STIX Two Text", "Arial Narrow", sans-serif',
    'font-mono': '"IBM Plex Mono", "STIX Two Text", Menlo, Consolas, monospace',
    'font-math': '"STIX Two Text", "Cambria Math", "Times New Roman", serif',
}


def page_parts(page):
    """Colour tokens of the light theme, the CSS of the figures and the figures
    (section id, <svg> markup) of the HTML page."""
    marker = '/* ---------- inline SVG ---------- */'
    style = next(s for s in re.findall(r'<style>(.*?)</style>', page, re.S) if marker in s)
    root = re.search(r':root\s*\{(.*?)\}', style, re.S).group(1)       # light values
    tokens = dict(re.findall(r'--([\w-]+)\s*:\s*([^;]+);', root))
    tokens.update(STACKS)
    css = style[style.index(marker):]
    css = re.sub(r'var\(--([\w-]+)\)', lambda m: tokens[m.group(1)].strip(), css)
    figures = []
    for sec in re.finditer(r'<section id="([\w-]+)">(.*?)</section>', page, re.S):
        svg = re.search(r'<svg\b.*?</svg>', sec.group(2), re.S)
        if svg:
            figures.append((sec.group(1), svg.group(0)))
    return tokens, css.strip(), figures


def fetch_fonts(cache):
    """TTF files of FONTS_CSS, downloaded once: [(family, style, weight, path)]."""
    os.makedirs(cache, exist_ok=True)
    index = os.path.join(cache, f'fonts-{hashlib.sha1(FONTS_CSS.encode()).hexdigest()[:8]}.css')
    if not os.path.isfile(index):
        with urllib.request.urlopen(FONTS_CSS, timeout=30) as r:
            css = r.read().decode('utf-8')
        with open(index, 'w', encoding='utf-8') as f:
            f.write(css)
    with open(index, encoding='utf-8') as f:
        css = f.read()
    faces = []
    for block in re.findall(r'@font-face\s*\{(.*?)\}', css, re.S):
        get = dict(re.findall(r"(font-family|font-style|font-weight|src)\s*:\s*([^;]+);", block))
        url = re.search(r'url\(([^)]+)\)', get['src']).group(1)
        path = os.path.join(cache, os.path.basename(url))
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        faces.append((get['font-family'].strip("'\""), get['font-style'], get['font-weight'],
                      path))
    if not faces:
        raise RuntimeError(f"No fonts in {index}; delete it and try again.")
    return faces


def font_faces(faces, text):
    """@font-face rules with the fonts subset to the characters of text, as WOFF without
    hinting, which browsers do not need for these figures."""
    from fontTools import subset
    from fontTools.ttLib import TTFont
    rules = []
    for family, style, weight, path in faces:
        opts = subset.Options()
        opts.flavor = 'woff'
        opts.hinting = False
        opts.layout_features = FEATURES
        opts.name_IDs = ['*']                  # keep the copyright and license notices
        font = TTFont(path, recalcTimestamp=False)         # the same files on every run
        sub = subset.Subsetter(opts)
        sub.populate(text=text)
        sub.subset(font)
        buf = io.BytesIO()
        font.save(buf)
        data = base64.b64encode(buf.getvalue()).decode('ascii')
        rules.append(f'@font-face {{ font-family: "{family}"; font-style: {style}; '
                     f'font-weight: {weight}; src: url(data:font/woff;base64,{data}) '
                     'format("woff"); }')
    return '\n'.join(rules)


def standalone(svg, css, background, fonts=''):
    """The inline <svg> as an SVG file with its styles, size, title and background."""
    tag = re.match(r'<svg\b([^>]*)>', svg)
    attrs = tag.group(1)
    w, h = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', attrs).groups()
    label = re.search(r'aria-label="([^"]*)"', attrs)
    title = html.escape(html.unescape(label.group(1)), quote=False) if label else ''
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"{attrs}>\n'
            f'<title>{title}</title>\n'
            f'<style>\n{fonts}\n{css}\n</style>\n'
            f'<rect width="100%" height="100%" fill="{background}"/>\n'
            + svg[tag.end():].lstrip('\n') + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--page', default=PAGE, help="HTML page with the figures")
    p.add_argument('--out', default=OUT, help="folder of the SVG files")
    p.add_argument('--font-dir', default=CACHE, help="cache of the downloaded fonts")
    p.add_argument('--no-fonts', action='store_true',
                   help="do not embed the fonts (the text then uses the installed fonts)")
    a = p.parse_args(argv)

    with open(a.page, encoding='utf-8') as f:
        tokens, css, figures = page_parts(f.read())
    faces = [] if a.no_fonts else fetch_fonts(a.font_dir)
    os.makedirs(a.out, exist_ok=True)
    for i, (name, svg) in enumerate(figures, 1):
        mono = re.search(r'class="[^"]*\bmono\b', svg) is not None
        used = [face for face in faces if mono or 'Mono' not in face[0]]
        text = html.unescape(re.sub(r'<[^>]+>', '', svg))
        fonts = font_faces(used, ''.join(sorted(set(text)))) if used else ''
        path = os.path.join(a.out, f'method_scheme_{i}_{name}.svg')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(standalone(svg, css, tokens['surface'].strip(), fonts))
        print(f"saved {os.path.relpath(path)} ({os.path.getsize(path) / 1024:.0f} kB)")


if __name__ == '__main__':
    main()
