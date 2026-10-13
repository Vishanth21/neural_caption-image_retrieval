#!/usr/bin/env bash
# Rebuild writeup.pdf, slides.pdf and slides.pptx from their Markdown sources.
#
# Needs Node.js and Google Chrome (or set CHROME_PATH). Fonts (Noto Sans, SIL
# OFL) are downloaded into deliverables/fonts/ on first run.
#
#   cd deliverables && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

CHROME_PATH="${CHROME_PATH:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
export CHROME_PATH

base=https://github.com/notofonts/notofonts.github.io/raw/main/fonts
mkdir -p fonts
for f in NotoSans/hinted/ttf/NotoSans-{Regular,Italic,SemiBold,Bold}.ttf \
         NotoSansMono/hinted/ttf/NotoSansMono-Regular.ttf; do
  [ -f "fonts/$(basename "$f")" ] || curl -sSfL -o "fonts/$(basename "$f")" "$base/$f"
done

[ -d node_modules ] || npm install --no-save @marp-team/marp-cli markdown-it

# slides: Marp -> PDF and PPTX (with speaker notes)
npx marp slides.md --allow-local-files --pdf  -o slides.pdf  < /dev/null
npx marp slides.md --allow-local-files --pptx -o slides.pptx < /dev/null

# write-up: Markdown -> HTML -> A4 PDF via headless Chrome
node render_writeup.mjs writeup.md writeup.html
"$CHROME_PATH" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/writeup.pdf" "file://$PWD/writeup.html"
rm writeup.html
