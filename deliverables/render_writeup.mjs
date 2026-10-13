// Render writeup.md to writeup.html (A4, print-ready). See deliverables/build.sh.
// Usage: node render_writeup.mjs writeup.md writeup.html
import { readFileSync, writeFileSync } from "node:fs";
import MarkdownIt from "markdown-it";

const [src, out] = process.argv.slice(2);
const body = new MarkdownIt({ html: true }).render(readFileSync(src, "utf8"));

const face = (weight, style, file) =>
  `@font-face { font-family: 'Noto Sans'; font-weight: ${weight}; font-style: ${style}; ` +
  `src: local('${file}'), url('fonts/${file}.ttf'); }`;

const css = `
${face(400, "normal", "NotoSans-Regular")}
${face(400, "italic", "NotoSans-Italic")}
${face(600, "normal", "NotoSans-SemiBold")}
${face(700, "normal", "NotoSans-Bold")}
@font-face { font-family: 'Noto Sans Mono'; src: local('NotoSansMono-Regular'), url('fonts/NotoSansMono-Regular.ttf'); }
@page { size: A4; margin: 20mm 0 18mm 0; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Noto Sans', sans-serif; font-size: 11px; line-height: 1.55; color: #1f2328;
       width: 570px; margin: 0 0 0 75px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 10px; padding-bottom: 6px; border-bottom: 1px solid #d1d9e0; }
h2 { font-size: 16.5px; font-weight: 600; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #d1d9e0; }
p { margin: 0 0 8px; }
code { font-family: 'Noto Sans Mono', monospace; font-size: 85%; background: #f0f2f4; padding: 1px 4px; border-radius: 4px; }
table { border-collapse: collapse; margin: 4px 0 9px; font-size: 10.5px; }
th, td { border: 1px solid #d1d9e0; padding: 3px 10px; }
th { font-weight: 600; }
tr:nth-child(2n) td { background: #f6f8fa; }
`;

writeFileSync(out, `<!doctype html><html><head><meta charset="utf-8">
<title>Caption-Image Retrieval — write-up</title><style>${css}</style></head>
<body>${body}</body></html>`);
