from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

folder = Path(__file__).resolve().parent
figures = folder / "figures"
files = sorted(figures.glob("*.jpg"))
if not files:
    raise SystemExit("No JPG figures found")
columns = 2
cell_w = 820
cell_h = 520
margin = 22
rows = (len(files) + columns - 1) // columns
canvas = Image.new("RGB", (margin * 2 + columns * cell_w, margin * 2 + rows * cell_h), "white")
draw = ImageDraw.Draw(canvas)
font_candidates = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
]
font_path = next((path for path in font_candidates if Path(path).exists()), None)
font = ImageFont.truetype(font_path, 27) if font_path else ImageFont.load_default()
small_font = ImageFont.truetype(font_path, 19) if font_path else ImageFont.load_default()
for index, path in enumerate(files):
    row, col = divmod(index, columns)
    x = margin + col * cell_w
    y = margin + row * cell_h
    draw.rounded_rectangle((x, y, x + cell_w - 12, y + cell_h - 12), radius=14, outline="#d2d9df", width=2, fill="#f8fafc")
    title = path.stem.replace("_", " ")
    draw.text((x + 18, y + 12), title, fill="#17212b", font=font)
    with Image.open(path) as source:
        preview = source.convert("RGB")
        preview.thumbnail((cell_w - 40, cell_h - 82), Image.Resampling.LANCZOS)
        px = x + (cell_w - preview.width) // 2
        py = y + 52 + (cell_h - 76 - preview.height) // 2
        canvas.paste(preview, (px, py))
    draw.text((x + 18, y + cell_h - 44), f"{index + 1:02d} / {len(files)}", fill="#53616e", font=small_font)
out = folder / "contact_sheet.jpg"
canvas.save(out, "JPEG", quality=92, optimize=True, progressive=True, dpi=(150, 150))
print({"figures": len(files), "size": canvas.size, "bytes": out.stat().st_size, "path": str(out)})
