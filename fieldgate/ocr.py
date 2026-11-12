"""Read a receipt image with docling (https://github.com/docling-project/docling) and keep two views of it.

`reading`  docling's own reading order, as its markdown export gives it. This is what most pipelines
           send to a model.
`lines`    the raw OCR lines with their boxes and recognition confidence, which docling keeps when
           `generate_parsed_pages` is on. `fieldgate.layout` rebuilds printed rows from these.

Results are cached as JSON per receipt, so OCR runs once.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Line:
    text: str
    conf: float
    quad: list[float]      # x0,y0,x1,y1,x2,y2,x3,y3 in image pixels, origin top-left, starting bottom-left


@dataclass
class Page:
    width: float
    height: float
    lines: list[Line]
    reading: str           # docling's markdown export

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "reading": self.reading,
                "lines": [asdict(l) for l in self.lines]}

    @staticmethod
    def from_json(d: dict) -> "Page":
        return Page(d["width"], d["height"], [Line(**l) for l in d["lines"]], d["reading"])


_converter = None


def converter():
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption

        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = False
        opts.generate_parsed_pages = True                       # keeps OCR cells with boxes and confidence
        opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True, lang=["en"])
        _converter = DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
    return _converter


def read_image(image: bytes) -> Page:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(image)
        tmp.flush()
        result = converter().convert(tmp.name)
    page = result.pages[0]
    lines: list[Line] = []
    parsed = page.parsed_page
    for cell in (parsed.textline_cells if parsed else []):
        r = cell.rect
        text = cell.text.strip()
        if text:
            lines.append(Line(text, float(cell.confidence),
                              [float(v) for v in (r.r_x0, r.r_y0, r.r_x1, r.r_y1, r.r_x2, r.r_y2, r.r_x3, r.r_y3)]))
    return Page(float(page.size.width), float(page.size.height), lines, result.document.export_to_markdown())


def read_cached(doc_id: str, image: bytes, cache_dir: str | Path) -> Page:
    path = Path(cache_dir) / f"{doc_id}.json"
    if path.exists():
        return Page.from_json(json.loads(path.read_text()))
    page = read_image(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(page.to_json(), ensure_ascii=False))
    return page
