from __future__ import annotations

import re
import subprocess
from pathlib import Path

import fitz  # type: ignore
from PIL import Image, ImageChops


BASE = Path(__file__).resolve().parent
TABLES_DIR = BASE / "tables"
TMP_DIR = BASE / "table_raster_tmp"
IMG_DIR = BASE / "table_images"
REPORT_TEX = BASE / "report.tex"
WRAPPER_TEX = BASE / "report-wrapper.tex"
DOCX_REPORT_TEX = BASE / "report-docx-images.tex"
DOCX_WRAPPER_TEX = BASE / "report-wrapper-docx-images.tex"


def to_wsl_path(path: Path) -> str:
    path = path.resolve()
    drive = path.drive[0].lower()
    rest = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def run_wsl(cmd: str) -> None:
    result = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"WSL command failed: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def split_latex_row(row_text: str) -> list[str]:
    row_text = row_text.strip()
    if row_text.endswith(r"\\"):
        row_text = row_text[:-2].rstrip()
    return [cell.strip() for cell in row_text.split(" & ")]


def parse_longtable(table_text: str) -> tuple[list[str], list[list[str]], str, str] | None:
    lines = [line.rstrip() for line in table_text.splitlines()]
    try:
        begin_idx = next(index for index, line in enumerate(lines) if line.startswith(r"\begin{longtable}"))
        end_idx = next(index for index, line in enumerate(lines) if line == r"\end{longtable}")
        endlast_idx = next(index for index, line in enumerate(lines) if line == r"\endlastfoot")
        top_idx = next(index for index, line in enumerate(lines) if line == r"\toprule")
    except StopIteration:
        return None

    header_line = next(
        (
            line.strip()
            for line in lines[top_idx + 1 : endlast_idx]
            if line.strip() and not line.strip().startswith(r"\midrule")
        ),
        None,
    )
    data_rows = [line.strip() for line in lines[endlast_idx + 1 : end_idx] if line.strip()]
    if header_line is None or not data_rows:
        return None

    header_cells = split_latex_row(header_line)
    data_cells = [split_latex_row(row) for row in data_rows]
    if any(len(cells) != len(header_cells) for cells in data_cells):
        return None

    prefix = "\n".join(lines[:begin_idx])
    suffix = "\n".join(lines[end_idx + 1 :])
    return header_cells, data_cells, prefix, suffix


def transpose_longtable(table_text: str) -> str | None:
    parsed = parse_longtable(table_text)
    if parsed is None:
        return None

    header_cells, data_cells, prefix, suffix = parsed
    row_count = len(data_cells)
    col_count = len(header_cells)
    if col_count < 5 or row_count > 5:
        return None

    if row_count == 1:
        value_cells = data_cells[0]
        body_lines = [
            r"\centering",
            r"\begin{tabularx}{0.98\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.38\linewidth}>{\RaggedRight\arraybackslash}X@{}}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
        ]
        for header, value in zip(header_cells, value_cells):
            body_lines.append("{} & {} {}".format(header, value, "\\" * 2))
        body_lines.extend([r"\bottomrule", r"\end{tabularx}"])
    else:
        row_labels = [cells[0] for cells in data_cells]
        value_matrix = [cells[1:] for cells in data_cells]
        transposed_columns = "".join(r">{\RaggedLeft\arraybackslash}X" for _ in row_labels)
        body_lines = [
            r"\centering",
            rf"\begin{{tabularx}}{{0.98\linewidth}}{{@{{}}>{{\RaggedRight\arraybackslash}}p{{0.28\linewidth}}{transposed_columns}@{{}}}}",
            r"\toprule",
            "Metric & " + " & ".join(row_labels) + r" \\",
            r"\midrule",
        ]
        for column_index in range(1, col_count):
            row_values = [value_matrix[row_index][column_index - 1] for row_index in range(row_count)]
            body_lines.append(" & ".join([header_cells[column_index], *row_values]) + "\\" * 2)
        body_lines.extend([r"\bottomrule", r"\end{tabularx}"])

    return "\n".join(part for part in [prefix, "\n".join(body_lines), suffix] if part)


def build_table_document(table_body: str, stem: str) -> Path:
    out_tex = TMP_DIR / f"{stem}.tex"
    content_lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[paperwidth=11.7in,paperheight=8.3in,margin=0.2in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage{lmodern}",
        r"\usepackage{makecell}",
        r"\usepackage[expansion=false]{microtype}",
        r"\usepackage{ragged2e}",
        r"\usepackage{tabularx}",
        r"\usepackage{xurl}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\let\oldfootnotesize\footnotesize",
        r"\renewcommand{\footnotesize}{\small}",
        r"\let\oldscriptsize\scriptsize",
        r"\renewcommand{\scriptsize}{\footnotesize}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        table_body,
        r"\end{document}",
        "",
    ]
    out_tex.write_text("\n".join(content_lines), encoding="utf-8")
    return out_tex


def render_table_pdf(wrapper_tex: Path) -> Path:
    wsl_tmp = to_wsl_path(TMP_DIR)
    run_wsl(f"cd '{wsl_tmp}' && pdflatex -interaction=nonstopmode '{wrapper_tex.name}' >/dev/null")
    return wrapper_tex.with_suffix(".pdf")


def pdf_to_jpg(pdf_path: Path, out_prefix: str) -> list[str]:
    out_files: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2.8, 2.8), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            diff = ImageChops.difference(image, white_bg)
            bbox = diff.getbbox()
            if bbox:
                pad = 18
                left = max(0, bbox[0] - pad)
                top = max(0, bbox[1] - pad)
                right = min(image.width, bbox[2] + pad)
                bottom = min(image.height, bbox[3] + pad)
                image = image.crop((left, top, right, bottom))

            out_name = f"{out_prefix}-{i}.jpg"
            out_path = IMG_DIR / out_name
            image.save(out_path, format="JPEG", quality=92, optimize=True)
            out_files.append(out_name)
    finally:
        doc.close()
    return out_files


def build_docx_tex(table_images: dict[str, list[str]]) -> None:
    text = REPORT_TEX.read_text(encoding="utf-8")

    def replace_input(match: re.Match[str]) -> str:
        file_name = match.group(1)
        imgs = table_images.get(file_name, [])
        if not imgs:
            return match.group(0)
        blocks = []
        for img in imgs:
            blocks.append(
                "\\begin{center}\n"
                f"\\includegraphics[width=\\linewidth]{{table_images/{img}}}\n"
                "\\end{center}"
            )
        return "\n\n".join(blocks)

    new_text = re.sub(r"\\input\{tables/([^}]+\.tex)\}", replace_input, text)
    method_overview_img = IMG_DIR / "method_overview-1.jpg"
    if method_overview_img.exists():
        method_overview_block = (
            "\\subsubsection*{Method Overview}\n\n"
            "\\begin{center}\n"
            "\\includegraphics[width=\\linewidth]{table_images/method_overview-1.jpg}\n"
            "\\end{center}\n\n"
            "\\subsubsection*{Tuned Model Settings}"
        )
        new_text = re.sub(
            r"\\subsubsection\*\{Method Overview\}\s*.*?\\subsubsection\*\{Tuned Model Settings\}",
            lambda _m: method_overview_block,
            new_text,
            flags=re.DOTALL,
        )
    DOCX_REPORT_TEX.write_text(new_text, encoding="utf-8")

    wrapper_text = WRAPPER_TEX.read_text(encoding="utf-8")
    wrapper_text = wrapper_text.replace("\\input{report.tex}", "\\input{report-docx-images.tex}")
    DOCX_WRAPPER_TEX.write_text(wrapper_text, encoding="utf-8")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for jpg_file in IMG_DIR.glob("*.jpg"):
        jpg_file.unlink()

    report_text = REPORT_TEX.read_text(encoding="utf-8")
    tables = re.findall(r"\\input\{tables/([^}]+\.tex)\}", report_text)

    table_images: dict[str, list[str]] = {}
    for table in tables:
        table_text = (TABLES_DIR / table).read_text(encoding="utf-8")
        transposed_table_text = transpose_longtable(table_text)
        table_body = transposed_table_text if transposed_table_text is not None else f"\\input{{../tables/{table}}}"
        wrapper = build_table_document(table_body, Path(table).stem)
        pdf = render_table_pdf(wrapper)
        table_images[table] = pdf_to_jpg(pdf, Path(table).stem)

    method_overview_tex = TMP_DIR / "method_overview.tex"
    if method_overview_tex.exists():
        method_overview_pdf = render_table_pdf(method_overview_tex)
        pdf_to_jpg(method_overview_pdf, "method_overview")

    build_docx_tex(table_images)


if __name__ == "__main__":
    main()
