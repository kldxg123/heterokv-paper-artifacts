"""Render a source-grounded scientific architecture as native vector objects."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
WIDTH, HEIGHT = 1800, 1220
INK, LINE = "#202C36", "#5F7382"
BLUE, GREEN, ORANGE, GRAY = "#E2EDF5", "#E4F0E7", "#F8E9DA", "#F0F2F4"
ACCENTS = {"external": "#738392", "control": "#537E9C", "text": "#609278", "image": "#B18B64"}
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
import os


def font_file(filename, fallback):
    candidates = [Path(os.environ.get("HETEROKV_FONT_DIR", ".")) / filename,
                  Path("C:/Windows/Fonts") / filename,
                  Path("/usr/share/fonts/truetype/dejavu") / fallback]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Install DejaVu Sans fonts or set HETEROKV_FONT_DIR")


for suffix, filename, fallback in (("", "arial.ttf", "DejaVuSans.ttf"),
                                  ("Bold", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
                                  ("Italic", "ariali.ttf", "DejaVuSans-Oblique.ttf")):
    pdfmetrics.registerFont(TTFont("AcademicArial" + suffix, font_file(filename, fallback)))
pdfmetrics.registerFont(TTFont("AcademicMath", font_file("seguisym.ttf", "DejaVuSans.ttf")))


def math_face(text):
    glyphs = pdfmetrics.getFont("AcademicArialItalic").face.charToGlyph
    return "AcademicArialItalic" if all(ord(c) in glyphs for c in text) else "AcademicMath"


def math_width(text, size):
    return pdfmetrics.stringWidth(text, math_face(text), size)


def math_spans(markup):
    root = ET.fromstring("<math>" + markup + "</math>")
    spans = []
    if root.text:
        spans.append((root.text, 1.0, 0))
    for child in root:
        spans.append((child.text or "", .70, .28 if child.tag == "sub" else -.44))
        if child.tail:
            spans.append((child.tail, 1.0, 0))
    return spans


class Scene:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def node(self, ident, x, y, w, h, lines=(), *, fill=GRAY, stroke=LINE,
             font=24, bold=False, parent="1", container=False, align="center", dashed=False):
        if ident in self.nodes:
            raise ValueError("Duplicate object: " + ident)
        self.nodes[ident] = dict(id=ident, x=x, y=y, w=w, h=h, lines=list(lines),
                                 fill=fill, stroke=stroke, font=font, bold=bold,
                                 parent=parent, container=container, align=align,
                                 dashed=dashed, math=None)
        return ident

    def label(self, ident, x, y, w, h, lines, *, font=24, bold=False, parent="1", align="left"):
        return self.node(ident, x, y, w, h, lines, fill=None, stroke=None, font=font,
                         bold=bold, parent=parent, align=align)

    def formula(self, ident, x, y, w, h, markup, *, font=25, parent="1", align="center"):
        self.label(ident, x, y, w, h, [re.sub("<[^>]+>", "", markup)],
                   font=font, parent=parent, align=align)
        self.nodes[ident]["math"] = math_spans(markup)
        self.nodes[ident]["markup"] = markup

    def block(self, ident, x, y, w, h, title, lines, *, fill=GRAY, accent=LINE,
              font=24, parent="1", dashed=False):
        self.node(ident, x, y, w, h, fill=fill, stroke=accent,
                  parent=parent, container=True, dashed=dashed)
        self.label(ident + "_title", x+14, y+5, w-28, 30, [title],
                   font=24, bold=True, parent=ident, align="left")
        if lines:
            self.label(ident + "_body", x+14, y+46, w-28, h-54, lines,
                       font=font, parent=ident, align="left")
        return ident

    def edge(self, ident, source, target, *, start=(1, .5), end=(0, .5), via=(), role="data"):
        a, b = self.nodes[source], self.nodes[target]
        p0 = (a["x"]+start[0]*a["w"], a["y"]+start[1]*a["h"])
        p1 = (b["x"]+end[0]*b["w"], b["y"]+end[1]*b["h"])
        self.edges.append(dict(id=ident, source=source, target=target, start=start, end=end,
                               points=[p0, *via, p1], role=role))

    @staticmethod
    def line_y(node):
        size = node["font"]
        spacing = size * 1.16
        y = node["y"]+(node["h"]-len(node["lines"])*spacing)/2+size*.9
        return [y+i*spacing for i in range(len(node["lines"]))]

    def validate(self):
        errors = []
        for node in self.nodes.values():
            x, y, w, h = (node[key] for key in ("x", "y", "w", "h"))
            if min(x, y) < 0 or x+w > WIDTH or y+h > HEIGHT:
                errors.append("Outside canvas: " + node["id"])
            if node["math"]:
                widths = [math_width(t, node["font"]*scale)
                          for t, scale, _ in node["math"]]
                if sum(widths) > w-4:
                    errors.append("Formula too wide: " + node["id"])
            else:
                face = "AcademicArialBold" if node["bold"] else "AcademicArial"
                for line in node["lines"]:
                    if pdfmetrics.stringWidth(line, face, node["font"]) > w-4:
                        errors.append("Text too wide: " + node["id"] + " / " + line)
                if len(node["lines"])*node["font"]*1.16 > h:
                    errors.append("Text too tall: " + node["id"])
        actual = {(e["source"], e["target"]) for e in self.edges}
        expected = {("raw", "normalization"), ("normalization", "units"), ("units", "index"),
                    ("question", "source_filter"), ("index", "source_filter"),
                    ("source_filter", "candidate"), ("candidate", "base_score"),
                    ("base_score", "rerank"), ("rerank", "b0_pack"),
                    ("source_filter", "b2_pack"), ("b0_pack", "packet"),
                    ("b2_pack", "packet"), ("packet", "processor"),
                    ("processor", "vlm"), ("vlm", "answer_trace")}
        if not expected <= actual:
            errors.append("Missing mechanism edge")
        if any(a in {"base_score", "rerank", "candidate"} and b == "b2_pack" for a, b in actual):
            errors.append("B2 must not depend on text ranking")
        if any(b == "question" for a, b in actual):
            errors.append("Records must not flow into the question")
        if not any(e["source"] == "question" and e["role"] == "control" for e in self.edges):
            errors.append("Source handle must be a control input")
        if errors:
            raise ValueError("\n".join(errors))

    @staticmethod
    def edge_color(edge):
        return {"data": INK, "control": "#537E9C", "trace": "#738392"}[edge["role"]]

    @staticmethod
    def edge_dash(edge):
        return {"data": (), "control": (7, 4), "trace": (2, 4)}[edge["role"]]

    def drawio(self):
        root = ET.Element("mxfile", host="app.diagrams.net", version="26.0.0")
        diagram = ET.SubElement(root, "diagram", id="heterokv-academic", name="Academic architecture")
        model = ET.SubElement(diagram, "mxGraphModel", page="1", pageScale="1", grid="1",
                              gridSize="5", pageWidth=str(WIDTH), pageHeight=str(HEIGHT), background="#FFFFFF")
        cells = ET.SubElement(model, "root")
        ET.SubElement(cells, "mxCell", id="0")
        ET.SubElement(cells, "mxCell", id="1", parent="0")
        for node in self.nodes.values():
            style = ("rounded=1;arcSize=4;html=1;whiteSpace=wrap;spacing=0;fontFamily=Arial;"
                     f"fontColor={INK};fontSize={node['font']};fontStyle={1 if node['bold'] else 0};"
                     f"align={node['align']};fillColor={node['fill'] or 'none'};"
                     f"strokeColor={node['stroke'] or 'none'};strokeWidth=1.3;")
            if node["container"]:
                style += "container=1;pointerEvents=0;"
            if node["dashed"]:
                style += "dashed=1;dashPattern=6 4;"
            value = "<br>".join(html.escape(line) for line in node["lines"])
            if node["math"]:
                value = "<i>" + node["markup"] + "</i>"
            cell = ET.SubElement(cells, "mxCell", id=node["id"], value=value, style=style,
                                 vertex="1", parent=node["parent"])
            parent = self.nodes.get(node["parent"], {"x": 0, "y": 0})
            ET.SubElement(cell, "mxGeometry", x=str(node["x"]-parent["x"]),
                          y=str(node["y"]-parent["y"]), width=str(node["w"]),
                          height=str(node["h"]), **{"as": "geometry"})
        for edge in self.edges:
            sx, sy = edge["start"]
            tx, ty = edge["end"]
            style = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
                     f"strokeColor={self.edge_color(edge)};strokeWidth=1.8;endArrow=block;endFill=1;"
                     f"exitX={sx};exitY={sy};entryX={tx};entryY={ty};exitDx=0;exitDy=0;entryDx=0;entryDy=0;")
            dash = self.edge_dash(edge)
            if dash:
                style += f"dashed=1;dashPattern={dash[0]} {dash[1]};"
            cell = ET.SubElement(cells, "mxCell", id=edge["id"], style=style, edge="1",
                                 parent="1", source=edge["source"], target=edge["target"])
            geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
            if len(edge["points"]) > 2:
                points = ET.SubElement(geometry, "Array", **{"as": "points"})
                for x, y in edge["points"][1:-1]:
                    ET.SubElement(points, "mxPoint", x=str(x), y=str(y))
        return root

    def export_svg(self, path, mxfile):
        root = ET.Element(f"{{{SVG_NS}}}svg", width=str(WIDTH), height=str(HEIGHT),
                          viewBox=f"0 0 {WIDTH} {HEIGHT}", content=ET.tostring(mxfile, encoding="unicode"))
        defs = ET.SubElement(root, "defs")
        for role in ("data", "control", "trace"):
            color = self.edge_color({"role": role})
            marker = ET.SubElement(defs, "marker", id="arrow_"+role, markerWidth="10", markerHeight="10",
                                   refX="9", refY="5", orient="auto", markerUnits="userSpaceOnUse")
            ET.SubElement(marker, "path", d="M0,0 L10,5 L0,10 Z", fill=color)
        ET.SubElement(root, "rect", x="0", y="0", width=str(WIDTH), height=str(HEIGHT), fill="#FFFFFF")
        for node in self.nodes.values():
            group = ET.SubElement(root, "g", id=node["id"])
            if node["fill"] or node["stroke"]:
                attributes = dict(x=str(node["x"]), y=str(node["y"]), width=str(node["w"]),
                                  height=str(node["h"]), rx="5", fill=node["fill"] or "none",
                                  stroke=node["stroke"] or "none", **{"stroke-width": "1.3"})
                if node["dashed"]:
                    attributes["stroke-dasharray"] = "6 4"
                ET.SubElement(group, "rect", **attributes)
            if node["math"]:
                font = node["font"]
                width = sum(math_width(t, font*scale)
                            for t, scale, _ in node["math"])
                x = node["x"] + ((node["w"]-width)/2 if node["align"] == "center" else 0)
                y = node["y"] + node["h"]/2 + font*.32
                for text, scale, offset in node["math"]:
                    label = ET.SubElement(group, "text", x=str(x), y=str(y+offset*font),
                                          **{"font-family": "Arial" if math_face(text) == "AcademicArialItalic" else "Segoe UI Symbol",
                                             "font-style": "italic" if math_face(text) == "AcademicArialItalic" else "normal",
                                             "font-size": str(font*scale), "fill": INK})
                    label.text = text
                    x += math_width(text, font*scale)
            else:
                for line, y in zip(node["lines"], self.line_y(node)):
                    centered = node["align"] == "center"
                    label = ET.SubElement(group, "text", x=str(node["x"]+(node["w"]/2 if centered else 0)),
                                          y=str(y), **{"font-family": "Arial", "font-size": str(node["font"]),
                                                        "font-weight": "bold" if node["bold"] else "normal",
                                                        "text-anchor": "middle" if centered else "start", "fill": INK})
                    label.text = line
        for edge in self.edges:
            attributes = dict(d="M " + " L ".join(f"{x},{y}" for x, y in edge["points"]),
                              fill="none", stroke=self.edge_color(edge), id=edge["id"],
                              **{"stroke-width": "1.8", "marker-end": "url(#arrow_"+edge["role"]+")"})
            dash = self.edge_dash(edge)
            if dash:
                attributes["stroke-dasharray"] = f"{dash[0]} {dash[1]}"
            ET.SubElement(root, "path", **attributes)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def export_pdf(self, path):
        pdf = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT), invariant=1,
                            pageCompression=1, initialFontName="AcademicArial")
        pdf.setTitle("HeteroKV: source-cued evidence architecture and activation contract")
        pdf.setAuthor("")
        pdf.translate(0, HEIGHT)
        pdf.scale(1, -1)
        for node in self.nodes.values():
            x, y, w, h = (node[key] for key in ("x", "y", "w", "h"))
            if node["fill"] or node["stroke"]:
                pdf.saveState()
                pdf.setLineWidth(1.3)
                if node["fill"]:
                    pdf.setFillColor(HexColor(node["fill"]))
                if node["stroke"]:
                    pdf.setStrokeColor(HexColor(node["stroke"]))
                if node["dashed"]:
                    pdf.setDash(6, 4)
                pdf.roundRect(x, y, w, h, 5, fill=int(bool(node["fill"])), stroke=int(bool(node["stroke"])))
                pdf.restoreState()
            if node["math"]:
                size = node["font"]
                width = sum(math_width(t, size*scale)
                            for t, scale, _ in node["math"])
                tx = x + ((w-width)/2 if node["align"] == "center" else 0)
                baseline = y+h/2+size*.32
                for text, scale, offset in node["math"]:
                    self.pdf_text(pdf, text, tx, baseline+offset*size, size*scale, face_override=math_face(text))
                    tx += math_width(text, size*scale)
            else:
                for line, baseline in zip(node["lines"], self.line_y(node)):
                    centered = node["align"] == "center"
                    self.pdf_text(pdf, line, x+(w/2 if centered else 0), baseline, node["font"],
                                  bold=node["bold"], center=centered)
        for edge in self.edges:
            pdf.saveState()
            color = HexColor(self.edge_color(edge))
            pdf.setStrokeColor(color)
            pdf.setFillColor(color)
            pdf.setLineWidth(1.8)
            dash = self.edge_dash(edge)
            if dash:
                pdf.setDash(*dash)
            curve = pdf.beginPath()
            curve.moveTo(*edge["points"][0])
            for point in edge["points"][1:]:
                curve.lineTo(*point)
            pdf.drawPath(curve)
            prev, tip = edge["points"][-2:]
            angle = math.atan2(tip[1]-prev[1], tip[0]-prev[0])
            arrow = pdf.beginPath()
            arrow.moveTo(*tip)
            for side in (-1, 1):
                arrow.lineTo(tip[0]-10*math.cos(angle)+side*4*math.sin(angle),
                             tip[1]-10*math.sin(angle)-side*4*math.cos(angle))
            arrow.close()
            pdf.setDash()
            pdf.drawPath(arrow, fill=1, stroke=0)
            pdf.restoreState()
        pdf.showPage()
        pdf.save()

    @staticmethod
    def pdf_text(pdf, text, x, y, size, *, bold=False, italic=False, center=False, face_override=None):
        pdf.saveState()
        pdf.translate(x, y)
        pdf.scale(1, -1)
        pdf.setFillColor(HexColor(INK))
        pdf.setFont(face_override or "AcademicArial"+("Italic" if italic else "Bold" if bold else ""), size)
        if center:
            pdf.drawCentredString(0, 0, text)
        else:
            pdf.drawString(0, 0, text)
        pdf.restoreState()


def architecture():
    s = Scene()
    s.label("title", 20, 8, 1760, 43,
            ["HeteroKV: Source-Cued Evidence Architecture and Activation Contract"], font=34, bold=True)
    for panel, x, title in (("panel_a", 20, "A  Full-source VLM (conceptual reference)"),
                            ("panel_b", 915, "B  Source-local text top-k (selection controls)")):
        s.node(panel, x, 62, 865, 178, fill="#F6F7F8", stroke=None, container=True)
        s.label(panel+"_title", x+15, 72, 835, 35, [title], font=28, bold=True, parent=panel)
    s.node("a_source", 40, 120, 230, 72, ["All source content", "OCR / layout / images"], font=23, parent="panel_a")
    s.node("a_payload", 305, 120, 205, 72, ["Full-source", "input payload"], fill=BLUE, font=24, parent="panel_a")
    s.node("a_vlm", 550, 120, 130, 72, ["VLM"], fill=BLUE, font=27, bold=True, parent="panel_a")
    s.label("a_answer", 725, 133, 125, 44, ["Answer"], font=26, parent="panel_a")
    s.edge("a_forward", "a_source", "a_payload")
    s.edge("a_invoke", "a_payload", "a_vlm")
    s.edge("a_output", "a_vlm", "a_answer")
    s.label("a_boundary", 40, 201, 820, 29,
            ["Visual evidence retained; active input may grow with source content."], font=23, parent="panel_a")
    s.node("b_source", 935, 120, 170, 72, ["Same supplied", "source scope"], font=23, parent="panel_b")
    s.node("b_selector", 1140, 115, 235, 82, ["Random / OCR lexical", "layout-aware ranking"], fill=GREEN, font=22, parent="panel_b")
    s.node("b_text", 1410, 120, 155, 72, ["Top-k", "text units"], fill=GREEN, font=24, parent="panel_b")
    s.node("b_vlm", 1600, 120, 115, 72, ["VLM"], fill=BLUE, font=27, bold=True, parent="panel_b")
    s.edge("b_scope", "b_source", "b_selector")
    s.edge("b_selection", "b_selector", "b_text")
    s.edge("b_invoke", "b_text", "b_vlm")
    s.label("b_boundary", 935, 201, 820, 29,
            ["Bounded text payload; image pixels and visual relations are not forwarded."], font=23, parent="panel_b")

    s.node("panel_c", 20, 255, 1760, 785, fill="#FFFFFF", stroke=LINE, container=True)
    s.label("c_title", 37, 266, 1730, 39,
            ["C  HeteroKV: query-independent external memory + source-scoped bounded activation"], font=30, bold=True, parent="panel_c")
    planes = (("ingest_plane", 35, 350, "01  Ingest & memory", "Query-independent; CPU / disk"),
              ("selection_plane", 400, 400, "02  Source-local selection", "Supplied handle, not source discovery"),
              ("budget_plane", 880, 435, "03  Evidence-family budgets", "B2 and B0 are distinct operating points"),
              ("generation_plane", 1340, 425, "04  VLM request boundary", "Only packed evidence is activated"))
    for ident, x, w, title, sub in planes:
        s.node(ident, x, 316, w, 708, fill="#F7F9FA", stroke=None, parent="panel_c", container=True)
        s.label(ident+"_title", x+13, 326, w-26, 34, [title], font=25, bold=True, parent=ident)
        s.label(ident+"_sub", x+13, 363, w-26, 30, [sub], font=20, parent=ident)

    s.node("raw", 50, 413, 320, 72, ["Pages / OCR blocks", "Layout spans / image refs"], font=24, parent="ingest_plane")
    s.block("normalization", 50, 527, 320, 88, "Content normalization",
            ["OCR text + source coordinates"], font=20, fill=GRAY, parent="ingest_plane")
    s.block("units", 50, 653, 320, 152, "Typed MemoryUnits",
            ["source_id, unit_id, modality", "text, span / bbox, image ref", "provenance; unit confidence"],
            font=21, fill=GRAY, parent="ingest_plane")
    s.block("index", 50, 847, 320, 149, "Source-local index / files",
            ["unit_id -> typed record", "token -> source-local unit IDs", "CPU index; disk image files"],
            font=21, fill=GRAY, parent="ingest_plane")
    s.edge("normalize_ingest", "raw", "normalization", start=(.5, 1), end=(.5, 0))
    s.edge("typed_ingest", "normalization", "units", start=(.5, 1), end=(.5, 0))
    s.edge("external_write", "units", "index", start=(.5, 1), end=(.5, 0))

    s.node("question", 415, 411, 365, 66, ["Question q + source handle s", "Given by application"],
           fill=BLUE, font=22, bold=True, parent="selection_plane")
    s.block("source_filter", 415, 517, 365, 65, "Source-handle filtering", [],
            fill=BLUE, parent="selection_plane")
    s.formula("source_filter_math", 431, 553, 333, 25,
              "U<sub>s</sub> = {u : source(u) = s}", font=22, parent="source_filter")
    s.block("candidate", 415, 622, 365, 94, "Lexical candidate union", [], fill=BLUE, parent="selection_plane")
    s.formula("candidate_math", 431, 659, 333, 28,
              "C<sub>s</sub> = \u22c3<sub>t \u2208 tok(q)</sub> inv<sub>s</sub>(t)", font=23, parent="candidate")
    s.label("candidate_empty", 431, 689, 333, 24, ["No hit: source-local text fallback"], font=20, parent="candidate")
    s.block("base_score", 415, 743, 365, 125, "Deterministic fusion score", [], fill=BLUE, parent="selection_plane")
    s.formula("score_formula", 431, 782, 333, 31,
              "\u03c3<sub>base</sub>(q,u) = w<sup>T</sup>[L,H,M,A,C]", font=25, parent="base_score")
    s.label("score_weights", 431, 817, 333, 26, ["w = [0.38, 0.34, 0.12, 0.08, 0.08]"], font=21, parent="base_score")
    s.label("score_status", 431, 844, 333, 24, ["Heuristic; not a learned encoder"], font=20, parent="base_score")
    s.block("rerank", 415, 908, 365, 103, "Document / answer rerank",
            ["Top-3k -> document top-k -> reorder", "Query / unit cues; stable ID tie break"],
            font=20, fill=BLUE, parent="selection_plane")
    s.edge("q_handle", "question", "source_filter", start=(.5, 1), end=(.5, 0), role="control")
    s.edge("index_scope", "index", "source_filter", via=((390, 921.5), (390, 549.5)))
    s.edge("scope_candidates", "source_filter", "candidate", start=(.5, 1), end=(.5, 0))
    s.edge("candidate_scoring", "candidate", "base_score", start=(.5, 1), end=(.5, 0))
    s.edge("score_rerank", "base_score", "rerank", start=(.5, 1), end=(.5, 0))

    s.block("b2_pack", 895, 413, 405, 174, "B2  Source-image payload", [],
            fill=ORANGE, accent=ACCENTS["image"], parent="budget_plane")
    s.label("b2_payload", 910, 461, 375, 77,
            ["One routed source image + question", "One image; processor pixel cap", "Independent of text-unit ranking"],
            font=21, parent="b2_pack")
    s.formula("b2_math", 910, 541, 375, 34,
              "P<sub>B2</sub> = Pack<sub>image</sub>(I<sub>s</sub>, q; P<sub>max</sub>)", font=24, parent="b2_pack")
    s.edge("scope_image", "source_filter", "b2_pack", start=(1, .25), end=(0, .48),
           via=((830, 533.25), (830, 496.52)))
    s.formula("image_edge_label", 804, 450, 82, 34, "I<sub>s</sub>", font=24, parent="panel_c")
    s.block("hybrid", 895, 634, 405, 123, "Optional conservative hybrid", [],
            fill="#FFFFFF", dashed=True, parent="budget_plane")
    s.label("hybrid_policy", 910, 681, 375, 72,
            ["Units / regions -> heuristic gate", "Sufficient: unit-based payload", "Insufficient: B2 source-image fallback"],
            font=20, parent="hybrid")
    s.label("family_warning", 910, 769, 375, 28, ["Different text/image cost; not equal payload"],
            font=19, parent="budget_plane", align="center")
    s.block("b0_pack", 895, 817, 405, 179, "B0  Text-only payload", [],
            fill=GREEN, accent=ACCENTS["text"], parent="budget_plane")
    s.label("b0_payload", 910, 864, 375, 74,
            ["At most k OCR / layout units; no image", "E3: 520 chars / block; 2,200 combined", "Common instruction + question added"],
            font=21, parent="b0_pack")
    s.formula("b0_select", 910, 938, 375, 25,
              "E<sub>k</sub> = Top<sub>k</sub>(R(q,U<sub>s</sub>))", font=23, parent="b0_pack")
    s.formula("b0_math", 910, 965, 375, 25,
              "P<sub>B0</sub> = Clip<sub>C_text</sub>(Serialize(E<sub>k</sub>)) + q", font=21, parent="b0_pack")
    s.edge("rank_text", "rerank", "b0_pack", end=(0, .50),
           via=((835, 952), (835, 906.5)))

    s.block("packet", 1355, 413, 395, 195, "EvidencePacket assembly",
            ["Selected unit / image references", "source_id, unit_id, span / bbox", "Scores + breakdown (text route)",
             "Source / transform provenance", "Payload family + active counts"], font=22, fill=BLUE, parent="generation_plane")
    s.edge("image_packet", "b2_pack", "packet", end=(0, .28),
           via=((1325, 500), (1325, 467.6)))
    s.edge("text_packet", "b0_pack", "packet", end=(0, .78),
           via=((1325, 906.5), (1325, 565.1)))
    s.block("processor", 1355, 660, 395, 78, "Chat / vision processor",
            ["Text + selected image, if present"], font=20, fill=BLUE, parent="generation_plane")
    s.block("vlm", 1355, 790, 395, 78, "Qwen2-VL generator",
            ["Local / shared; no weight updates"], font=20, fill=BLUE, parent="generation_plane")
    s.node("answer_trace", 1355, 920, 395, 76, ["Answer + provenance / accounting trace", "No unselected record bypass to the VLM"],
           font=21, fill=BLUE, parent="generation_plane")
    s.edge("packet_processor", "packet", "processor", start=(.5, 1), end=(.5, 0))
    s.edge("processor_model", "processor", "vlm", start=(.5, 1), end=(.5, 0))
    s.edge("model_output", "vlm", "answer_trace", start=(.5, 1), end=(.5, 0))

    s.node("contract", 20, 1055, 1760, 118, fill="#F5F7F9", stroke=LINE, container=True)
    s.label("features_header", 37, 1065, 550, 31, ["Score features (heuristic, not learned)"], font=23, bold=True, parent="contract")
    s.label("features_text", 37, 1100, 550, 70,
            ["L: lexical; H: SHA-1 text-hash cosine; M: metadata", "A: history if recorded; C: stored confidence", "Current hash route does not encode image pixels."], font=20, parent="contract")
    s.label("resource_header", 622, 1065, 650, 31, ["Residency contract: active input vs external storage"], font=23, bold=True, parent="contract")
    s.formula("resource_formula", 622, 1101, 650, 32,
              "M<sub>system</sub> = M<sub>model</sub> + M<sub>active</sub>(P) + M<sub>external</sub>(U) + M<sub>runtime</sub>",
              font=24, align="left", parent="contract")
    s.label("resource_text", 622, 1136, 650, 29,
            ["Input caps are fixed; external CPU / disk storage can grow."], font=21, parent="contract")
    s.label("accounting_header", 1320, 1065, 435, 31, ["Observed request accounting"], font=23, bold=True, parent="contract")
    s.label("accounting_text", 1320, 1100, 435, 70,
            ["Active units / images / tokens; payload bytes", "GPU allocated / reserved; latency", "Provenance is traceability, not a security proof."], font=19, parent="contract")
    s.edge("trace_observation", "answer_trace", "contract", start=(.5, 1), end=(.87, 0),
           via=((1552.5, 1033), (1551.2, 1033)), role="trace")

    for i, (color, title) in enumerate(((GRAY, "External data / storage"), (BLUE, "Control / service"),
                                       (GREEN, "B0 text payload"), (ORANGE, "B2 image payload"))):
        x = 35+i*300
        s.node("legend_swatch"+str(i), x, 1190, 20, 15, fill=color, stroke=LINE)
        s.label("legend_label"+str(i), x+29, 1183, 265, 29, [title], font=21)
    s.label("edge_legend", 1250, 1183, 535, 29,
            ["Solid: evidence   Dashed: control   Dotted: accounting"], font=20)
    return s


def main():
    scene = architecture()
    scene.validate()
    mxfile = scene.drawio()
    stem = OUT / "figures/heterokv_architecture_academic"
    ET.ElementTree(mxfile).write(stem.with_suffix(".drawio"), encoding="utf-8", xml_declaration=True)
    scene.export_svg(stem.with_suffix(".svg"), mxfile)
    scene.export_pdf(stem.with_suffix(".pdf"))
    source_paths = ["remote_edit/src/memory/context_memory_v3.py", "remote_edit/src/memory/document_context_v3.py",
                    "remote_edit/scripts/run_workflow4_multimodal_memory_retrieval.py",
                    "remote_edit/scripts/run_workflow4_qwenvl_vqa_eval.py"]
    report = {
        "status": "geometry_and_flow_checks_passed",
        "format": "native vector shapes and selectable embedded-font text; no raster trace",
        "canvas": [WIDTH, HEIGHT], "objects": len(scene.nodes), "edges": len(scene.edges),
        "layout": "semantic lanes; Graphviz and native draw.io CLI unavailable",
        "main_pipeline": [[e["source"], e["target"]] for e in scene.edges if not e["id"].startswith(("a_", "b_"))],
        "source_hashes": {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in source_paths},
        "outputs": {ext: hashlib.sha256(stem.with_suffix(ext).read_bytes()).hexdigest() for ext in (".drawio", ".svg", ".pdf")},
        "new_experiments": 0, "model_code_changed": False,
        "claims": ["source-cued evidence contract", "B2 independent of unit ranking", "heuristic text score",
                   "optional hybrid is not learned/calibrated", "external storage can grow"],
    }
    (OUT / "audit/academic_architecture_build.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
