"""
CalcDoc - Template-based document generation library.

Wraps python-docx to use CalcTemplate.docx as a template with predefined styles
and header placeholders.

Rich Text Markup Syntax:
    _{text}  -> subscript
    ^{text}  -> superscript
    **text** -> bold
    All other text renders normally.
    
    Example: "gamma_{G,sup} = 1.35" renders as: gamma (with G,sup subscript) = 1.35
"""

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import RGBColor    
from datetime import datetime
from pathlib import Path
from io import BytesIO
import re


class CalcDoc:
    HEADING_STYLES = {
        1: 'D4S Heading 1',
        2: 'Heading 2',
        3: 'Heading 3',
        4: 'Heading 4',
    }
    
    TABLE_STYLE = 'Grid Table Light'
    
    # Default equation rendering settings
    EQ_FONTSIZE = 13
    EQ_DPI = 200
    EQ_MAX_WIDTH_CM = 14
    
    def __init__(
        self,
        project_name: str,
        project_number: str,
        design_element: str,
        calc_title: str,
        calc_by: str = "",
        checked_by: str = "",
        date: str = None,
        template_path: str = None
    ):
        if template_path is None:
            template_path = Path(__file__).parent / "CalcTemplate.docx"
        
        if not Path(template_path).exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        self._doc = Document(template_path)
        self._heading1_count = 0
        
        self.project_name = project_name
        self.project_number = project_number
        self.design_element = design_element
        self.calc_title = calc_title
        self.calc_by = calc_by
        self.checked_by = checked_by
        self.date = date or datetime.now().strftime("%d/%m/%Y")
        
        self._fill_header_placeholders()
        self._clear_template_content()
    
    def _fill_header_placeholders(self):
        replacements = {
            '{{PROJECT NAME}}': self.project_name,
            '{{PROJECT NUMBER}}': self.project_number,
            '{{DESIGN ELEMENT}}': self.design_element,
            '{{CALC TITLE}}': self.calc_title,
            '{{DATE}}': self.date,
            '{{CALC BY}}': self.calc_by,
            '{{CHECKED BY}}': self.checked_by,
        }
        
        for section in self._doc.sections:
            if section.header:
                self._replace_in_element(section.header, replacements)
            if section.footer:
                self._replace_in_element(section.footer, replacements)
        
        for para in self._doc.paragraphs:
            self._replace_in_paragraph(para, replacements)
        for table in self._doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        self._replace_in_paragraph(para, replacements)
    
    def _replace_in_element(self, element, replacements):
        for para in element.paragraphs:
            self._replace_in_paragraph(para, replacements)
        for table in element.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        self._replace_in_paragraph(para, replacements)
    
    def _replace_in_paragraph(self, para, replacements):
        for placeholder, value in replacements.items():
            if placeholder in para.text:
                for run in para.runs:
                    if placeholder in run.text:
                        run.text = run.text.replace(placeholder, value)
                full_text = para.text
                if placeholder in full_text:
                    new_text = full_text.replace(placeholder, value)
                    if para.text != new_text:
                        for i, run in enumerate(para.runs):
                            if i == 0:
                                run.text = new_text
                            else:
                                run.text = ""
    
    def _clear_template_content(self):
        paras_to_remove = []
        for para in self._doc.paragraphs:
            text = para.text.strip()
            if text in ['Heading 1', 'Body text', 'Heading 2', 'Heading 3', 'Heading 4', '']:
                paras_to_remove.append(para)
        for para in paras_to_remove:
            p = para._element
            p.getparent().remove(p)
    
    # =========================================================================
    # Rich Text Parsing
    # =========================================================================
    
    @staticmethod
    def parse_rich_tokens(text):
        """
        Parse text with markup into tokens.
        
        Markup:
            _{text}  -> subscript
            ^{text}  -> superscript
            **text** -> bold
        
        Returns list of dicts: {'text': str, 'sub': bool, 'sup': bool, 'bold': bool}
        """
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            # Check for bold **...**
            if text[i:i+2] == '**':
                end = text.find('**', i + 2)
                if end != -1:
                    tokens.append({'text': text[i+2:end], 'sub': False, 'sup': False, 'bold': True})
                    i = end + 2
                    continue
            
            # Check for subscript _{...}
            if i < n - 1 and text[i] == '_' and text[i+1] == '{':
                end = text.find('}', i + 2)
                if end != -1:
                    tokens.append({'text': text[i+2:end], 'sub': True, 'sup': False, 'bold': False})
                    i = end + 1
                    continue
            
            # Check for superscript ^{...}
            if i < n - 1 and text[i] == '^' and text[i+1] == '{':
                end = text.find('}', i + 2)
                if end != -1:
                    tokens.append({'text': text[i+2:end], 'sub': False, 'sup': True, 'bold': False})
                    i = end + 1
                    continue
            
            # Regular text - accumulate until next special token
            j = i + 1
            while j < n:
                if text[j:j+2] == '**':
                    break
                if j < n - 1 and text[j] == '_' and text[j+1] == '{':
                    break
                if j < n - 1 and text[j] == '^' and text[j+1] == '{':
                    break
                j += 1
            tokens.append({'text': text[i:j], 'sub': False, 'sup': False, 'bold': False})
            i = j
        
        return tokens
    
    @staticmethod
    def apply_tokens_to_paragraph(para, tokens, font_name="Arial", font_size=Pt(10)):
        """Apply parsed tokens as runs to a paragraph."""
        sub_size = Pt(max(6, font_size.pt * 0.8))
        for tok in tokens:
            run = para.add_run(tok['text'])
            run.font.name = font_name
            run.font.size = font_size
            if tok['sub']:
                run.font.subscript = True
                run.font.size = sub_size
            if tok['sup']:
                run.font.superscript = True
                run.font.size = sub_size
            if tok['bold']:
                run.bold = True
    
    
    
    def add_rich_paragraph(self, text, style=None, font_size=Pt(10)):
        """
        Add a paragraph with rich text markup.
        
        Supports: _{subscript}, ^{superscript}, **bold**
        
        Example: "The factor \u03b3_{G,sup} = 1.35 for **permanent** loads"
        """
        para = self._doc.add_paragraph(style=style)
        tokens = self.parse_rich_tokens(text)
        self.apply_tokens_to_paragraph(para, tokens, font_size=font_size)
        return para
    
    # =========================================================================
    # Equation Rendering (LaTeX -> Image)
    # =========================================================================
    
    @staticmethod
    def render_latex(latex_str, fontsize=13, dpi=200):
        """
        Render a LaTeX math string to a PNG image in memory.
        Uses matplotlib mathtext (no LaTeX installation needed).
        
        Args:
            latex_str: LaTeX math string (without $ delimiters)
            fontsize: Font size for rendering
            dpi: Resolution
        
        Returns:
            BytesIO buffer containing PNG image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f'${latex_str}$', fontsize=fontsize,
                 fontfamily='serif', math_fontfamily='cm',
                 verticalalignment='baseline')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    pad_inches=0.08, facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf
    
    def add_equation(self, latex_str, width=None, indent=Cm(1.0)):
        """
        Add a displayed equation rendered from LaTeX.
        
        Args:
            latex_str: LaTeX math string (without $ delimiters).
                       Uses matplotlib mathtext syntax.
            width: Image width (default: auto-sized, capped at EQ_MAX_WIDTH_CM)
            indent: Left indent for the equation paragraph
        
        Returns:
            Paragraph containing the equation image
        """
        buf = self.render_latex(latex_str, fontsize=self.EQ_FONTSIZE, dpi=self.EQ_DPI)
        
        # Get image dimensions
        from PIL import Image
        img = Image.open(buf)
        img_w, img_h = img.size
        buf.seek(0)
        
        # Calculate appropriate width
        natural_width_cm = (img_w / self.EQ_DPI) * 2.54
        
        if width is None:
            width = Cm(min(natural_width_cm, self.EQ_MAX_WIDTH_CM))
        
        para = self._doc.add_paragraph()
        para.paragraph_format.left_indent = indent
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(4)
        
        run = para.add_run()
        run.add_picture(buf, width=width)
        
        return para
    
    def add_equation_with_ref(self, latex_str, ref_text, width=None, indent=Cm(1.0)):
    
        buf = self.render_latex(latex_str, fontsize=self.EQ_FONTSIZE, dpi=self.EQ_DPI)
        from PIL import Image
        img = Image.open(buf)
        img_w, img_h = img.size
        buf.seek(0)
    
        natural_width_cm = (img_w / self.EQ_DPI) * 2.54
        if width is None:
            width = Cm(min(natural_width_cm, self.EQ_MAX_WIDTH_CM))
    
        para = self._doc.add_paragraph()
        para.paragraph_format.left_indent = indent
        para.paragraph_format.space_before = Pt(4)
        para.paragraph_format.space_after = Pt(4)
    
        # Calculate right margin position in twips, relative to page left edge
        section = self._doc.sections[-1]
        content_width_emu = section.page_width - section.left_margin - section.right_margin
        # Tab pos is relative to the paragraph indent, so subtract it
        tab_pos_twips = int((content_width_emu - indent) / 914400 * 1440)
    
        pPr = para._element.get_or_add_pPr()
        tabs = pPr.makeelement(qn('w:tabs'), {})
        tab_el = tabs.makeelement(qn('w:tab'), {
            qn('w:val'): 'right',
            qn('w:pos'): str(tab_pos_twips),
        })
        tabs.append(tab_el)
        pPr.append(tabs)
    
        # Suppress default tab stops so only our right tab is used
        pPr_existing = para._element.pPr
        # Remove any existing default tab interval if present, or set to very large
        # This is actually a document-level setting, so set it once:
        settings = self._doc.settings.element
        existing_dtsi = settings.find(qn('w:defaultTabStop'))
        if existing_dtsi is None:
            settings.append(settings.makeelement(
                qn('w:defaultTabStop'), {qn('w:val'): str(tab_pos_twips)}
            ))
    
        # Equation image
        run = para.add_run()
        run.add_picture(buf, width=width)
    
        # Tab then ref text, pinned to right margin
        tab_run = para.add_run()
        tab_run._element.append(tab_run._element.makeelement(qn('w:tab'), {}))
        ref_run = para.add_run(ref_text)
        ref_run.font.size = Pt(8)
        ref_run.font.name = "Arial"
        ref_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    
        return para
    
    # =========================================================================
    # Main document methods
    # =========================================================================
    
    def add_heading(self, text, level=1):
        if level not in self.HEADING_STYLES:
            raise ValueError(f"Heading level must be 1-4, got {level}")
        
        if level == 1:
            self._heading1_count += 1
            if self._heading1_count > 1:
                self._doc.add_page_break()
        
        style = self.HEADING_STYLES[level]
        
        # If text contains rich markup, parse it
        if '_{' in text or '^{' in text or '**' in text:
            para = self._doc.add_paragraph(style=style)
            tokens = self.parse_rich_tokens(text)
            # Get heading font size from style (approximate)
            heading_sizes = {1: Pt(14), 2: Pt(12), 3: Pt(11), 4: Pt(10)}
            font_sz = heading_sizes.get(level, Pt(10))
            for tok in tokens:
                run = para.add_run(tok['text'])
                run.font.size = font_sz
                if tok['sub']:
                    run.font.subscript = True
                    run.font.size = Pt(max(7, font_sz.pt * 0.8))
                if tok['sup']:
                    run.font.superscript = True
                    run.font.size = Pt(max(7, font_sz.pt * 0.8))
                if tok['bold']:
                    run.bold = True
        else:
            para = self._doc.add_paragraph(text, style=style)
        
        return para
    
    def add_paragraph(self, text="", style=None):
        return self._doc.add_paragraph(text, style=style)
    
    def add_table(self, rows, cols, style=None):
        table = self._doc.add_table(rows=rows, cols=cols)
        table.style = style or self.TABLE_STYLE
        return table
    
    def add_picture(self, image_path_or_stream, width=None, height=None,
                max_height=Cm(21.4)):
        from PIL import Image
        from io import BytesIO
    
        # Determine native aspect ratio
        if isinstance(image_path_or_stream, BytesIO):
            pos = image_path_or_stream.tell()
            img = Image.open(image_path_or_stream)
            image_path_or_stream.seek(pos)
        else:
            img = Image.open(image_path_or_stream)
        w_px, h_px = img.size
        aspect = h_px / w_px
    
        # If width is specified, check implied height
        if width is not None and height is None and max_height is not None:
            implied_height = int(width * aspect)   # EMU * ratio = EMU
            if implied_height > max_height:
                # Constrain by height, drop width to preserve aspect ratio
                height = max_height
                width = None
    
        return self._doc.add_picture(image_path_or_stream, width=width, height=height)
    
    #def add_picture(self, image_path_or_stream, width=None, height=None):
    #    return self._doc.add_picture(image_path_or_stream, width=width, height=height)
    
    def add_page_break(self):
        return self._doc.add_page_break()
    
    def save(self, path):
        self._doc.save(path)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def paragraphs(self):
        return self._doc.paragraphs
    
    @property
    def tables(self):
        return self._doc.tables
    
    @property
    def sections(self):
        return self._doc.sections
    
    @property
    def styles(self):
        return self._doc.styles
    
    @property
    def core_properties(self):
        return self._doc.core_properties
    
    @property
    def document(self):
        return self._doc


__all__ = ['CalcDoc', 'Inches', 'Pt', 'Cm', 'WD_ALIGN_PARAGRAPH', 'WD_TABLE_ALIGNMENT']