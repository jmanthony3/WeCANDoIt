# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1647874395.0343273
_enable_loop = True
_template_filename = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Installing\\installing_asciidoc.adoc'
_template_uri = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Installing\\installing_asciidoc.adoc'
_source_encoding = 'utf-8'
_exports = []



from engineering_notation import EngNumber as engr
import numpy as np


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        __M_writer = context.writer()
        __M_writer('// document metadata\r\n= Installation Instructions for AsciiDoc\r\nJoby M. Anthony III <jmanthony1@liberty.edu>\r\n:affiliation: PhD Student\r\n:document_version: 1.0\r\n:revdate: March 14, 2022\r\n:description: Short tutorial to install/use the AsciiDoc markup language.\r\n// :keywords: AsciiDoc, Ruby, Markup\r\n:imagesdir: ./installing_asciidoc\r\n:bibtex-file: installing_asciidoc.bib\r\n:toc: auto\r\n:xrefstyle: short\r\n:sectnums: |,all|\r\n:chapter-refsig: Chap.\r\n:section-refsig: Sec.\r\n:stem: latexmath\r\n:eqnums: AMS\r\n// :stylesdir: ./\r\n// :stylesheet: asme.css\r\n// :noheader:\r\n// :nofooter:\r\n// :docinfo: private\r\n// :docinfodir: ./\r\n:front-matter: any\r\n:!last-update-label:\r\n// :source-highlighter: rouge\r\n\r\n// example variable\r\n// :fn-1: footnote:[]\r\n\r\n// Python modules\r\n')
        __M_writer("\r\n// end document metadata\r\n\r\n\r\n\r\n\r\n\r\n// begin document\r\n// [abstract]\r\n// .Abstract\r\n\r\n// *Keywords:* _{keywords}_\r\n\r\n\r\n\r\n[#sec-requirements, {counter:secs}, {counter:subs}, {counter:figs}]\r\n== Software Requirements and Installation Links to Executables/Binaries\r\n:subs: 0\r\n:figs: 0\r\n\r\nTo write https://asciidoctor.org/[AsciiDoc] markup documents and compile to other formats cite:[AsciidoctorFastOpen], be able to run the https://www.ruby-lang.org/en/[Ruby language] cite:[RubyProgrammingLanguage].\r\nRefer to https://asciidoctor.org/[official AsciiDoc installation instructions] cite:[AsciidoctorFastOpen] for local machine setup.\r\n\r\n\r\n\r\n[#sec-gems, {counter:secs}, {counter:subs}, {counter:figs}]\r\n== Installing Gems\r\n:subs: 0\r\n:figs: 0\r\n\r\n\r\n[#sec-gems-required, {counter:subs}]\r\n=== Required\r\n* `gem install asciidoctor` cite:[AsciidoctorFastOpen]\r\n\r\n\r\n[#sec-gems-recommended, {counter:subs}]\r\n=== Recommended\r\n.Syntax Highlighting cite:[AsciidoctorPDFAsciidoctor]\r\n* `gem install rouge`\r\n* `gem install pygments.rb`\r\n* `gem install coderay`\r\n\r\nSet the appropriate variable (example): `:source-highlighter: rouge`\r\n\r\n.PDF Output cite:[AsciidoctorPDFAsciidoctor]\r\n* `gem install asciidoctor-pdf`\r\n\r\n.LaTeX Support for Typesetting Mathematics cite:[AsciidoctorLaTeX2022]\r\n* `gem install asciidoctor-latex --pre`\r\n\r\n.BibTeX Files and Citations cite:[AsciidoctorbibtexBibtexIntegration2022]\r\n* `gem install asciidoctor-bibtex`\r\n\r\n.AsciiDoxy cite:[Installation]\r\n* `pip3 install --update asciidoxy`\r\n\r\n\r\n\r\n// [appendix#sec-appendix-Figures]\r\n// == Figures\r\n\r\n\r\n\r\n[bibliography]\r\n== Bibliography\r\nbibliography::[]\r\n// end document\r\n\r\n\r\n\r\n\r\n\r\n// that's all folks")
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Installing\\installing_asciidoc.adoc", "uri": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Installing\\installing_asciidoc.adoc", "source_encoding": "utf-8", "line_map": {"16": 32, "17": 33, "18": 34, "19": 35, "20": 36, "21": 0, "26": 1, "27": 35, "33": 27}}
__M_END_METADATA
"""
