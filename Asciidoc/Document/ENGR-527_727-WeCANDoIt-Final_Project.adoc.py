# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1651085556.2920156
_enable_loop = True
_template_filename = 'C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc'
_template_uri = 'C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc'
_source_encoding = 'utf-8'
_exports = []



from engineering_notation import EngNumber as engr
import numpy as np


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        __M_writer = context.writer()
        __M_writer('// document metadata\r\n= Final Project\r\nJoby M. Anthony III <jmanthony1@liberty.edu>; Carson W. Farmer <cfarmer6@liberty.edu>\r\n:affiliation: PhD Students\r\n:document_version: 1.0\r\n:revdate: April 27, 2022\r\n// :description: \r\n// :keywords: \r\n:imagesdir: {docdir}/ENGR-527_727-WeCANDoIt-Final_Project\r\n:bibtex-file: ENGR-527_727-WeCANDoIt-Final_Project.bib\r\n:toc: auto\r\n:xrefstyle: short\r\n:sectnums: |,all|\r\n:chapter-refsig: Chap.\r\n:section-refsig: Sec.\r\n:stem: latexmath\r\n:eqnums: AMS\r\n:stylesdir: C:/Users/cfarmer6/Documents/GitHub/WeCANDoIt/Asciidoc/Document/\r\n:stylesheet: asme.css\r\n:noheader:\r\n:nofooter:\r\n:docinfodir: C:/Users/cfarmer6/Documents/GitHub/WeCANDoIt/Asciidoc/Document/\r\n:docinfo: private\r\n:front-matter: any\r\n:!last-update-label:\r\n\r\n// example variable\r\n// :fn-1: footnote:[]\r\n\r\n// Python modules\r\n')
        __M_writer("\r\n// end document metadata\r\n\r\n\r\n\r\n\r\n\r\n// begin document\r\n[abstract]\r\n.Abstract\r\ncite:[uguralAdvancedMechanicsMaterials2019]\r\n// *Keywords:* _{keywords}_\r\n\r\n\r\n\r\n[#sec-intro, {counter:secs}]\r\n== Introduction\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\n\r\n\r\n// [appendix#sec-appendix-Figures]\r\n// == Figures\r\n\r\n\r\n\r\n[bibliography]\r\n== Bibliography\r\nbibliography::[]\r\n// end document\r\n\r\n\r\n\r\n\r\n\r\n// that's all folks")
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "uri": "C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "source_encoding": "utf-8", "line_map": {"16": 31, "17": 32, "18": 33, "19": 34, "20": 35, "21": 0, "26": 1, "27": 34, "33": 27}}
__M_END_METADATA
"""
