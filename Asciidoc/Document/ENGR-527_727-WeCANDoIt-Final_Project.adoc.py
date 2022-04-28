# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1651108664.467735
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
        __M_writer('\r\n// end document metadata\r\n\r\n\r\n\r\n\r\n\r\n// begin document\r\n[abstract]\r\nAs the consumption of sugary carbonated drinks has increased in recent years, an overwhelming number of aluminum cans have entered waste disposal facilities. To reduce the spatial requirement of the individual cans, we propose a rotationally driven high-throughput can crushing device that operates either via human or machine power to easily deform cans to at least 20% of the original volume. By utilizing a varying thickness wheel, the rotation of the wheel compresses the can against the wall of the device to less than 20% of the volume. The shaft is constructed from AISI 316 while the crushing wheel is cast stainless steel. A finite element study was conducted to validate that the stresses experienced in the design did not surpass the yield stresses of the components. The proposed device solves the issue of crushing cans and allows for automatic reloading to allow for continued operation.\r\n\r\ncite:[uguralAdvancedMechanicsMaterials2019]\r\n.Abstract\r\n// *Keywords:* _{keywords}_\r\n[#sec-intro, {counter:secs}]\r\n\r\n[#sec-nomenclature, {counter:nomenclature}]\r\n== Nomenclature\r\n\r\n== Introduction\r\nAs the consumption of soda continues to increase internationally, landfills are being overwhelmed by the number of empty cans produced. To counteract the volumetric deman of waste faciliti\r\n\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\n[#sec-development, {counter:development}]\r\n== Development of Engineering Specifications\r\n\r\n[#sec-synthesis, {counter:synthesis}]\r\n== Sythesis of the Design\r\nstem:[\\sigma_y = \\frac{1}{2}]\r\n[#sec-design, {counter:design}]\r\n== Deisgn Analysis and Optimization\r\n\r\nFor analyzing the design of the can crushing mechanism, the analysis was broken down into two type: analytical and FEA. The analytical work focused on two points of interested: a point on the handle, and a point on the cross section of the shaft. Due to the complex geometry of the fly wheel, a FEA was carried out on the part assuming the maximum possible load applied by a human. \r\n\r\n=== Analytical Methods\r\n\r\n==== Shaft Cross-Section\r\nFrom Table 6.2 in cite:[uguralAdvancedMechanicsMaterials2019], the equations for the maximum shear stress, stem:[\\tau_A], and angular deflection, stem:[\\theta],  for a hexagonal cross section similar xref:Fig-1[] to are provided:\r\n\r\nimage::./images/shaft_cross_section.png[caption=<span class="figgynumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n[stem#eq-hex-cross-section, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{align}\r\n    \\tau_A = \\frac{5.7T}{a^3}\\')
        __M_writer('    \\theta = \\frac{8.8T}{a^4G}L\r\n\\end{align}\r\n++++\r\n\r\nwhere stem:[T] is the applied torque, stem:[a] is the height of the hexagon, and stem:[G] is the modulus of rigidity. From the geometry of the shaft, stem:[a] is equal to 1 inch. For AISI 316, the shear modulus is 78GPa. For an applied torque of 220lbf·in, the maximum shear stress is predicted to be 8.646MPa which closely matches the FEA results. Furthermore, the maximum predicted deflection is 2.05 milliradians. The delfection of the rod predicted via this equation is not comparable to the FEA results since the effects of the fly wheel prevent some of the deflection that would be experienced by the shaft. \r\n\r\n=== Handle\r\nThe handle of the mechanism is subject to a moment produced by the force applied to the handle. Since the cross section of the bar is rectangular, the standard equation for bending applied. At the end of the handle a force of 20 pounds force is applied. The handle has a length of 11 inches with a cross section of 0.75 inches by 0.6 inches. Using the equation for bending stress at point A on the cross section:\r\n\r\n[stem#eq-rect-cross-section, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\sigma_A = \\frac{M\\ y}{I}\r\n\\end{equation}\r\n++++\r\n\r\nwhere stem:[M = 20\\text{lb}_f*10.125\\text{inch} = 202.5\\text{lb}_f\\cdot\\text{inch}], stem:[y = 0.375\\text{inch}], and stem:[I = \\frac{1}{12}(0.6\\text{inch})(0.75\\text{inch})^3]. This gives a maximum normal stress of 24.8MPa. Once again, this closely matches the results determined in the FEA analysis near the point of interest. \r\n\r\n=== Conclusions\r\nWithin the brief analytical work conducted, both the shear stress in the shaft and the maximum normal stress are both well below the limits of the material. For the fly wheel, a FEA approach is employed due to the complex geometry of the contact surface with the can. The checks provided by the analytical work confirm that the FEA results are close to the predicted values. \r\n\r\nINCLUDE FIGURE OF THE HANDLE\r\n\r\n=== Finite Element Analysis\r\n\r\n[#sec-conclusions, {counter:conclusions}]\r\n== Conclustions\r\n\r\n// [appendix#sec-appendix-Figures]\r\n// == Figures\r\n\r\n\r\n\r\n[bibliography]\r\n== Bibliography\r\nbibliography::[]\r\n// end document\r\n\r\n\r\n\r\n\r\n\r\n// that\'s all folks')
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "uri": "C:\\Users\\cfarmer6\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "source_encoding": "utf-8", "line_map": {"16": 31, "17": 32, "18": 33, "19": 34, "20": 35, "21": 0, "26": 1, "27": 34, "28": 81, "34": 28}}
__M_END_METADATA
"""
