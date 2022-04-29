# -*- coding:utf-8 -*-
from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1651245688.392643
_enable_loop = True
_template_filename = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc'
_template_uri = 'C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc'
_source_encoding = 'utf-8'
_exports = []



from engineering_notation import EngNumber as engr
import numpy as np


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        __M_writer = context.writer()
        __M_writer('// document metadata\r\n= Final Project\r\nJoby M. Anthony III <jmanthony1@liberty.edu>; Carson W. Farmer <cfarmer6@liberty.edu>\r\n:affiliation: PhD Students\r\n:document_version: 1.0\r\n:revdate: April 27, 2022\r\n// :description: \r\n:keywords: foo, bar\r\n:imagesdir: ./ENGR-527_727-WeCANDoIt-Final_Project\r\n:bibtex-file: ENGR-527_727-WeCANDoIt-Final_Project.bib\r\n:toc: auto\r\n:xrefstyle: short\r\n// :sectnums: |,all|\r\n:chapter-refsig: Chap.\r\n:section-refsig: Sec.\r\n:stem: latexmath\r\n:eqnums: AMS\r\n:stylesdir: C:/Users/jmanthony1/Documents/GitHub/WeCANDoIt/Asciidoc/Document\r\n// :stylesdir: C:/Users/cfarmer6/Documents/GitHub/WeCANDoIt/Asciidoc/Document\r\n:stylesheet: asme.css\r\n:noheader:\r\n:nofooter:\r\n:docinfodir: C:/Users/jmanthony1/Documents/GitHub/WeCANDoIt/Asciidoc/Document/\r\n// :docinfodir: C:/Users/cfarmer6/Documents/GitHub/WeCANDoIt/Asciidoc/Document\r\n:docinfo: private\r\n:front-matter: any\r\n:!last-update-label:\r\n\r\n// example variable\r\n// :fn-1: footnote:[]\r\n\r\n// Python modules\r\n')
        __M_writer('\r\n// end document metadata\r\n\r\n\r\n\r\n\r\n\r\n// begin document\r\n[abstract]\r\n.Abstract\r\nAs the consumption of sugary, carbonated drinks has increased in recent years, an overwhelming number of aluminum cans have entered waste disposal facilities.\r\nTo reduce the spatial requirement of the individual cans, we propose a rotationally driven high-throughput can crushing device that operates either via human or machine power to easily deform cans to at least 20% of the original volume.\r\nBy utilizing a varying thickness wheel, the rotation of the wheel functions as a cam to compress the can against the wall of the device to less than 20% of the volume.\r\nThe shaft is constructed from AISI 316 while the crushing wheel is cast stainless steel.\r\nA finite element study was conducted to verify that the stresses experienced in the design did not surpass the yield stresses of the components.\r\nThe proposed device solves the issue of crushing cans and allows for automatic reloading to allow for continuous operation.\r\n\r\n*Keywords:* _{keywords}_\r\n\r\n\r\n\r\n[#sec-nomenclature]\r\n== Nomenclature\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\n[stem#eq-nomenclature, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{align*}\r\n    \\tau &= \\text{shear stress} \\\\')
        __M_writer('    \\theta &= \\text{angular displacement} \\\\')
        __M_writer('    a &= \\text{height of the hexagonal cross-section} \\\\')
        __M_writer('    b & = \\text{base of rectangular cross-section} \\\\')
        __M_writer('    h &= \\text{height of rectangular cross-section} \\\\')
        __M_writer('    F &= \\text{force applied at the end of the handle} \\\\')
        __M_writer('    M &= \\text{resultant moment from the force applied to the handle} \\\\')
        __M_writer('    T &= \\text{torque applied to the shaft} \\\\')
        __M_writer('    y &= \\text{distance from neutral axis for bending stress} \\\\')
        __M_writer('    G &= \\text{shear modulus} \\\\')
        __M_writer('    L &= \\text{handle length}\r\n\\end{align*}\r\n++++\r\n\r\n\r\n\r\n// necessary to move to after `Nomenclature` to avoid section numbering\r\n:sectnums: |,all|\r\n\r\n[#sec-intro, {counter:secs}]\r\n== Introduction\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\nAs the consumption of soda continues to increase internationally, landfills are being overwhelmed by the number of empty cans produced.\r\nCurrent estimates by the EPA predict that 1.9 millions tons of aluminum are produced as beverage packaging cite:[epa] (PER YEAR??).\r\nTo improve the ability to transport the cans to either a recycling facility or a landfill, a reduction of volume is necessary to decrease the required space and increase the ability to transport more cans in a single delivery.\r\nCan crushing mechanisms are typically either human-powered or complex electro-mechanical machines.\r\nThe human aspect of the design encourages people to be engaged in the recycling process and be conscious of their choices.\r\nHowever, creating an electrically driven device automates process and provides capability for high throughput crushing.\r\n\r\nIn the design of the device, an survey of similar can crushing devices was performed to understand the required features for the desired device.\r\nNext, the design was formulated and components were selected from common part providers or were designed to be easily fabricated.\r\nTo ensure the feasability of the design and materials selection, an analytical analysis of two key points was conducted along with a Finite Element Analysis (FEA) of the entire system at the maximum crushing force of the can.\r\nLastly, a comparison of the FEA results with the fatigue life of the components ensures that the devices meets the requirements for an infite life design. \r\n\r\n\r\n\r\n[#sec-development, {counter:secs}]\r\n== Development of Engineering Specifications\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\nTo develop the specifications for the design, several considerations were made to ensure that the design would be constructed effectively and within the limits of human strength.\r\nFirst, the average pulling force for a human was determined to be stem:[20~\\text{lb}_f] which is consistent with NASA standards cite:[christensenman].\r\nIn an effort to reduce the environmental impact of the design, the additional constraint of purchasing as many parts as possible without requiring custom fabricated parts.\r\nIn the final design, less than half of the parts required were custom designed. The remainder were able to be purchased from McMaster-Carr.\r\n\r\nTo determine the required crushing force for an aluminum beverage can, six uniaxial compression tests were conducted and the results are shown in xref:fig-can_plot[].\r\nThe avereage maximum crushing force was found to be approximately stem:[1.3~kN] to reduce the can to stem:[20\\%] of the original volume.\r\nFor the dimensions of an aluminum beverage can, the can was considered to have an original height of stem:[157~mm] and to achieve stem:[80\\%] reduction in volume, the can would need to be deformed to a thickness of stem:[31.4~mm].\r\nThe dimensions of the can were determined via measurements taken from the beverage cans used in the compression tests.\r\nFurthermore, from the project instructions, the final design was required to have the ability to be either mechanically or electrically driven.\r\nAdditionally, the requirement of having the crushing mechanism automatically reload was added to improve the total throughput of the cans to be crushed.\r\nThe metric for the human force and crushing force are determined via mechanical analysis.\r\nFor the material selection, the commercially listed for the components will be used in the design.\r\n\r\n[#fig-can_plot]\r\n.Results from crushing six different aluminum soda cans with the regions of buckling and maximum force denoted. \r\nimage::./compression_results.png[caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}",role=text-center]\r\n\r\n\r\n\r\n[#sec-synthesis, {counter:secs}]\r\n== Sythesis of the Design\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\nIn creating the final design of the system, several commercially and custom fabricated can crushing solutions were considered.\r\nWhile the common lever-based design (xref:fig-manual_device[]) allows for a can to be easily crushed, the challenge arises in creating an automated and high-throughput can crushing device.\r\nThe common industrial devices serve to shred rather than compress the aluminum cans for use in recycling facilities.\r\nBy investigating Do-It-Yourself (DIY) designs, a common method for automating the process is to create a crank-slider mechanism to compress the can.\r\nHowever, the number of moving parts in the design creates a higher likelihood of component failure if the device is operated for long periods of time.\r\n\r\nThe design of the proposed device functions on the use of a "cam" with varying thickness to compress the aluminum can against the wall of the device.\r\nThe thickness of the cam varies angularly allowing for a gradual crushing of the can.\r\nOnce the can has been reduce to stem:[20\\%] of the original volume, the can exits the device via a slot in the bottom of the housing.\r\nAs the wheel completes a revolution, the opening for a can reaches the can entry location and allows the can to drop into the crushing area.\r\nThe cam is suggested to be made from cast stainless steel.\r\n\r\nThe shaft of the cam is hexagonal and purchased from McMaster-Carr and is made from AISI 316 cite:[mcmaster-carrCorrosionResistant316316L2021].\r\nIn the current iteration, a handle purchased from McMaster-Carr cite:[mcmaster-carrCrankHandleMachinable2021] is attached to the shaft and a human is expected to provide the required rotational input to the device.\r\nHowever, a coupler could be used to attach the shaft to a motor to improve the rate of compression for the cans.\r\nThe device is shown in xref:fig-design[].\r\n\r\n[#fig-design]\r\n.Design of the can crushing device with a potion of the housing cut away to view internal crushing chamber.\r\nimage::./design.png[caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n[#fig-manual_device]\r\n.Example of a common design for manual can crushing devices.\r\nimage::./manual_device.jpg[caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\n\r\n\r\n[#sec-design, {counter:secs}]\r\n== Deisgn Analysis and Optimization\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\nFor analyzing the design of the can crushing mechanism, the analysis was broken down into two type: analytical and FEA.\r\nThe analytical work focused on two points of interested: a point on the handle, and a point on the cross-section of the shaft. Due to the complex geometry of the cam, a FEA was carried out on the part assuming the maximum possible load applied by a human. \r\n\r\n\r\n[#sec-design-analytical, {counter:subs}]\r\n=== Analytical Methods\r\n\r\n==== Shaft Cross-Section\r\n[#fig-hex_cross_section]\r\n.Diagram for the hexagonal shaft cross-section with the applied torque and key dimensions highlighted. \r\nimage::./shaft_cross_section.png[caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\nFrom Table 6.2 in cite:[uguralAdvancedMechanicsMaterials2019], the equations for the maximum shear stress, stem:[\\tau_A], and angular deflection, stem:[\\theta], for a hexagonal cross-section.\r\nThe free body diagram for the torque applied to the shaft is shown in xref:fig-hex_cross_section[].\r\nTo calculate the shear stress and angular deflection, the equations for shear stress and deflection from the textbook cite:[uguralAdvancedMechanicsMaterials2019]:\r\n\r\n[stem#eq-hex-cross-section, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{align}\r\n    \\tau_A = \\frac{5.7T}{a^{3}}\\')
        __M_writer('    \\theta = \\frac{8.8T}{a^{4}G}L\r\n\\end{align}\r\n++++\r\nwhere stem:[T] is the applied torque, stem:[a] is the height of the hexagon, and stem:[G] is the modulus of rigidity.\r\nFrom the geometry of the shaft, stem:[a = 1~in].\r\nFor AISI 316, the shear modulus, stem:[G = 78~GPa].\r\nFor an applied torque of stem:[220~\\text{lb}_f\\cdot\\text{in}], the maximum shear stress is predicted to be stem:[8.646~MPa] which closely matches the FEA results.\r\nFurthermore, the maximum predicted deflection is 2.05 milliradians.\r\nThe delfection of the rod predicted via this equation is not comparable to the FEA results since the effects of the cam prevent some of the deflection that would be experienced by the shaft.\r\n\r\n==== Handle\r\n[#fig-handle_fbd]\r\n.FBD for the handle to determine the maximum bending stress at the connection.\r\n// image::./handle.png[width = 20, caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\nimage::./handle.png[caption=<span class="floatnumber">Figure {secs}-{counter:figs}. </span>, reftext="Fig. {secs}-{figs}"]\r\n\r\nThe handle of the mechanism is subject to a moment produced by the force applied to the handle(see xref:fig-handle_fbd[]).\r\nSince the cross-section of the bar is rectangular, the standard equation for bending is applied.\r\nAt the end of the handle, a force of stem:[20~\\text{lb}_f], which is the human strength found from NASA cite:[christensenman], is applied.\r\nThe handle has a length, stem:[L = 11~in] with a cross-sectional area of stem:[0.75~inches~\\times~0.6~in].\r\nUsing the equation for bending stress at point stem:[A] on the cross-section:\r\n\r\n[stem#eq-rect-cross-section, reftext="Eq. {secs}-{counter:eqs}"]\r\n++++\r\n\\begin{equation}\r\n\\sigma_A = \\frac{M\\ y}{I}\r\n\\end{equation}\r\n++++\r\nwhere stem:[M = 20~\\text{lb}_f*10.125~\\text{in} = 202.5~\\text{lb}_f\\cdot\\text{in}], stem:[y = 0.375~\\text{in}], and stem:[I = \\frac{1}{12}(0.6~\\text{in})(0.75~\\text{in})^{3}].\r\nThis gives a maximum normal stress of stem:[24.8~MPa].\r\nOnce again, this closely matches the results determined in the FEA analysis near the point of interest.\r\n\r\n==== Conclusions\r\nWithin the brief analytical work conducted, both the shear stress in the shaft and the maximum normal stress are both well below the limits of the material.\r\nFor the cam, an FEA approach is employed due to the complex geometry of the contact surface with the can.\r\nThe checks provided by the analytical work confirm that the FEA results are close to the predicted values.\r\n\r\n\r\n[#sec-design-fea, {counter:subs}]\r\n=== Finite Element Analysis (FEA)\r\n\r\n\r\n\r\n[#sec-conclusions, {counter:secs}]\r\n== Conclusions\r\n:!subs:\r\n:!figs:\r\n:!tabs:\r\n\r\n\r\n\r\n// [appendix#sec-appendix-Figures]\r\n// == Figures\r\n\r\n\r\n\r\n[bibliography]\r\n== References\r\nbibliography::[]\r\n// end document\r\n\r\n\r\n\r\n\r\n\r\n// that\'s all folks\r\n')
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "uri": "C:\\Users\\jmanthony1\\Documents\\GitHub\\WeCANDoIt\\Asciidoc\\Document\\ENGR-527_727-WeCANDoIt-Final_Project.adoc", "source_encoding": "utf-8", "line_map": {"16": 33, "17": 34, "18": 35, "19": 36, "20": 37, "21": 0, "26": 1, "27": 36, "28": 67, "29": 68, "30": 69, "31": 70, "32": 71, "33": 72, "34": 73, "35": 74, "36": 75, "37": 76, "38": 190, "44": 38}}
__M_END_METADATA
"""
