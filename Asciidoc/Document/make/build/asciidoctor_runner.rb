require 'asciidoctor'
require 'asciidoctor-bibtex'
Asciidoctor.convert_file 'C:\Users\cfarmer6\Documents\GitHub\WeCANDoIt\Asciidoc\Document\make\build\intermediate\ENGR-527_727-WeCANDoIt-Final_Project.adoc', to_file: 'C:\Users\cfarmer6\Documents\GitHub\WeCANDoIt\Asciidoc\Document\ENGR-527_727-WeCANDoIt-Final_Project.html', safe: :unsafe, backend: 'html5', mkdirs: true, basedir: 'C:\Users\cfarmer6\Documents\GitHub\WeCANDoIt\Asciidoc\Document\make\build\intermediate', attributes: 'imagesdir@=''images'' stylesheet@=''asciidoxy-no-toc.css'' multipage'
logger = Asciidoctor::LoggerManager.logger
exit 1 if (logger.respond_to? :max_severity) &&
  logger.max_severity &&
  logger.max_severity >= (::Logger::Severity.const_get 'FATAL')
