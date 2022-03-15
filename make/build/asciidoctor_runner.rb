require 'asciidoctor'
require 'asciidoctor-bibtex'
Asciidoctor.convert_file 'C:\Users\jmanthony1\Documents\GitHub\WeCANDoIt\make\build\intermediate\installing_asciidoc.adoc', to_file: 'C:\Users\jmanthony1\Documents\GitHub\WeCANDoIt\installing_asciidoc.html', safe: :unsafe, backend: 'html5', mkdirs: true, basedir: 'C:\Users\jmanthony1\Documents\GitHub\WeCANDoIt\make\build\intermediate', attributes: 'imagesdir@=''images'' stylesheet@=''asciidoxy-no-toc.css'' multipage'
logger = Asciidoctor::LoggerManager.logger
exit 1 if (logger.respond_to? :max_severity) &&
  logger.max_severity &&
  logger.max_severity >= (::Logger::Severity.const_get 'FATAL')
