AUTOMAKE_OPTIONS=foreign  
lib_LTLIBRARIES=libfacemodule.la
libfacemodule_la_SOURCES=cJSON.c FaceDetect.cpp hiredis/sds.c hiredis/net.c hiredis/hiredis.c hiredis/dict.c hiredis/async.c
ACLOCAL_AMFLAGS=-I m4
LDFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann
