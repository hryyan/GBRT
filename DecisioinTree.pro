TEMPLATE = subdirs
#SUBDIRS += tree ensemble test_tree test_ensemble
SUBDIRS += tree
#SUBDIRS += test_tree
#test_tree.depends = tree
#ensemble.depends = tree
#test_ensemble.depends = ensemble

#CONFIG += console
#CONFIG -= app_bundle
#CONFIG -= qt

#SOURCES += main.cpp \
#    criterion.cpp \
#    splitter.cpp \
#    treebuilder.cpp \
#    tree.cpp

#LIBS += -L/usr/local/lib
#LIBS += -lopencv_core

#HEADERS += \
#    criterion.h \
#    splitter.h \
#    treebuilder.h \
#    tree.h

#configA {
#TARGET = targetA
#DE
#}
