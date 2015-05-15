TEMPLATE = app
CONFIG += console debug
CONFIG -= app_bundle
#CONFIG -= qt
CONFIG += c++11

INCLUDEPATH += ../tree

LIBS += -L../tree
#LIBS += -ltree

HEADERS += criterion_test.h \
           splitter_test.h \
           util_test.h \
           tools.h \
           ../tree/criterion.h \
           ../tree/splitter.h \
           ../tree/basetree.h \
           ../tree/tree.h \
           ../tree/treebuilder.h \
           ../tree/util.h

SOURCES += main.cpp \
           criterion_test.cpp \
           splitter_test.cpp \
           util_test.cpp \
           tools.cpp \
           ../tree/criterion.cpp \
           ../tree/splitter.cpp \
           ../tree/basetree.cpp \
           ../tree/tree.cpp \
           ../tree/treebuilder.cpp \
           ../tree/util.cpp

LIBS += -L/usr/local/lib
LIBS += -lopencv_core

TARGET = test_tree
