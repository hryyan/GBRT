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
           ../tree/criterion.h \
           ../tree/splitter.h \
           ../tree/util.h \
    util_test.h \
    tools.h
SOURCES += main.cpp \
           criterion_test.cpp \
           splitter_test.cpp \
           ../tree/criterion.cpp \
           ../tree/splitter.cpp \
           ../tree/util.cpp \
    util_test.cpp \
    tools.cpp

LIBS += -L/usr/local/lib
LIBS += -lopencv_core

TARGET = test_tree
