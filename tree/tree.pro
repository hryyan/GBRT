#TEMPLATE = lib
#CONFIG += debug staticlib
#QMAKE_CXXFLAGS += -fPIC -std=c++0x
TEMPLATE = app
CONFIG += console debug
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += criterion.cpp \
    splitter.cpp \
    treebuilder.cpp \
    main.cpp \
    basetree.cpp \
    tree.cpp \
    util.cpp

HEADERS += criterion.h \
    splitter.h \
    treebuilder.h \
    basetree.h \
    tree.h \
    util.h

LIBS += -L/usr/local/lib
LIBS += -lopencv_core

TARGET = tree
