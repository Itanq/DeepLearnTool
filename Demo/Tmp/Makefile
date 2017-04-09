ProDir = $(shell pwd)

INCLUDEFLAGS = -I./include \
			   -I/usr/include

LIBSFLAGS = -L$(HOME)/caffe/build/lib \
			-L/usr/lib/x86_64-linux-gnu 


SOURCES = $(notdir $(wildcard $(ProDir)/src/*.cpp))
OBJS    = $(SOURCES:.cpp=.o)


LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lcaffe
CFLAGS  = -g -Wall -std=c++11

.PHONY:all 

all : test

test:$(OBJS)
	$(CXX) -o $@ $^ $(LIBSFLAGS) $(LDFLAGS)

%.o:$(ProDir)/src/%.cpp
	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDEFLAGS)

#main.o:$(ProDir)/src/main.cpp $(ProDir)/include/hello.hpp
#	$(CXX) -o $@ -c $< $(CFLAGS) $(INCLUDEFLAGS)

clean:
	rm -rf *.o test
