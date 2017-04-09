
PRODIR=$(shell pwd)

SOURCES=$(notdir $(wildcard $(PRODIR)/src/*.cpp))
OBJS=$(SOURCES:.cpp=.o)

INCLUDE=-I$(HOME)/caffe/include \
		-I$(HOME)/caffe/build/src \
		-I$(PRODIR)/include \
		-I/usr/include

LIBSDIR=-L$(HOME)/caffe/build/lib \
		-L$(PRODIR)/build \
		-L/usr/lib/x86_64-linux-gnu

LDFLAGS= -lcaffe -lopencv_core -lopencv_imgproc -lopencv_highgui
		
CXXFLAGS += -g -Wall -std=c++11 -shared -fPIC -D CPU_ONLY


test:$(OBJS)
	@echo $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(LIBSDIR) $(LDFLAGS)

%.o:$(PRODIR)/src/%.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCLUDE)

clean:
	rm -rf test *.o
