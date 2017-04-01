
ProjectDir=$(shell pwd)/..

HeadFiles=$(wildcard $(ProjectDir)/include/*hpp)
Objects=$(addprefix $(ProjectDir)/,$(subst *.hpp,*o,$(HeadFiels)))

INCLUDE=-I$(HOME)/caffe/include \
		-I$(ProjectDir)/include \
		-I/usr/include

LIBSDIR=-L$(HOME)/caffe/build/lib \
		-L$(ProjectDir)/build \
		-L/usr/lib/x86_64-linux-gnu 

LDFLAGS=-l:libpencv_core.so.2.4 -l:libopencv_imgproc.so.2.4 -l:libopencv_highgui.so.2.4 \
		-l:libcaffe.so
CXXFLAGS += -Wall -std=c++11


test:$(Objects)
	(CXX) -o $@ $^ $(LIBSDIR) $(LDFLAGS)

$(Projects)/%.o:%.cpp
