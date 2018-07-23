libmtsi : textsynthinterface.cpp
	g++ textsynthinterface.cpp `pkg-config --cflags --libs mtsynth` -shared -fPIC -Wno-write-strings -lpthread -o libmtsi.so
