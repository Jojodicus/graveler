.PHONY: all, clean

NVCXX ?= nvcc
NVCXXFLAGS ?= -O3 -arch=native -m64
BINNAME ?= graveler

all: graveler

clean:
	rm -f $(BINNAME)

graveler: graveler.cu colors.h
	$(NVCXX) $(NVCXXFLAGS) $(DEFINES) -o $(BINNAME) $<
