LDFLAGS = -lglfw -lvulkan

all: neurons

neurons: clean validateCompute
	g++ $(CFLAGS) -o build/main/neurons $(LDFLAGS) main.cpp

.PHONY: clean debug validateCompute

validateCompute:
	glslangValidator -V -g -o build/debug/comp.spv compute.comp
	glslangValidator -V -g -o build/main/comp.spv compute.comp

clean:
	rm -f build/main/comp.spv
	rm -f build/main/neurons
	rm -f build/debug/comp.spv
	rm -f build/debug/neurons


debug: clean validateCompute
	g++ $(CFLAGS) -Wall -g -D DEBUG -o build/debug/neurons $(LDFLAGS) main.cpp

run: neurons
	./neurons
