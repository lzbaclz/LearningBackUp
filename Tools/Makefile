
# tutorial introduction: https://youtu.be/_r7i5X0rXJk

# target: dependencies
# 	action

output: main.o hello.o
	@echo ">>> Compiling" $< "<<<"
	g++ main.o hello.o -o output

main.o: main.cpp
	g++ -c main.cpp

hello.o: hello.cpp hello.h
	g++ -c hello.cpp

clean: 
	rm *.o output