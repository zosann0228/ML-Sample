
obj=unit.o layer.o

test: $(obj) test.o
	gcc -o $@ $^
