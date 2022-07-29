# Use gcc/g++ to compile a c/cpp file

## Procedure
1. 预处理（Preprocessing）

g++  -E  test.cpp  -o  test.i //生成预处理后的.i文件

2. 编译（Compilation）

g++ -S test.i -o test.s //生成汇编.s文件

3. 汇编（Assembly）

g++  -c  test.s  -o  test.o    //生成二进制.o文件

4. 链接（Linking）

g++ test.o  -o  test.out      //生成二进制.out可执行文件 


## Command format

gcc [-c|-S|-E] [-std=standard]
    [-g] [-pg] [-Olevel]
    [-Wwarn...] [-pedantic]
    [-Idir...] [-Ldir...]
    [-Dmacro[=defn]...] [-Umacro]
    [-foption...] [-mmachine-option...]
    [-o outfile] [@file] infile...



## Official option

> official mannual: <https://gcc.gnu.org/onlinedocs/gcc-6.1.0/gcc.pdf>

> use command: man gcc/g++


> ref: <https://cloud.tencent.com/developer/article/1176744>


--- 

## Makefile tutorial

> ref: https://seisman.github.io/how-to-write-makefile/

> ref: https://github.com/seisman/how-to-write-makefile

#### Code file structure
- build 
    - bin --> .out
    - objects --> .o
- src --> .cpp / .c / .h
- Makefile
- main.out

#### Makefile grammar rules

- $@: 目标文件
- $^: 所有依赖文件
- $<: 第一个依赖文件
- Compilation grammar
```
target: prerequisites
    command
```

0. 定义变量
    - 文件路径
    - 编译命令参数
    - 最终目标文件

```shell
dest_dir = build
src_dir = src
obj_dir = $(dest_dir)/objects
bin_dir = $(dest_dir)/bin

CC = g++
RESULT = main
CFLAGS = -Wall -O3 -std=c++14
CFILES = A.cpp B.cpp
ofiles = $(CFILES:%.cpp=$(obj_dir)/%.o)

program = $(bin_dir)/$(RESULT)
$(program): $(ofiles)
```


1. 产生 . o 文件
    - 第一句：输出编译信息
    - 第二句：如果不存在路径上的文件夹，则创建文件夹
    - 第三句：执行 g++ -c xxx.c 命令

```shell
# src 中所有 cpp 文件
$(obj_dir)/%.o: $(src_dir)/%.cpp
	@echo ">>> Compiling" $< "<<<"
	@if [ ! -d $(obj_dir) ]; then mkdir -p $(obj_dir); fi;
	$(CC) $(CFLAGS) -c $< -o $@

# src 所有子文件夹中 cpp 文件
$(obj_dir)/%.o: $(src_dir)/*/%.cpp
	@echo ">>> Compiling" $< "<<<"
	@if [ ! -d $(obj_dir) ]; then mkdir -p $(obj_dir); fi;
	$(CC) $(CFLAGS) -c $< -o $@

```


2. 将所有 .o 文件链接在一起生成可执行文件，并在最外层生成可执行文件的链接文件
- 第一句：输出编译信息
- 第二句：如果不存在路径上的文件夹，则创建文件夹
- 第三句：执行 g++ -o 目标可执行文件 xxx1.o xxx2.o ... 命令
- 第四句：将目标可执行文件链接到最外层

```shell
$(bin_dir)/%:
	@echo ">>> Linking" $@ "<<<"
	@if [ ! -d $(bin_dir) ]; then mkdir -p $(bin_dir); fi;
	$(CC) -o $@ $^
	ln -sf $@ $(notdir $@)

```

3. make cleann 命令
    - 当我们需要重新编译时，我们想要快速地删除上一次编译所产生的文件
    - 我们可以自定义 make clean 命令，实现对上一次编译生成文件的快速清理

```shell
.PHONY: clean
clean:
	rm -rf $(dest_dir)
	rm -f $(RESULT)
```


#### MacOS 下编译 C/C++ 工程的固定模板

> 每次需要编译，则先执行 make clean，再执行 make。增加需要编译的文件: 在 CFILES 变量后面继续添加即可

```shell
# $@: 目标文件, $^: 所有依赖文件, $<: 第一个依赖文件
# 语法规则:
# 	targets: prerequisites
# 		command
dest_dir = build
src_dir = src
obj_dir = $(dest_dir)/objects
bin_dir = $(dest_dir)/bin

CC = g++
RESULT = main
CFLAGS = -Wall -O3 -std=c++14
CFILES = A.cpp B.cpp
ofiles = $(CFILES:%.cpp=$(obj_dir)/%.o)

program = $(bin_dir)/$(RESULT)
$(program): $(ofiles)

$(bin_dir)/%:
	@echo ">>> Linking" $@ "<<<"
	@if [ ! -d $(bin_dir) ]; then mkdir -p $(bin_dir); fi;
	$(CC) -o $@ $^
	ln -sf $@ $(notdir $@)

$(obj_dir)/%.o: $(src_dir)/%.cpp
	@echo ">>> Compiling" $< "<<<"
	@if [ ! -d $(obj_dir) ]; then mkdir -p $(obj_dir); fi;
	$(CC) $(CFLAGS) -c $< -o $@

$(obj_dir)/%.o: $(src_dir)/*/%.cpp
	@echo ">>> Compiling" $< "<<<"
	@if [ ! -d $(obj_dir) ]; then mkdir -p $(obj_dir); fi;
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf $(dest_dir)
	rm -f $(RESULT)
```




