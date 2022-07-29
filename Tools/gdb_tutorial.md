# GNU gdb tutorial

> ref: [introduction to GDB a tutorial - Harvard CS50 (vedio)](https://youtu.be/sCtY--xRUyI)

> ref: [GDB Tutorial (vedio)](https://youtu.be/svG6OPyKsrw)

> ref: [GNU Debugger Tutorial](https://www.tutorialspoint.com/gnu_debugger/index.htm)


- run 
- break
- next
- list
- print
- quit
- up
- down
- display
- undisplay + [display_id]
- backtrace --> call stack
- step (go inside the function)
- continue (run into the breakpoint at anywhere of the program)
- finish








> [The LLDB Debugger](https://lldb.llvm.org/)

- break (b) - 设置断点，也就是程序暂停的地方
- run (r) - 启动目标程序，如果遇到断点则暂停
- step (s) - 进入下一条指令中的函数内部
- backtrace (bt) - 显示当前的有效函数
- frame (f) - 默认显示当前栈的内容，可以通过 `frame arg` 进入特定的 frame（用作输出本地变量）
- next (n) - 运行当前箭头指向行
- continue (c) - 继续运行程序直到遇到断点。

---

## CPP 复习

> ref: https://zhuanlan.zhihu.com/p/516819910



---

## Makefile 编译工具



--- 

## gcc 编译

gcc -Wall -g -o dot dot.c
 
其中，-Wall 代表编译器在编译过程中会输出警告信息（Warning），比如有些变量你并没有使用，指针指向的类型有误，main 函数没有返回整数值等。这类信息虽然不是错误，不影响编译，但是很可能是程序 bug 的源头，也有助于你寻找代码中的错误，规范代码格式。所以建议每次编译时都加上 -Wall 参数。
  
-g 代表编译器会收集调试（debug）信息，这样如果你的程序运行出错，就可以通过 gdb 或者 lldb 等工具进行逐行调试，方便找出错误原因。如果你不是百分之百确定你的程序毫无问题，建议加上 -g 参数。这样 debug 的时候会方便很多。

-o 代表编译器会将编译完成后的可执行文件以你指定的名称输出到你指定的文件夹下。-o 的空格后的名称就是输出的文件的名称。例如我这里 -o 后是 dot，就是说 gcc 会在编译成功后在我的当前目录下生成一个叫 dot 的可执行文件。如果不加这个参数，每次编译后生成的可执行文件都会放在根目录下，名字叫做 a.out。每次编译成功后都会把上一次的 a.out 文件覆盖。所以建议加上 -o 参数，这样可以更加条理。

最后一项便是你要编译的的源代码的名称了。我这里是 dot.c。注意加上后缀 .c。

如果提示你 Permission Denied，则是因为你没有赋予这个文件执行权限，需要在终端里输入

chmod u+x dot （这里是你自己的文件名）