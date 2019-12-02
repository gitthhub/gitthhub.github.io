> 以下内容根据个人理解整理而成，如有错误，欢迎指出，不胜感激。
### 0. 写在前面
本文是《C++ Primer Plus 第六版》16.2节关于智能指针的阅读笔记，主要总结auto_ptr、unique_ptr和shared_ptr这三种智能指针之间的联系和区别，这里仅记录主要概念，更为具体的代码示例可参考书籍16.2节。

### 1. 什么是智能指针
对普通指针，一旦使用new分配了内存，则程序结束后需要程序员手动调用delete来释放相应内存，如果忘记删除，则该部分内存无法释放，会造成内存泄漏。
```
std::string *ps = new std::string('abc');
...
delete ps;
```
智能指针的出现则是为了解决这一问题，它可以保证在删除指针变量ps的同时释放它指向的内存。
事实上，智能指针是行为类似于指针的类对象，这句话指明了智能指针其实是一个类的实例，这也就意味着它有自己的析构函数，可以在该对象过期时使用相应的析构函数释放指向的内存，而不需要再手动调用delete。

C++中常用的有以下三种智能指针，它们包含在头文件`memory`中：
* auto_ptr: C++98中提供，C++11已摒弃
* unique_ptr: C++11提供
* shared_ptr: C++11提供

其使用方式也很简单，以auto_ptr为例：
```
std::auto_ptr<std::string> ps (new std::string('abc'))
```

### 2. 为什么有多种智能指针
（智能指针一共有4种，这里不讨论weak_ptr）
在C++98中，仅有auto_ptr这一种智能指针，为什么后来把它舍弃并引入另外两种智能指针？
使用下面的例子来说明：
```
std::auto_ptr<std::string> ps (new std::string('abc'))
std::auto_ptr<std::string> pr;
pr = ps;
```
上述例子中出现了指针之间的赋值操作，如果ps和pr都是普通指针，它们将指向同一块内存，如果先释放了ps指向的内存，当再释放pr指向的内存显然会出错。
对智能指针，有如下两种策略来解决这个问题：
* 建立所有权的概念：对于一个特定的对象，如`new std::string('abc')`，只有一个智能指针可以拥有它，可对它执行删除操作。当指针间相互赋值时，相当于所有权也在转让，如上述例子中ps失去对象的所有权。这是用于auto_ptr和unique_ptr的策略，但unique_ptr的策略更为严格。
* 引用计数：对于一个特定的对象，如`new std::string('abc')`，可以有多个智能指针拥有它，同时记录拥有它的智能指针的个数，仅当最后一个智能指针过期时才释放对象。这是shared_ptr的策略。

unique_ptr比auto_ptr策略更为严格的地方在于：在遇到一个非临时unique_ptr赋给另一个时(如果是临时unique_ptr赋给另一个，由于临时unique_ptr随即被销毁，所以允许这种赋值，如函数返回的unique_ptr赋给相应变量)，编译器会直接报错，将安全隐患消除在编译阶段。
当然，如果非要执行unique_ptr的相互赋值操作，C++标准库函数std::move()可以安全地完成该操作。
unique_ptr相比auto_ptr的另一个优点是，它有一个可用于数组的变体(有使用new[] 和delete[] 的版本):
```
std::unique_ptr<double[]> pda(new double(5));
```

### 3. 如何选择智能指针
根据shared_ptr和unique_ptr的特性选择即可：
* 如果需要多个指针指向同一对象，则使用shared_ptr，否则可选择unique_ptr。
