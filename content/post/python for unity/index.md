

+++
date = "2025-01-02"
draft = false
title = "通过Process在Unity中调用**Python**"
image = "title.jpg"
categories = ["编程"]
tags = ["Unity","python","大模型","api"]

+++

# 通过Process在Unity中调用**Python**(未完成)

[TOC]



> 由于个人对Unity比较熟悉，因此打算将api接入unity实现对话的图形化...不过大部分的模型api都是通过python来调用，因此决定通过unity来实现对python脚本的调用。经过了解，有许多文案写的是通过**Python for Unity**来实现调用。然而该组件实现的脚本只能放在Editor目录下，也没办法打包，只能说完全不适配。尽管还存在**`UnityWebRequest`**等实现方式，不过决定通过**`Process`**来实现。

# 一、`Process`类简介

> [Process](https://learn.microsoft.com/zh-cn/dotnet/api/system.diagnostics.process?view=net-8.0)组件提供对计算机上正在运行的进程的访问权限。 用最简单的术语说，进程是一个正在运行的应用。 线程是操作系统分配处理器时间的基本单元。 线程可以执行进程代码的任何部分，包括当前由另一个线程执行的部分。
>
> ​														——————————文档介绍

​	`Process` 类是 C# 中 `System.Diagnostics` 命名空间下的一个类，可以管理和控制操作系统进程。其可以调用的范围包括：

- *.exe*

- *.bat / .cmd*

- *任何包含可打开该后缀的文件，比如当电脑安装了python便可以打开`.py`，当电脑安装了音乐播放器就可以打开`.mp3`。*

- *处理正在运行的进程，可以进行包括但不限于*

  - #### **获取进程信息**

  - #### **终止进程**

  - #### **重定向输入流**

  - #### **重定向输出流**

  - #### **重定向错误流**

  因此，通过打开`.py`文件作为进程，重定向输入、输出和错误，就可以实现在**Unity**中调用`python`。

# 二、`Process`语法



就要用得上的部分分析，包括以下几个类：

- **`Process`**
- **`ProcessStartInfo`**
- **`Trace`**
- **`Stopwatch`**
- **`Debug`**

其中，**`Process`**和**`ProcessStartInfo`**是刚需，剩下的作为性能优化考虑。

- ## **`Process`**

  - 
