---
title: 工程推送至github流程
date: 2024-07-26 17:42:42
tags: github学习
---
## 步骤：
1.github创建一个与工程名同名的仓库，不要勾选任何可选项

2.进入工程名目录

3.右键打开git bash

4.运行以下代码：

	git init
	touch README.md
	git add README.md
	git commit -m "first commit"
	git branch -M main
	git remote add origin git@github.com:wu-diu-diu/仓库名.git
	git push -u origin main

main可以换成master

**本步骤适用于将一个完善了的project上传至github的情况**