![1729033516631](image/白嫖A100-git/1729033516631.png)

申明 以下部分内容来源于活动教学文档：

# [Docs](https://aicarrier.feishu.cn/wiki/Q7CSwHcDfiSCaxkpqifcxu4ynqJ "Docs")

# git 安装

是一个开源的分布式版本控制系统，被广泛用于软件协同开发。程序员的必备基础工具。

![](https://i-blog.csdnimg.cn/direct/5c087b3661914c489c971e6fcf6748a3.png)![]()**编辑**

1. ## 常用的 Git 操作
2. `git init`

* 初始化一个新的 Git 仓库，在当前目录创建一个 `.git` 隐藏文件夹来跟踪项目的版本历史。

1. `git clone <repository-url>`

* 从指定的 URL 克隆一个远程仓库到本地。

1. `git add <file>` 或 `git add.`

* 将指定的文件或当前目录下的所有修改添加到暂存区，准备提交。

1. `git commit -m "message"`

* 提交暂存区的修改，并附带一个有意义的提交消息来描述更改的内容。

1. `git status`

* 查看工作目录和暂存区的状态，包括哪些文件被修改、添加或删除。

1. `git log`

* 查看提交历史，包括提交的作者、日期和提交消息。

1. `git branch`

* 列出所有本地分支。

1. `git branch <branch-name>`

* 创建一个新的分支。

1. `git checkout <branch-name>`

* 切换到指定的分支。

1. `git merge <branch-name>`

* 将指定的分支合并到当前分支。

1. `git push`

* 将本地的提交推送到远程仓库。

1. `git pull`

* 从远程仓库拉取最新的更改并合并到本地分支。

1. `git stash`

* 暂存当前未提交的修改，以便在需要时恢复。

1. `git stash pop`

* 恢复最近暂存的修改。
  # 作业
* ## 破冰活动：自我介绍
* 命名<camp3_`<id>`.md>
* 路径：./data/Git/task/
* 【大家可以叫我】可以为github昵称，微信昵称等，或其他网名
* 作业提交对应的PR链接
* ### 任务

  提交自己的破冰介绍.md

  ### 要求
* 命名<camp3_`<id>`.md>
* 路径：./data/Git/task/
* 【大家可以叫我】可以为github昵称，微信昵称等，或其他网名
* 作业提交对应的PR链接

# 完成作业：

连接：[https://github.com/Aorg/Assignment/blob/master/camp3_974.md](https://github.com/Aorg/Assignment/blob/master/camp3_974.md "https://github.com/Aorg/Assignment/blob/master/camp3_974.md")

![](https://i-blog.csdnimg.cn/direct/b9710e4a09de46089e898edf24a0f7f3.png)![]()**编辑**

## 完成步骤：

### 1. 创建自我介绍文件

    id 是问卷星id

    按照[camp3_id.md](https://github.com/Aorg/Assignment/blob/master/camp3_974.md "camp3_id.md") 格式命名

    比如我的[camp3_974.md](https://github.com/Aorg/Assignment/blob/master/camp3_974.md "camp3_974.md")

### 2.连接自己的仓库

    生成自己的ssh keys，cat查看， 给仓库

![](https://i-blog.csdnimg.cn/direct/5d8c5ac35094441ab899eca4f166ebe1.png)![]()**编辑**

![](https://i-blog.csdnimg.cn/direct/1615247e466442e0a51f41e38c2ef4f6.png)![]()**编辑**

 ![](https://i-blog.csdnimg.cn/direct/27db5006d4f443ca846d0b8f4a62b404.png)![]()**编辑**

连接好后 在本地设置你的github账号

如果不想重新创建库，可以用一下代码 添加新文件到github

### 3.上传

示例代码：

```bash
git init
git config --global user.name "帐号名"
git config --global user.email "账号邮箱"
git remote add origin 准备上传的仓库地址
git add camp3_974.md #改成你准备上传的的文件名 
git commit -m "camp3_974.md"  #引号记得改成你的文件名
git push origin master #上传
```

![]()

detail:

![](https://i-blog.csdnimg.cn/direct/967fe2e112264ba9bf9ee89b1ead56c1.png)![]()**编辑**

上图最后一段是git上传成功提示

查看仓库

![](https://i-blog.csdnimg.cn/direct/9b46eb69831048d7b1911a9dd07ca983.png)![]()**编辑**

参考：

[Git 如何在不先克隆整个远程 Git 仓库（Github）的情况下添加文件|极客教程](https://geek-docs.com/git/git-questions/41_git_how_do_i_add_file_to_remote_git_repo_github_without_cloning_the_whole_repo_first.html "Git 如何在不先克隆整个远程 Git 仓库（Github）的情况下添加文件|极客教程")
