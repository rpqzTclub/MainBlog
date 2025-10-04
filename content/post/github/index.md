

+++
date = "2025-01-02"
draft = false
title = "Github使用日志"
image = "title.png"
categories = ["编程"]
tags = ["Github","数据库","共享","学习"]

+++

# Github使用攻略

> 在部署博客的过程中由于对github缺乏使用经验，导致了很多问题。于是决定在现在完成后记录一下github的使用。



> **前排提示**：***如果你是一个萌新，建议直接看***`具体工作流示范（从`0`开始）`***部分！*** 

[TOC]

> **2020年GitHub的日志数达到了8.6亿条，活跃代码仓库达到了5,421万个，活跃开发者数达到了1,454万人，拥有超过3,100万开发人员和9,600多万个存储库。**

## Github和Git的具体概念

* 首先，github和git是两个不同的概念。GitHub 本身是一个基于web的服务平台，其通过提供git仓库的托管进行服务。而git则是开源的[分布式](https://baike.baidu.com/item/分布式/19276232?fromModule=lemma_inlink)版本控制系统，两者绝对不能混为一谈。

* git并不是github独有的。包括**Bitbucket**、**SourceForge**、**Gogs**、**Gitbucket**、**GitLab**、**Gitee**、**Azure DevOps**、**Gitea**在内的多个平台使用的都是git。

  ### git的具体概念

  > Git 是一个开源的分布式版本控制系统，用于敏捷高效地处理任何或小或大的项目。
  >
  > Git 是 Linus Torvalds 为了帮助管理 Linux 内核开发而开发的一个开放源码的版本控制软件。
  >
  > Git 与常用的版本控制工具 CVS, Subversion 等不同，它采用了分布式版本库的方式，不必服务器端软件支持。

  * **对象存储**：Git使用内容寻址文件系统来存储内容。每个文件和目录都以对象的形式存储，并通过SHA-1哈希值进行索引。

  * **分支管理**：在Git中，分支是一个引用（轻量级的分支）或是一个分支对象（重量级的分支）。分支切换实际上是改变当前HEAD指针的位置。

  * **索引（Index）**：Git的索引是一个准备区，用于暂存即将提交的文件变更。

  * **冲突解决**：当两个分支有冲突时，Git会标记出冲突的文件，需要手动解决冲突后才能进行合并。

  * **标签（Tag）**：用于标记特定的提交，通常用于版本发布。

  * **仓库（Repository）**：Git用来保存项目文件和版本历史的数据库。每个项目都有一个Git仓库。

  * **提交（Commit）**：项目文件的一个快照，包括文件的内容和元数据（如作者、日期、提交信息）。

  * **分支（Branch）**：指向特定提交的可移动的指针，用于隔离开发流程的不同部分。

  * **合并（Merge）**：将两个或多个不同的开发历史合并在一起。

  * **克隆（Clone）**：创建一个仓库的副本，包括所有文件和提交历史。

  * **远程仓库（Remote Repository）**：托管在服务器上的仓库，可以是GitHub、GitLab等。

    ### github的具体概念

    > GitHub是一个面向[开源](https://baike.baidu.com/item/开源/20720669?fromModule=lemma_inlink)及私有软件项目的托管平台，因为只支持Git作为唯一的版本库格式进行托管，故名GitHub。GitHub拥有1亿以上的开发人员，400万以上组织机构和3.3亿以上资料库。

  - **代码托管**：GitHub允许用户托管Git仓库，并提供了一个图形界面来浏览代码、提交历史、分支和标签。

  - **协作工具**：GitHub提供了issues（问题跟踪系统）、pull requests（代码审查和合并请求）、wikis（项目文档）和项目看板等工具，以支持团队协作。

  - **社交功能**：GitHub有关注（following）、星标（starring）、观察（watching）等社交功能，允许用户跟踪项目和开发者的活动。

  - **集成和自动化**：GitHub提供了API和Webhooks，允许开发者集成外部服务和自动化工作流程。

    - **代码审查和合并**：通过pull requests，GitHub支持代码审查和讨论，确保代码质量，并简化合并流程。

    ## Github的使用

    * GitHub允许你创建一个远程库。需要通过git来将本地的库同步到github中。

    * Github允许你提交一个**SSH**密钥到账号。SSH允许你无需账号密码来同步文件。

    * Github包含一个**Issues**，用于追踪项目中的错误和功能请求。可以在仓库的页面上找到New issue，填写相关信息后提交。

    * Github允许你通过**Pull Requests**来请求将某个分支的变更合并到主分支，便于代码审查。在仓库页面，点击"Pull requests"，然后点击"New pull request"，选择要合并的分支，添加更改说明后提交。

    * Github包含一个**Wikis**，可以在仓库中托管项目文档。在仓库页面，点击"Wiki"标签，然后点击"Add or edit pages"，创建或编辑文档页面。

    * Github包含**GitHub Actions**可以实现自动化部署和持续集成（CI/CD）。例如在同步仓库时自动更新readme等操作。若要使用，则需在仓库的`.github/workflows`目录下创建一个YAML文件，定义工作流程和触发条件。

    * Github还包含**Stars**（点赞/关注）、**Forks**（克隆）、**Watching**（订阅）等内容。这是一种用户间的互动。

    * Github可以创建组织，方便同步文件。登录GitHub账户，点击右上角的"+"号，选择"New organization"，填写组织信息后创建。在组织的页面，点击"Teams"，然后点击"New team"，设置团队名称和成员。

    * [^总结：Github的主要语法并没有多少，语法主要还是git的，github只是作为一个远程仓库]: 

    ## Git的使用

    * git的本地仓库包含三个部分：其一是工作目录，其保存着实际的文件。其二是暂存区，类似于缓存，保存临时改动。其三是HEAD区，指向最后一次提交的结果。

    * `git init`：初始化一个git仓库。

    * `git clone path`：克隆一个本地仓库。把**path**换成具体路径。

    * `git clone [url]`：克隆一个远程仓库。包含https克隆和SSH克隆。https的链接通常类似于这样：<u>https://github.com/username/repository.git</u>。SSH的链接通常类似于这样：<u>git clone git@github.com:username/repository.git</u>。

    * `git add <filename>`：添加文件到暂存区。如果filename为**`.`**（就是一个点）就是指当前目录下的所有文件。文件名不添加路径则是当前目录下的文件。当选择的是文件夹会递归的添加其下的所有文件。

      * `git add -u`：这个命令只添加已经跟踪的文件（即之前已经添加到Git仓库的文件），不包括新文件。
      * `git add -A` / `git add --all`：这些命令添加所有变化的文件和新文件。

    * `git commit -m "代码提交信息"`：将改动提交到HEAD。

    * `git push origin master`：将这些改动推送到远端仓库。其中，**origin**是远程仓库的默认名称，当你克隆一个远程仓库时，Git 自动将远程仓库的引用设置为 `origin`。这个名称是可替的。master是提交分支名，可以自行更改。

    * `git remote -v`：查看远程仓库的URL，`origin` 会显示在列表中。

    * `git remote add new_origin <repository_url>`：添加一个新的远程仓库，命名为**new_origin**。

    * **master**是git的默认分支。

    * `git checkout feature_x`：切换到某个分支。**feature_x**是该分支的名称。

    * `git checkout -b feature_x`：创建并切换到某个分支，**feature_x**是该分支的名称。

    * `git branch -d feature_x`：删除某个分支，**feature_x**是该分支的名称。

    * `git push origin <branch>`：推送这个分支。没有推送的分支在远程上是不可见的。

    * `git pull [remote] [branch]`:从远程仓库拉取代码变更，并尝试将这些变更自动合并到当前本地分支。`[remote]`：这是远程仓库的名称，默认是 `origin`。`branch`：这是远程仓库中你想要拉取的分支名称。如果你不指定 `[remote]` 和 `branch`，Git 会默认拉取 `origin` 远程仓库中与当前本地分支关联的分支的变更。

    * `git merge <branch>`：合并一个分支到当前分支。**branch**是该分支的名称（<>是不要的）。

    * `git diff <source_branch> <target_branch>`：预览两个分支的差异。

    * `git log`：获得提交ID。

    * `git tag 1.0.0 id`：创建一个叫做 *1.0.0* 的标签。id指提交 ID 的前 10 位字符。

    * `git checkout -- <filename>`：使用 HEAD 中的最新内容替换掉你的工作目录中的文件。

    * `git fetch [remote]`：从远程仓库获取数据，并下载远程分支的更新和提交，但不会自动合并这些更改到你的本地分支。

    * `git reset [--hard] [<commit>]`：重置当前HEAD和索引（暂存区）。`[--hard]`：这是一个可选的选项，表示重置时连同工作目录一起重置，即放弃所有本地未提交的更改。`[<commit>]`：这是一个占位符，表示你想要重置到的特定的提交（commit）。可以是一个分支名、标签或者提交的哈希值。

    * `gitk`：图形化git。

    * `git config color.ui true/false`开启/关闭彩色输出。

    * `git config format.pretty oneline`：显示历史记录时，每个提交的信息只显示一行。

    * `git add -i`：交互式添加文件到暂存区。

  - ### 具体语法实例：

  - ```bash
    # 初始化一个Git仓库
    git init
    
    # 克隆一个远程仓库到本地
    git clone https://github.com/username/repository.git
    
    # 添加文件到暂存区
    git add index.md
    
    # 添加所有变化的文件和新文件
    git add -A
    
    # 添加已经跟踪的文件（不包括新文件）
    git add -u
    
    # 提交暂存区的更改到本地仓库
    git commit -m "Add index.md with Github usage log"
    
    # 查看远程仓库的URL
    git remote -v
    
    # 添加一个新的远程仓库引用
    git remote add origin https://github.com/username/repository.git
    
    # 推送本地仓库的更改到远程仓库
    git push -u origin master
    
    # 从远程仓库拉取代码变更，并合并到当前本地分支
    git pull origin master
    
    # 合并远程分支的更改到当前分支
    git merge origin/master
    
    # 显示两个分支的差异
    git diff master feature_x
    
    # 查看提交历史
    git log
    
    # 创建一个标签
    git tag 1.0.0 <commit_id>
    
    # 检出标签对应的提交
    git checkout 1.0.0
    
    # 检出HEAD中的最新内容替换工作目录中的文件
    git checkout -- index.md
    
    # 从远程仓库获取数据，但不自动合并
    git fetch origin
    
    # 重置当前HEAD和索引（暂存区）到指定的提交
    git reset --hard <commit_id>
    
    # 重置当前HEAD和索引（暂存区）到远程分支的状态
    git reset --hard origin/master
    
    # 打开图形化Git工具
    gitk
    
    # 开启/关闭彩色输出
    git config --global color.ui true
    
    # 显示历史记录时，每个提交的信息只显示一行
    git config --global format.pretty oneline
    
    # 交互式添加文件到暂存区
    git add -i
    ```

    

  ## 身份认证

  - 身份认证主要涉及与远程仓库的交互，例如推送（push）和拉取（pull）代码。在进行这些操作时，会进行身份认证。
  - 在“Settings”页面，选择“Developer settings”。
  - 点击“Personal access tokens”，然后点击“Generate new token”。
  - 选择需要的权限范围，生成个人访问令牌。
  - 复制生成的 PAT 并妥善保管，因为之后无法再次查看完整的 PAT。
    - 生成 PAT 是为了在使用 HTTPS 认证时，可以使用 PAT 代替密码进行身份验证，提高安全性，避免在代码操作过程中频繁输入密码，同时可以为不同的用途生成不同的令牌，便于管理和权限控制。
  - 配置 Git 用户信息
    - 在本地计算机上打开终端或命令行工具.
    - 使用命令 `git config --global user.name "Your Name"` 设置全局用户名.
    - 使用命令 `git config --global user.email "your_email@example.com"` 设置全局用户邮箱.
    - <!--配置 Git 用户信息是为了在提交代码时，能够记录提交者的真实身份信息，方便项目管理和追踪代码变更的来源.-->
  - 生成 SSH 密钥对（使用 SSH 认证）
    - 生成SSH可以避免每一次登录都要输入账号密码
    - 打开终端，输入命令 `ssh-keygen -t rsa -b 4096 -C "your_email@example.com"` 生成 SSH 密钥对.
    - 按照提示操作，输入文件保存路径和密码（可选）.
    - 生成的公钥文件通常位于 `~/.ssh/id_rsa.pub`，私钥文件位于 `~/.ssh/id_rsa`.
  - 添加 SSH 公钥到 GitHub
    - 如果不添加，GitHub 无法识别本地计算机的身份，使用 SSH 认证时会出现权限拒绝的错误。
    - 登录 GitHub 账户，点击右上角的头像，选择“Settings”.
    - 在左侧菜单选择“SSH and GPG keys”，点击“New SSH key”.
    - 输入 SSH 密钥的标题，将公钥内容粘贴到“Key”框中，点击“Add SSH key”.
  - 克隆远程仓库
    - 使用 SSH URL 克隆远程仓库，例如 `git clone git@github.com:username/repository.git`.
    - 在首次克隆时，可能会提示输入 SSH 密钥的密码（如果设置了密码）.
  - 推送和拉取代码（使用 SSH 认证）
    - 在本地仓库中进行代码更改后，使用 `git push` 推送代码到远程仓库.
    - 使用 `git pull` 从远程仓库拉取代码更新.
    - 由于使用了 SSH 密钥认证，不需要输入用户名和密码.
  - 使用 HTTPS 认证（不推荐）
    - 每次操作都需要输入用户名和密码，增加了操作的复杂性，且在安全性上可能不如 SSH 认证。
    - 使用 HTTPS URL 克隆远程仓库，例如 `git clone https://github.com/username/repository.git`.
    - 在推送和拉取代码时，输入用户名和密码（或 PAT）进行身份验证.
    - 如果使用 PAT，可以在 Git 命令中输入 `username` 和 `PAT` 作为凭证.

  ## 具体工作流示范（从`0`开始）

  * ### 一、注册及相关准备工作

    * 在具体使用Github前，需要先注册一个账号。在国内由于“*多方面*”原因可能比较难以登上。如果遇到了无法登上的问题，可以考虑更换浏览器或者使用[Watt Toolkit](https://steampp.net/)
    * 访问[GitHub官网](https://github.com/)，点击右上角的[**Sign up**](https://github.com/signup?ref_cta=Sign+up)按钮，按照提示填写信息（*邮箱，密码，用户名*）创建一个新的GitHub账户。
    * 登录到GitHub账户后，点击右上角的**+**按钮，选择**New repository**。
    * 在创建仓库界面，要注意的选项有以下几项，根据自己的需求来决定：
      * **Repository name**（必填）：这个项会决定仓库的名称。
      * **Public/Private**（默认public）：选择可以决定仓库是否公开，也就是别人能不能看到。
      * **Add a README file**（可选）：可以在仓库中加入一个介绍的文本，方便写项目介绍、更新日志之类的东西。
      * **Add .gitignore**（可选）：创建一个上传选择文件，可以在仓库上传时选择性的忽略一部分文件。
      * **Choose a license**（可选）：可以决定别人要怎么对待你这个项目。简单来说可以从以下几个选择一个，具体的可以查[Github许可证使用手册中文版](http://choosealicense.online/)：
        * **MIT许可证**：最宽松，只要保留版权和许可声明，就可以随意使用、修改和分发代码，适合希望代码被广泛传播的个人或小项目。
        * **Apache许可证**：也很宽松，和MIT类似，但多了专利保护条款，适合大型项目，尤其是涉及多个开发者和组织合作的项目。
        * **GPL许可证**：要求修改后的代码也必须开源，适合希望保持项目开源精神，防止代码被闭源的项目。
        * **BSD许可证**：比较宽松，允许自由使用和修改代码，但需保留版权声明，适合希望代码能被广泛应用，包括商业用途的项目。
  
  * ### 二、本地git下载和配置
  
    * 通过链接简单安装git到电脑上：[git windows版下载地址](https://git-scm.com/downloads/win) 
  
    * 配置你的用户名和邮箱，这个会决定你提交更改时显示的用户信息。输入`win + R`,输入`cmd`并打开。
  
    * 在其中输入以下代码。注意将**Your Name**换成你自己的用户名，**your_email@example.com**换成你自己的邮箱地址：
  
      * ```bash
        git config --global user.name "Your Name"
        git config --global user.email "your_email@example.com"
        ```
  
  * ### 三、身份认证
  
    * 上传代码到仓库时需要身份认证。如果每一次都输入账号密码就很麻烦，所以采用**SSH密钥认证**来跳过这个步骤。
  
    * 在`cmd`中输入以下片段。将`your_email@example.com`替换成你在GitHub上注册的邮箱地址：
  
      * ```bash
        ssh-keygen -t rsa -b 4096 -C “your_email@example.com”
        ```
  
        
  
    * 登录GitHub，点击右上角头像，选择**Settings**。
  
    * 在左侧菜单中选择**SSH and GPG keys**，点击**New SSH key**。
  
    * 在**Title**字段中，为SSH密钥添加名字；在**Key**字段中，将`id_rsa.pub`文件的内容复制粘贴进去，然后点击**Add SSH key**确认添加。
  
    * 继续在`cmd`中输入，将**Your Name**改为你的用户名，将**your_email@example.com**改为你的邮箱：
  
      * ```bash
        git config --global user.name "Your Name"
        git config --global user.email "your_email@example.com"
        ```
  
        
  
  * ### 四、下载仓库和拉取分支（更改）到本地
  
    * 打开你要找的仓库，找到界面中的***`< > Code`***选项，打开复制链接。
  
    * 打开`cmd`，输入以下代码，这会将项目下载到指定目录下：
  
      * ```bash
        cd <你要的具体路径>
        git clone <刚才复制的链接>
        ```
  
    * 拉取分支通过以下操作。如果你不知道分支是什么，可以理解为延申出去的不同方面的更改：
  
      * ```bash
        #打开到指定目录
        cd <仓库路径>
        #切换到你要的指定分支
        git checkout main
        #拉取指定分支
        git pull origin <分支名称>
        #拉取当前分支的最新内容
        git pull
        ```
  
        
  
  * ### 五、 上传文件到仓库
  
    * 首次上传时，打开`cmd`，执行以下操作：
  
      * ```bash
        cd <你要上传的文件路径>
        #初始化git，这会创建一个.git隐藏目录，用于存储Git的元数据和对象数据库
        git init
        #创建一个分支，用于提交，并切换到这个分支上
        git checkout -b <分支名称>
        #选择你要上传到的仓库
        git remote add origin <仓库链接>
        #决定要上传的文件
        git add <fileName>
        #如果你希望上传当前目录所有文件，用这个
        git add .
        #先提交到本地仓库上
        git commit -m "提交信息"
        #发送分支到目标仓库
        git push -u origin new-branch-name
        ```
  
    * 在要更新时，可以简单的这么做：
  
      * ```bash
        git add .
        git commit -m "Update"
        git push
        ```
  
        




  ## 总结

  * Git 作为一个开源的分布式版本控制系统，以其高效和灵活性被广泛应用于各种项目中。而 GitHub，作为一个基于 Web 的服务平台，提供了 Git 仓库托管和丰富的协作工具，极大地方便了开发者之间的代码共享和项目管理。
  * 本篇涉及了了 Git 的核心特性，和基础语法，大概阐释了涉及到的概念。可以作为我自身的查档用，也有一定的参考价值。
  * GitHub 的主要语法和操作实际上基于 Git，而 GitHub 则作为远程仓库的角色，使得代码的远程托管和管理变得更加便捷。希望本文能帮助您更好地理解和使用 Git 和 GitHub，提高您的工作效率，并在开源社区中发挥更大的作用。

  ## 相关资料

  	* [git - 简明指南](https://www.runoob.com/manual/git-guide/)
  	* [Git 教程|菜鸟编程](https://www.runoob.com/git/git-tutorial.html)
  	* [Github-百度百科](https://baike.baidu.com/item/github/10145341)
  	* [Git-百度百科](https://baike.baidu.com/item/GIT/12647237)
  	* [Git官方文档](https://git-scm.com/)
  	* [Git维基百科](https://en.wikipedia.org/wiki/Git)
  	* [Github维基百科](https://en.wikipedia.org/wiki/GitHub)

  

