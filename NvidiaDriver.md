# Nvidia driver install
## 打开终端，先删除旧的驱动

	sudo apt-get purge nvidia*
## 禁用自带的 nouveau nvidia驱动
	
>sudo vim /etc/modprobe.d/blacklist-nouveau.conf

>blacklist nouveau

>options nouveau modeset=0

再更新一下
	sudo update-initramfs -u
修改后需要重启系统。确认下Nouveau是已经被你干掉，使用命令

	lsmod | grep nouveau
##　重启系统至init 3文本模式, 也可先进入图形桌面再运行init 3进入文本模式, 再安装下载的驱动就无问题

首先我们需要结束x-window的服务，否则驱动将无法正常安装

关闭X-Window，很简单
	sudo service lightdm stop
然后切换tty1控制台：Ctrl+Alt+F1即可
## 接下来就是最关键的一步了
	sudo sh ./NVIDIA.run
开始安装，安装过程比较快，根据提示选择即可最后安装完毕后，重新启动X-Window
	sudo service lightdm start
然后Ctrl+Alt+F7进入图形界面

	nvidia-smi

	nvidia-settings
## 注意事项
Nvidia driver cannot be installed with OpenGL.
How to solve it:

enter Ctrl+Alt+F1

### Uninstall any previous drivers
	sudo apt-get remove nvidia-*
	sudo apt-get autoremove
### Uninstall the drivers from the .run file
	sudo nvidia-uninstall
we can login normally.

Follow the instruction above(re-do it), and then
	sudo service lightdm stop
	sudo sh NVIDIA-Linux-x86_64-381.22.run -no-x-check -no-nouveau-check -no-opengl-files
	sudo service lightdm restart



