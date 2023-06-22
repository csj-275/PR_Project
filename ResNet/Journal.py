import os
import datetime


class Journal:
    def __init__(self, Class, Version, seed, t_num, batch_size, lr, momentum, layer):
        # 获取当前日期
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        # 设置文件名和后缀
        # Version = str(Version)
        name = Class + '-ResNet' + str(layer) + '_V' + Version + '日志'
        filepath = 'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class + '_ResNet' + str(layer) + '_V' + Version + '/'
        self.filename = filepath + name + ".txt"

        with open(self.filename, "a") as f:
            f.write(f'{date}\n')
            f.write('《' + Class + '-ResNet' + str(layer) + '_V' + Version + '日志》' + "\n\n"
                    + '训练集占比' + str(t_num) + "\n"
                    + '随机种子' + str(seed) + "\n"
                    + '训练批次大小' + str(batch_size) + "\n"
                    + '学习率' + str(lr) + "\n"
                    + '学习率' + str(momentum) + "\n\n")
        print(f"文件 {self.filename} 创建成功！")

    def write_journal(self, content):
        # 指定要写入文件的内容
        # content = "这是要写入文件的内容"
        # 打开文件并写入内容
        with open(self.filename, "a") as f:
            for line in content.splitlines():
                f.write(line + "\n")


