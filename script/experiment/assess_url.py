# -*- coding: utf-8 -*-
import requests
import time
import urllib  # 负责url编码处理
from bs4 import BeautifulSoup


url_pre = "https://blog.csdn.net/qq_35226955/article/details/"

"""list = [99986865, 98986160, 99840015, 99838149, 99705672, 99701302, 99694906, 99693508, 99692674, 99644784,
        99644122, 99643937, 99635013, 99439361, 99405019, 96313886]"""
list = [100198731, 100191007, 100178528, 100177921, 100176052, 100173434, 100170680, 100169527, 100156298, 100155011,
        100154368, 100147758, 100147758, 100145438, 100113027]

url = "https://blog.csdn.net/qq_35226955"

def get_assess_and_rank_num():
    returns = []
    response = urllib.request.urlopen(url)
    html_text = response.read().decode('utf-8')

    final = []
    bs = BeautifulSoup(html_text, "html.parser")  # 创建BeautifulSoup对象
    body = bs.body  # 获取body部分
    #print(body)
    data = body.find('div', {'class': 'grade-box clearfix'}) # 找class为'grade-box clearfix'的div
    #data = body.find('div', {'id': '7d'})  # 找到id为'7d'的div

    i = 0
    #print(data)
    dls = data.find_all('dl')  # 获取所有的'dl'
    for dl in dls:  # 对每个ul标签中的内容进行遍历
        if i == 1:  # "访问"标签是第二个
            dds = dl.find_all('dd')  # 获取所有的'dd'
            assess_num = dds[0].string  # 转为string
            # print(assess_num)
            nums = []
            for num in assess_num:
                if num.isdigit():
                    nums.append(int(num))
            #print(nums)
            nums.reverse()
            len_ = len(nums)
            assess_num, i_ = 0, 0
            for num in nums:
                assess_num += num * (10 ** i_)
                i_ += 1
                assert i_ <= len_
            returns.append(assess_num)
        if i == 3:  # "排名"标签是第四个
            dls = str(dl)
            rank= int(dls.split('"')[1])   # 详细排名在位置2
            returns.append(rank)
        i += 1
    return returns


j = 0
while True:
    for index, i in enumerate(list):
        print(index, end=" ", flush=True)
        requests.get(url_pre + str(i))
        time.sleep(3)
    time.sleep(12)
    # Assess_Rank = get_assess_and_rank_num()

    j += 1
    print("  ", j, "次循环访问已完成")
    #print("  ", j, "次循环访问已完成, 您的博客当前访问量为", Assess_Rank[0], "当前排名为：", Assess_Rank[1])



