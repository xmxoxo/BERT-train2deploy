#!/usr/bin/env python3
#coding:utf-8

"""
 @Time    : 2019/2/28 15:31
 @Author  : xmxoxo (xmhexi@163.com)
 @File    : tran_service.py
"""
import argparse
import flask
import logging
import json
import os
import re
import sys
import string
import time
import numpy as np


from bert_base.client import BertClient


# 切分句子
def cut_sent(txt):
    #先预处理去空格等
    txt = re.sub('([　 \t]+)',r" ",txt)  # blank word
    txt = txt.rstrip()       # 段尾如果有多余的\n就去掉它
    nlist = txt.split("\n") 
    nnlist = [x for x in nlist if x.strip()!='']  # 过滤掉空行
    return nnlist


#对句子进行预测识别
def class_pred(list_text):
    #文本拆分成句子
    #list_text = cut_sent(text)
    print("total setance: %d" % (len(list_text)) )
    with BertClient(ip='192.168.15.111', port=5575, port_out=5576, show_server_config=False, check_version=False, check_length=False,timeout=10000 ,  mode='CLASS') as bc:
        start_t = time.perf_counter()
        rst = bc.encode(list_text)
        print('result:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))
    #返回结构为：
    # rst: [{'pred_label': ['0', '1', '0'], 'score': [0.9983683228492737, 0.9988993406295776, 0.9997349381446838]}]
    #抽取出标注结果
    pred_label = rst[0]["pred_label"]
    result_txt = [ [pred_label[i],list_text[i] ] for i in range(len(pred_label))]
    return result_txt

def flask_server(args):
    pass
    from flask import Flask,request,render_template,jsonify

    app = Flask(__name__)
    #from app import routes

    @app.route('/')
    def index():
        return render_template('index.html', version='V 0.1.2')
  
    @app.route('/api/v0.1/query', methods=['POST'])
    def query ():
        res = {}
        txt = request.values['text']
        if not txt :
            res["result"]="error"
            return jsonify(res)
        lstseg = cut_sent(txt)
        print('-'*30)        
        print('结果,共%d个句子:' % ( len(lstseg) ) )
        for x in lstseg:
            print("第%d句：【 %s】" %(lstseg.index(x),x))
        print('-'*30)        
        if request.method == 'POST' or 1:
            res['result'] = class_pred(lstseg)
        print('result:%s' % str(res))
        return jsonify(res)


    app.run(
        host = args.ip,     #'0.0.0.0',
        port = args.port,   #8910,  
        debug = True 
    )


def main_cli ():
    pass
    parser = argparse.ArgumentParser(description='API demo server')
    parser.add_argument('-ip', type=str, default="0.0.0.0",
                        help='chinese google bert model serving')
    parser.add_argument('-port', type=int, default=8910,
                        help='listen port,default:8910')

    args = parser.parse_args()

    flask_server(args)

if __name__ == '__main__':
    main_cli()



