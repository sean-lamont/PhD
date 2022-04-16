import os
import re
import sys
import json

s = []

with open("ss.txt","r") as f:
    s = f.read()

s = re.sub(' +',' ', s)

l1 = re.findall("list.rewrite:(.*?),", s.replace('\n',''))

l2 = re.findall("rich_list.rewrite:(.*?),", s.replace('\n',''))

delsimps_list = re.sub("\'", "\"", str(l1))

delsimps_richlist = re.sub("\'", "\"", str(l2))

print(delsimps_list)

print(delsimps_richlist)
