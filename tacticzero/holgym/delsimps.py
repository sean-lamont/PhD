import os
import re
import sys
import json

scripts = "listScripts.txt"

with open(scripts,"r") as f:
    s = f.read()


# l1 = re.findall("export_rewrites \[\"(.*?)\"\]", s)

# TypeBase.export [list_info\']\n  end;\n\nval _ = export_rewrites\n          ["APPEND_11",\n           "MAP2", "NULL_DEF",\n           "SUM", "APPEND_ASSOC", "CONS", "CONS_11",\n           "LENGTH_MAP", "MAP_APPEND",\n           "NOT_CONS_NIL", "NOT_NIL_CONS",\n           "CONS_ACYCLIC", "list_case_def",\n           "ZIP", "UNZIP", "ZIP_UNZIP", "UNZIP_ZIP",\n           "LENGTH_ZIP", "LENGTH_UNZIP",\n           "EVERY_APPEND", "EXISTS_APPEND", "EVERY_SIMP",\n           "EXISTS_SIMP", "NOT_EVERY", "NOT_EXISTS",\n           "FOLDL", "FOLDR", "LENGTH_LUPDATE",\n           "LUPDATE_LENGTH"];


# c = re.compile("export_rewrites\n +\[(.*?)\]", re.MULTILINE)

s = re.sub(' +',' ', s)

l = re.findall("export_rewrites +\[(.*?)\]", s.replace('\n',''))

ll = [re.sub('\"','', i) for i in l]

pre = [s.split(", ") for s in ll]

ss = [s[0] for s in pre if isinstance(s, list)]

rws = re.sub("\'", "\"", str(ss))

m = re.findall("\nTheorem (.*?)\[simp\]:\n", s)

n = re.findall("\"(.*?)\[simp\]\"", s)

simps = n + m

delsimps = re.sub("\'", "\"", str(ss+simps))
