#!/usr/bin/python3
import argparse
import json

parser = argparse.ArgumentParser(description = \
                    "it create kaggle kernel-metadata.json")

parser.add_argument("title", help = "title of kernel")
parser.add_argument("--code_file", help = "name of file that has code",
                        default = 'file_kaggle.py')
parser.add_argument("-kt", "--kernel_type", default = "script",
             help = "type of kernel either script or notebook [default script]")
parser.add_argument("--is_private", default = "true",
             help = "made it private or public [default private]")
parser.add_argument("-gpu", "--enable_gpu", default = "true",
             help = "to enable or disable gpu [default True]")
parser.add_argument("--enable_internet", default = "false",
            help = "to make kernel can download libraries [default False]")
args = parser.parse_args()


title = args.title
id_str = title.replace(' ', '-')

json = """{{
  "id": "abdulrahmanfarid/{}",
  "title": "{}",
  "code_file": "{}",
  "language": "{}",
  "kernel_type": "{}",
  "is_private": "{}",
  "enable_gpu": "{}",
  "enable_internet": "{}",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}}""".format(id_str, title,  args.code_file,  "python", args.kernel_type,
 args.is_private, args.enable_gpu,args.enable_internet)

with open('kernel-metadata.json', 'w') as kernel:
   kernel.write(json)

