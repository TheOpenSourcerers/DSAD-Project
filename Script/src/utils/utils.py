import os
from pathlib import Path
from src.utils.libs import *

INPUT_FILE_PATH = (Path(__file__).parent / "../../../Data/In/Data.csv").resolve()
OUTPUT_FILE_PATH = (Path(__file__).parent / "../../../Data/Out/").resolve()

def custom_print(title: str = "", printable: any = "", final_line: bool = False, round: bool = False):
  if len(title) == 0:
    print("="*83)
    return
  length = 83-len(title)-2
  size = int(length/2)
  print(f"{"="*size} {title} {"="*(length-size)}")
  if round:
    printable = printable.round(2)
  print(printable)
  if final_line:
    print("=" * 83)
  print("\n")

def join(input, new_line: bool = True):
  string = ", "
  if new_line:
    string = "\n"
  return string.join(input)

def getPath(name):
  return (OUTPUT_FILE_PATH / name).resolve()

if not os.path.isdir(OUTPUT_FILE_PATH):
    os.makedirs(OUTPUT_FILE_PATH)

if not os.path.isdir(getPath("PCA")):
  os.makedirs(getPath("PCA"))

def KMO_Interpretation(index):
  if 0.00 <= index < 0.50:
        return "Unacceptable"
  elif 0.50 <= index < 0.60:
      return "Miserable"
  elif 0.60 <= index < 0.70:
      return "Mediocre"
  elif 0.70 <= index < 0.80:
      return "Middling"
  elif 0.80 <= index < 0.90:
      return "Meritorious"
  elif 0.90 <= index <= 1.00:
      return "Marvelous"
  else:
      return ""

def Barlett_Interpretation(chi2, pval):
    if pval < 0.05:
        return "Reject the null hypothesis"
    else:
        return "Fail to reject the null hypothesis"