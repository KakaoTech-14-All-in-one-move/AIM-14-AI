# pip install olefile

import olefile
import zlib
import struct
import re
from docx import Document


def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not Valid HWP.")

    header = f.openstream("FileHeader")
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    nums = []
    for d in dirs:
        if d[0] == "BodyText":
            nums.append(int(d[1][len("Section"):]))
    sections = ["BodyText/Section" + str(x) for x in sorted(nums)]

    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        try:
            unpacked_data = zlib.decompress(data, -15) if is_compressed else data
        except zlib.error as e:
            print(f"Decompression error: {e}")
            continue

        section_text = ""
        i = 0
        size = len(unpacked_data)
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff

            if rec_type == 67:
                rec_data = unpacked_data[i + 4:i + 4 + rec_len]
                try:
                    decoded_text = rec_data.decode('utf-16-le', errors='ignore')

                    # 한글, 숫자, 영어, 공백만 유지
                    cleaned_text = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', decoded_text)
                    section_text += cleaned_text + "\n"
                except UnicodeDecodeError as e:
                    print(f"Decoding error at position {i}: {e}")

            i += 4 + rec_len

        text += section_text
        text += "\n"

    return text

def get_docx_text(filename):
    try:
        doc = Document(filename)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print("Error reading DOCX file:", e)
        return None

def get_txt_text(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print("Error reading TXT file:", e)
        return None

def extract_text(file_path):
    extension = file_path.split('.')[-1].lower()
    if extension == 'hwp':
        return get_hwp_text(file_path)
    elif extension == 'docx':
        return get_docx_text(file_path)
    elif extension == 'txt':
        return get_txt_text(file_path)
    else:
        print("Unsupported file type.")
        return None

file_path = '테스트 대본1.hwp'  # hwp. docx, txt
text = extract_text(file_path)
print(text)
