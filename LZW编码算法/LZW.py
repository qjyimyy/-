

class LZW_encode:
    '''
    初始化字典
    让字典中仅含有最基本的单元素key
    '''
    def InitialDict(self, code):
        j = 0
        dictionary = {code[0]: 0}
        for i in range(len(code)):
            if code[i] in dictionary:
                continue
            else:
                j = j + 1
                dictionary[code[i]] = j
        return dictionary
    '''
    LZW编码算法
    '''
    def lzw(self, code, dictionary):


        dict_size = len(dictionary.keys())
        ResultDict = []

        p = ""
        for c in code:
            pc = p + c
            if pc in dictionary.keys():
                # pc在字典内，则更新p = pc
                p = pc
            else:
                # 若pc不在字典内，则更新字典，将key = pc顺序加入字典中，并更新p
                ResultDict.append(dictionary[p])
                dictionary[pc] = dict_size
                dict_size = dict_size + 1
                p = c
        # 将字典中的值放到数组中
        if p:
            ResultDict.append(dictionary[p])
        return ResultDict
class LZW_decode:
    '''
    解码算法
    '''
    def lzw_decoding(self, decode):

        # 初始化字典,默认种类最多三种
        dictionary = {0: 'a', 1: 'b', 2: 'c'}
        dict_size = len(dictionary)


        # 存放第一个元素
        ResultCode = []
        pw = decode.pop(0)
        ResultCode.append(dictionary[pw])

        for cw in decode:
            if cw in dictionary:
                output = dictionary[cw]
                pc = dictionary[pw] + dictionary[cw][0]
            elif cw == dict_size:
                # 若解码遇到未加入的映射，说明映射在当前步加入
                output = dictionary[pw] + dictionary[pw][0]
            else:
                # 不符合编码规则的编码
                raise ValueError('cw: %s 是错误编码' % cw)
            ResultCode.append(output)
            dictionary[dict_size] = pc
            dict_size += 1
            pw = cw
        return ResultCode

if __name__=="__main__":
    encode = LZW_encode
    EncStr = input("请输入要压缩的字符串")
    dictionary = encode.InitialDict(LZW_encode, EncStr)
    print(encode.lzw(LZW_encode, EncStr, dictionary))
    decode = LZW_decode
    DecStr = input("请输入要解压的数字")
    array = list(map(int, DecStr))  # 字符串输入转换成数组
    result = decode.lzw_decoding(LZW_decode, array)
    print(result)
    print(''.join(result))
