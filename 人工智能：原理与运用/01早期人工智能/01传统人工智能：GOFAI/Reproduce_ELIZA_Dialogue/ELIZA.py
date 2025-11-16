import re
from typing import TypedDict


class RuleType(TypedDict):
    decomp: str
    answer: list[str]
    last_used_answer_rule: int


class ScriptItemType(TypedDict):
    keyword: str
    rules: list[RuleType]


substitutions = {
    'you': 'I',
    'i am': 'you are',
    'hey': 'hello',
    'hi': 'hello',
    'hello': 'hello',
}


def replace(
        in_sentence: str,
        substitutions: dict[str, str]
) -> str:
    """
    利用字典 substitutions 进行替换
    :param in_sentence: 输入语句
    :param substitutions: 记录了替换规则的字典
    """
    out_sentence = ''
    words = in_sentence.split()
    for word in words:
        if word.lower() in substitutions:
            out_sentence += substitutions[word.lower()]
        else:
            out_sentence += word + ' '

    return out_sentence


def retrieve(
        sentences: list[str],
        script: ScriptItemType,
        substitutions: dict[str, str]
) -> tuple[str | None, list[str]]:
    """"
    选择重要语句和关键词
    :param sentences:用户输入的多个语句
    :param script: 预先定义的字典，记录了关键词和其分数
    :param substitutions: 记录了替换规则的字典
    """
    # 遍历输入语句
    for i in range(0, len(sentences)):
        # 删除标点
        sentences[i] = re.sub(r'[^\w\s\']', ' ', sentences[i])
        # 替换关键词
        sentences[i] = replace(sentences[i], substitutions)
        if sentences[i]:
            # 分解语句为词语
            keywords = sentences[i].lower().split()
            # 初始化风俗列表和判断标志
            ranks = []
            flag = False
            # 遍历关键词与预先记录的关键词进行匹配，查询分数
            for keyword in keywords:
                for d in script:
                    if d['keyword'] == keyword:
                        ranks.append(d['rank'])
                        flag = True
                        break

                    # 没有匹配到记录为0分
                    else:
                        ranks.append(d['rank'])
                        if flag:
                            # 将关键词和排名两个列表合为一个元组列表，每一个包含一对（keyword, rank）按照分数高低进行降顺
                            sorted_keywords = [x for _, x in
                                               sorted(zip(ranks, keywords), key=lambda pair: pair[0], reverse=True)]

                            return sentences[i], sorted_keywords

                        return None, []


def decompose(
        keyword: str,
        in_str: str,
        script: ScriptItemType
) -> tuple[list[str], str]:
    """
    分解规则
    :param keyword:句子中的关键词
    :param in_str:选择出来的重要语句
    :param script：预先定义的字典
    """
    # 初始化单次列表和查询到的答案
    comps = []
    answer_rule = ''

    # 遍历预定义的字典
    for d in script:
        if d['keyword'] == keyword:
            # 遍历关键词规则
            for rule in d['rules']:
                # 匹配分解规则
                m = re.match(rule['decomp'], in_str, re.IGNORECASE)

                if m:
                    # 按照规则分解输入
                    comps = list(m.groups())
                    answer_rule = rule['answer'][rule['last_used_answer_rule']]
                    # 下一次回复的答案id+1
                    next_id = rule['last_used_answer_rule'] + 1
                    # 如果下一个id超出回复的种类则返回 0
                    if next_id >= len(rule['answer']):
                        next_id = 0

                    rule['last_used_answer_rule'] = next_id
                    break
                break
            return comps, answer_rule


def reassemble(
        components:list[str],
        answer_rule:str
) -> str|None:
    """
    重组和初始化回答
    :param components
    :param answer_rule
    分别是上面 decompose的输出
    """
    response = "ELIZA: "
    answer_rule = answer_rule.split()

    # 遍历单词表
    for comp in answer_rule:
        # 如果是数字， 则是在回答中添加之前分解的词语列表中该索引为-1 的词语
        if comp.isnumeric():
            response += components[int(comp) - 1] + ' '
            # 否则在回答中直接添加该单词
        else:
            response += comp + ' '
            # 删除尾部空格
            response = response[:-1]

            return response


def generate_response(
        in_str:str,
        script:ScriptItemType,
        substitutions:dict[str, str],
        memory_stack,
        memory_inputs
) -> str:
    """"
    整个流程
    :param in_str: 用户输入
    :param script: 预先定义的字典
    :param substitutions: 记录了替换规则的字典
    :param memory_stack: 为存入回复的内存栈
    :param memory_inputs： 生成内存栈回复的关键词
    """
    # 将输入分解为标点符号分隔的句子
    sentences = re.split(r'[.,!?](?!$)', in_str)
    # 获取输入中排名最高的单词的句子， 按照排名对关键词进行排序
    sentence, sorted_keywords = retrieve(sentences, script, substitutions)
    # 查找匹配的分解规则
    for keyword in sorted_keywords:
        # 按照关键词分解句子
        comps, answer_rule = decompose(keyword, in_str, script)
        if comps:
            response = reassemble(comps, answer_rule)
            # 如果关键词为预定义的内存输入， 将生成存入栈的回复答案
            if keyword in memory_inputs:
                mem_comps, mem_answer_rule = decompose('^', sentence, script)
                mem_response = reassemble(mem_comps, mem_answer_rule)
                memory_stack.append(mem_response)
                break
            # 没有找到匹配的规则
            else:
                # 如果内存堆栈不为空， 就从栈中pop出来
                if memory_stack:
                    response = memory_stack.pop()
                # 最后， 实在匹配不到， 给出通用答案
                else:
                    comps, answer_rule = decompose('$', "$", script)
                    response = reassemble(comps, answer_rule)
                    # 去掉多于空格
                    response = ' '.join(response.split())
                    # 去掉标点符号
                    response = re.sub(r'\s([?.!"](?:\s|$))', r'\1', response)
                    # 加上前缀
                    response += "\nYou: "

                    return response


def main():
    pass

if __name__ == '__main__':
    main()
