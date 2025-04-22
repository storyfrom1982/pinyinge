import json
from nltk.corpus import wordnet as wn
import nltk
import os

def setup_nltk():
    """设置NLTK并确保WordNet数据可用"""
    try:
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
            return True
        except:
            return False

def get_wordnet_entry(word, pos='v'):
    """
    获取WordNet中指定单词的所有信息并返回字典
    
    参数:
        word: 要查询的单词
        pos: 词性标记(n=名词, v=动词, a=形容词, r=副词)
    
    返回:
        包含所有WordNet信息的字典
    """
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None
    
    entry = {
        'word': word,
        'pos': pos,
        'synsets': []
    }
    
    for synset in synsets:
        synset_data = {
            'name': synset.name(),
            'definition': synset.definition(),
            'examples': synset.examples(),
            'lemmas': [],
            'relations': {}
        }
        
        # 处理词元(lemmas)
        for lemma in synset.lemmas():
            lemma_data = {
                'name': lemma.name(),
                'antonyms': [a.name() for a in lemma.antonyms()],
                'pertainyms': [p.name() for p in lemma.pertainyms()],
                'derivationally_related_forms': [d.name() for d in lemma.derivationally_related_forms()]
            }
            synset_data['lemmas'].append(lemma_data)
        
        # 处理语义关系
        relations = {
            'hypernyms': [{'name': s.name(), 'definition': s.definition()} for s in synset.hypernyms()],
            'hyponyms': [{'name': s.name(), 'definition': s.definition()} for s in synset.hyponyms()],
            'part_holonyms': [{'name': s.name(), 'definition': s.definition()} for s in synset.part_holonyms()],
            'part_meronyms': [{'name': s.name(), 'definition': s.definition()} for s in synset.part_meronyms()],
            'entailments': [{'name': s.name(), 'definition': s.definition()} for s in synset.entailments()],
            'causes': [{'name': s.name(), 'definition': s.definition()} for s in synset.causes()],
            'verb_groups': [{'name': s.name(), 'definition': s.definition()} for s in synset.verb_groups()],
            'similar_tos': [{'name': s.name(), 'definition': s.definition()} for s in synset.similar_tos()]
        }
        synset_data['relations'] = {k: v for k, v in relations.items() if v}
        
        entry['synsets'].append(synset_data)
    
    return entry

def save_to_json(data, filename):
    """将数据保存为JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到 {filename}")

if __name__ == '__main__':
    if not setup_nltk():
        print("无法加载WordNet数据，请手动下载")
        print("运行: nltk.download('wordnet')")
    else:
        # 获取'make'作为动词的所有信息
        make_verb_data = get_wordnet_entry('make', pos='v')
        
        # 获取'make'作为名词的所有信息
        make_noun_data = get_wordnet_entry('make', pos='n')
        
        # 合并结果
        result = {
            'verb_entries': make_verb_data if make_verb_data else "No verb entries found",
            'noun_entries': make_noun_data if make_noun_data else "No noun entries found"
        }
        
        # 保存为JSON文件
        save_to_json(result, 'wordnet_make_entries.json')
        
        # 打印确认信息
        verb_count = len(make_verb_data['synsets']) if make_verb_data else 0
        noun_count = len(make_noun_data['synsets']) if make_noun_data else 0
        print(f"找到动词义项: {verb_count} 个，名词义项: {noun_count} 个")