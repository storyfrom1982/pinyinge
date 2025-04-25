import json
from nltk.corpus import wordnet as wn, brown
from collections import defaultdict
import hashlib
import lemminflect
from nltk.corpus import cmudict
from g2p_en import G2p
import nltk
from typing import Optional
from nltk.probability import FreqDist  # 添加这行导入

# 初始化资源
nltk.download(['wordnet', 'brown', 'cmudict'], quiet=True)
g2p = G2p()
cmu = cmudict.dict()
brown_fd = FreqDist(brown.words())

# ARPABET 到 IPA 的美式映射表
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 
    'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ',
    'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f',
    'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l',
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'r',
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

def generate_id(text: str) -> str:
    """生成唯一的_id"""
    return hashlib.md5(text.encode()).hexdigest()

def get_phonetic(lemma: str) -> str:
    """增强版音标生成（美式IPA）"""
    try:
        if lemma.lower() in cmu:
            phonemes = cmu[lemma.lower()][0]
            ipa = []
            for p in phonemes:
                base_p = ''.join([c for c in p if not c.isdigit()])
                if base_p in ARPABET_TO_IPA:
                    ipa.append(ARPABET_TO_IPA[base_p])
            return ''.join(ipa)
        return ''.join([ARPABET_TO_IPA.get(p, p) for p in g2p(lemma) if p in ARPABET_TO_IPA])
    except Exception:
        return ""

def get_inflections(lemma: str, pos: str) -> list:
    """安全获取词形变化，确保始终返回列表"""
    try:
        if pos == 'n':  # 名词
            plural = lemminflect.getInflection(lemma, tag='NNS')
            return [plural[0]] if plural and plural[0] != lemma else []
        elif pos == 'v':  # 动词
            forms = set()
            for tense in ['VBG', 'VBD', 'VBN', 'VBZ']:
                inflected = lemminflect.getInflection(lemma, tag=tense)
                if inflected and inflected[0] != lemma:
                    forms.add(inflected[0])
            return list(forms)
        elif pos in ['a', 's']:  # 形容词
            result = []
            for tag in ['JJR', 'JJS']:
                inflected = lemminflect.getInflection(lemma, tag=tag)
                if inflected and inflected[0] != lemma:
                    result.append(inflected[0])
            return result
        return []
    except Exception:
        return []

def export_wordnet_to_mongo(max_entries: Optional[int] = None):
    """导出WordNet数据到JSON文件
    
    Args:
        max_entries: 最大导出条目数（None表示全部导出）
    """
    words_collection = []
    synsets_collection = []
    lemma_stats = defaultdict(int)
    
    # 预计算词频统计
    for word in brown.words():
        lemma_stats[word.lower()] += 1
    
    # 处理所有synset（带数量限制）
    synsets = list(wn.all_synsets())
    if max_entries is not None:
        synsets = synsets[:max_entries]
    
    for synset in synsets:
        pos = synset.pos()
        synset_id = synset.name()
        lemmas = synset.lemmas()
        
        # 构建synset文档
        synset_doc = {
            "_id": generate_id(synset_id),
            "synset_id": synset_id,
            "lemma": lemmas[0].name(),
            "pos": pos,
            "sense_number": lemmas[0].count(),
            "definition": {"en": synset.definition(), "zh": ""},
            "lexname": synset.lexname(),
            "lemmas": [lemma.name() for lemma in lemmas],
            "tag_count": sum(lemma_stats.get(lemma.name().lower(), 0) for lemma in lemmas),
            "relations": {},
            "examples": [{"en": ex, "zh": ""} for ex in synset.examples()]
        }
        
        # 获取语义关系
        relation_types = {
            'hypernym': lambda s: s.hypernyms(),
            'hyponym': lambda s: s.hyponyms(),
            'part_meronym': lambda s: s.part_meronyms(),
            'part_holonym': lambda s: s.part_holonyms(),
            'entailment': lambda s: s.entailments(),
            'cause': lambda s: s.causes(),
            'also_sees': lambda s: s.also_sees(),
            'similar_tos': lambda s: s.similar_tos()
        }
        
        for rel_name, rel_func in relation_types.items():
            related = [s.name() for s in rel_func(synset)]
            if related:
                synset_doc["relations"][rel_name] = related
        
        synsets_collection.append(synset_doc)
    
    # 构建words文档（自动去重）
    lemma_data = defaultdict(lambda: {
        "pos_list": defaultdict(lambda: {"sense_count": 0, "senses": []}),
        "inflections": set()
    })
    
    processed_lemmas = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            lemma_name = lemma.name()
            if lemma_name in processed_lemmas:
                continue
                
            pos = synset.pos()
            lemma_data[lemma_name]["lemma"] = lemma_name
            lemma_data[lemma_name]["pos_list"][pos]["sense_count"] += 1
            lemma_data[lemma_name]["pos_list"][pos]["senses"].append({
                "synset_id": synset.name()
            })
            
            inflections = get_inflections(lemma_name, pos)
            if inflections:
                lemma_data[lemma_name]["inflections"].update(inflections)
            
            processed_lemmas.add(lemma_name)
    
    # 构建最终words文档
    for lemma, data in lemma_data.items():
        main_synset = wn.synsets(lemma)[0] if wn.synsets(lemma) else None
        
        word_doc = {
            "_id": generate_id(lemma),
            "lemma": lemma,
            "phonetic": get_phonetic(lemma),
            "pinyin": "",
            "definition": main_synset.definition() if main_synset else "",
            "inflections": list(data["inflections"]),
            "pos_list": {pos: info for pos, info in data["pos_list"].items()},
            "usage_count": lemma_stats.get(lemma.lower(), 0)
        }
        words_collection.append(word_doc)
    
    # 保存为JSON文件
    with open("wordnet-words.json", "w", encoding="utf-8") as f:
        json.dump(words_collection, f, ensure_ascii=False, indent=2)
    
    with open("wordnet-synsets.json", "w", encoding="utf-8") as f:
        json.dump(synsets_collection, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 示例：导出前1000个条目（设为None则导出全部）
    export_wordnet_to_mongo(max_entries=100)