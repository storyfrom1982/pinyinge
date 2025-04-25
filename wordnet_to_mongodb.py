import json
from pymongo import MongoClient
from nltk.corpus import wordnet as wn, brown
from collections import defaultdict
import hashlib
import lemminflect
from nltk.corpus import cmudict
from g2p_en import G2p
import nltk
from typing import Optional, Dict, List
from nltk.probability import FreqDist

# 初始化资源
nltk.download(['wordnet', 'brown', 'cmudict', 'averaged_perceptron_tagger_eng'], quiet=True)
g2p = G2p()
cmu = cmudict.dict()
brown_fd = FreqDist(brown.words())

# MongoDB连接配置
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "wordnet_db"
WORDS_COLLECTION = "words"
SYNSETS_COLLECTION = "synsets"

ARPABET_TO_IPA = {
    # 元音（美式标准）
    'AA': 'ɑ',    # father /ˈfɑ.ðɚ/ (英式 ɑː)
    'AE': 'æ',     # cat /kæt/
    'AH': 'ə',     # about /əˈbaʊt/
    'AO': 'ɔ',     # thought /θɔt/ (英式 ɔː)
    'AW': 'aʊ',    # mouth /maʊθ/
    'AY': 'aɪ',    # price /praɪs/
    'EH': 'ɛ',     # dress /drɛs/ (英式 e)
    'ER': 'ɝ',     # nurse /nɝs/ (英式 ɜː)
    'EY': 'eɪ',    # face /feɪs/
    'IH': 'ɪ',     # kit /kɪt/
    'IY': 'i',     # fleece /flis/ (英式 iː)
    'OW': 'oʊ',    # goat /ɡoʊt/ (英式 əʊ)
    'OY': 'ɔɪ',    # choice /tʃɔɪs/
    'UH': 'ʊ',     # foot /fʊt/
    'UW': 'u',     # goose /ɡus/ (英式 uː)

    # 辅音（与英式基本相同）
    'B': 'b',      # bed /bɛd/
    'CH': 'tʃ',    # church /tʃɝtʃ/
    'D': 'd',      # dog /dɔɡ/
    'DH': 'ð',     # this /ðɪs/
    'F': 'f',      # fish /fɪʃ/
    'G': 'ɡ',      # game /ɡeɪm/
    'HH': 'h',     # hat /hæt/
    'JH': 'dʒ',    # judge /dʒʌdʒ/
    'K': 'k',      # cat /kæt/
    'L': 'l',      # leg /lɛɡ/
    'M': 'm',      # man /mæn/
    'N': 'n',      # now /naʊ/
    'NG': 'ŋ',     # sing /sɪŋ/
    'P': 'p',      # pen /pɛn/
    'R': 'r',      # red /rɛd/ (美式常为卷舌音)
    'S': 's',      # sun /sʌn/
    'SH': 'ʃ',     # shoe /ʃu/
    'T': 't',      # tea /ti/
    'TH': 'θ',     # think /θɪŋk/
    'V': 'v',      # voice /vɔɪs/
    'W': 'w',      # wet /wɛt/
    'Y': 'j',      # yes /jɛs/
    'Z': 'z',      # zoo /zu/
    'ZH': 'ʒ'      # pleasure /ˈplɛʒ.ɚ/
}

class WordNetExporter:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.words_col = self.db[WORDS_COLLECTION]
        self.synsets_col = self.db[SYNSETS_COLLECTION]
        
        # 创建索引
        self._create_indexes()

    def _create_indexes(self):
        """创建数据库索引"""
        self.words_col.create_index("lemma", unique=True, name="lemma_unique")
        self.words_col.create_index("phonetic", name="phonetic_index")
        self.words_col.create_index("contains_satellite", name="satellite_index")
        
        self.synsets_col.create_index("synset_id", unique=True, name="synset_id_unique")
        self.synsets_col.create_index("lemma", name="lemma_index")
        self.synsets_col.create_index([("definition.en", "text")], name="text_search")

    @staticmethod
    def generate_id(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def split_syllables(self, phonemes):
        syllables = []
        current_syllable = []
        vowels = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        
        for i, p in enumerate(phonemes):
            base_p = ''.join([c for c in p if not c.isdigit()])
            current_syllable.append(p)
            
            # 遇到元音时考虑分割
            if base_p in vowels:
                # 检查后续辅音是否能合法起始新音节
                next_idx = i + 1
                if next_idx < len(phonemes):
                    next_base = ''.join([c for c in phonemes[next_idx] if not c.isdigit()])
                    # 如果是词尾单辅音，不分割
                    if next_idx == len(phonemes) - 1 and next_base not in vowels:
                        continue
                    # 如果是辅音连缀且能合法起始（如tr, pl等），则分割
                    elif self.is_legal_onset(phonemes[next_idx:]):
                        syllables.append(current_syllable)
                        current_syllable = []
        
        if current_syllable:
            syllables.append(current_syllable)
        return syllables

    def is_legal_onset(self, consonants):
        """检查辅音序列是否能作为合法音节开头"""
        # 实现英语合法起始辅音组合的判断逻辑
        # 例如：tr, pl, str 合法；ng, mb 不合法
        return True  # 简化示例，实际需完整实现


    def convert_arpabet_to_ipa(self, phonemes):
        """生成标准 IPA 音标（含正确音节分割和重音）"""
        syllables = self.split_syllables(phonemes)
        ipa_parts = []
        
        for syllable in syllables:
            syllable_ipa = []
            stress = None
            
            for p in syllable:
                base_p = ''.join([c for c in p if not c.isdigit()])
                digit = next((c for c in p if c.isdigit()), None)
                
                if digit == '1':
                    stress = 'ˈ'
                elif digit == '2':
                    stress = 'ˌ'
                    
                if base_p in ARPABET_TO_IPA:
                    syllable_ipa.append(ARPABET_TO_IPA[base_p])
            
            # 将重音符号置于音节开头
            if stress:
                syllable_ipa[0] = stress + syllable_ipa[0]
            ipa_parts.append(''.join(syllable_ipa))
        
        return '.'.join(ipa_parts)

    def get_phonetics(self, word, output_format="ipa"):
        """获取音标（优先 CMUdict，回退到 g2p-en）"""
        word_lower = word.lower()
        phonemes = None
        
        if word_lower in cmu:
            phonemes = cmu[word_lower][0]
        else:
            phonemes = g2p(word_lower)
            phonemes = [p for p in phonemes if p in ARPABET_TO_IPA or (p.isalpha() and not p.isdigit())]
        
        if not phonemes:
            return None
        
        if output_format == "ipa":
            return self.convert_arpabet_to_ipa(phonemes)
        else:
            return " ".join(phonemes)

    def get_inflections(self, lemma: str, pos: str) -> List[str]:
        try:
            if pos == 'n':
                plural = lemminflect.getInflection(lemma, tag='NNS')
                return [plural[0]] if plural and plural[0] != lemma else []
            elif pos == 'v':
                forms = set()
                for tense in ['VBG', 'VBD', 'VBN', 'VBZ']:
                    inflected = lemminflect.getInflection(lemma, tag=tense)
                    if inflected and inflected[0] != lemma:
                        forms.add(inflected[0])
                return list(forms)
            elif pos in ['a', 's']:
                result = []
                for tag in ['JJR', 'JJS']:
                    inflected = lemminflect.getInflection(lemma, tag=tag)
                    if inflected and inflected[0] != lemma:
                        result.append(inflected[0])
                return result
            return []
        except Exception:
            return []

    def get_true_pos(self, synset) -> str:
        return getattr(synset, '_pos', synset.pos())

    def get_synset_relations(self, synset, true_pos: str) -> Dict[str, List[str]]:
        relations = {}
        standard_relations = {
            'hypernym': synset.hypernyms(),
            'hyponym': synset.hyponyms(),
            'part_meronym': synset.part_meronyms(),
            'part_holonym': synset.part_holonyms(),
            'entailment': synset.entailments(),
            'cause': synset.causes(),
            'also_sees': synset.also_sees(),
            'verb_groups': synset.verb_groups()
        }
        
        if true_pos == 's':
            standard_relations['similar_to'] = synset.similar_tos()
        
        for rel_type, targets in standard_relations.items():
            if targets:
                relations[rel_type] = [s.name() for s in targets]
        return relations

    def export_to_mongodb(self, max_entries: Optional[int] = None, batch_size: int = 1000):
        """导出数据到MongoDB
        
        Args:
            max_entries: 最大导出条目数
            batch_size: 批量插入的大小
        """
        # 清空现有集合
        self.words_col.delete_many({})
        self.synsets_col.delete_many({})
        
        # 处理所有synset
        all_synsets = list(wn.all_synsets())
        if max_entries:
            all_synsets = all_synsets[:max_entries]
        
        # 第一遍：收集lemma信息
        lemma_data = defaultdict(lambda: {
            "pos_list": defaultdict(lambda: {"sense_count": 0, "senses": []}),
            "inflections": set(),
            "is_satellite": False
        })
        
        for synset in all_synsets:
            true_pos = self.get_true_pos(synset)
            for lemma in synset.lemmas():
                lemma_name = lemma.name()
                lemma_data[lemma_name]["lemma"] = lemma_name
                lemma_data[lemma_name]["pos_list"][true_pos]["sense_count"] += 1
                lemma_data[lemma_name]["pos_list"][true_pos]["senses"].append({
                    "synset_id": synset.name()
                })
                if true_pos == 's':
                    lemma_data[lemma_name]["is_satellite"] = True
                if true_pos in ['a', 's']:
                    lemma_data[lemma_name]["inflections"].update(self.get_inflections(lemma_name, 'a'))
        
        # 批量插入words数据
        words_batch = []
        for lemma, data in lemma_data.items():
            main_synset = wn.synsets(lemma)[0] if wn.synsets(lemma) else None
            words_batch.append({
                "_id": self.generate_id(lemma),
                "lemma": lemma,
                "phonetic": self.get_phonetics(lemma),
                "pinyin": "",
                "definition": {"en": main_synset.definition() if main_synset else "", "zh": ""},
                "inflections": list(data["inflections"]),
                "pos_list": {pos: info for pos, info in data["pos_list"].items()},
                "contains_satellite": data["is_satellite"],
                "usage_count": brown_fd.get(lemma.lower(), 0)
            })
            
            if len(words_batch) >= batch_size:
                self.words_col.insert_many(words_batch)
                words_batch = []
        
        if words_batch:
            self.words_col.insert_many(words_batch)
        
        # 批量插入synsets数据
        synsets_batch = []
        for synset in all_synsets:
            true_pos = self.get_true_pos(synset)
            lemmas = synset.lemmas()
            synsets_batch.append({
                "_id": self.generate_id(synset.name()),
                "synset_id": synset.name(),
                "lemma": lemmas[0].name(),
                "pos": true_pos,
                "is_satellite": (true_pos == 's'),
                "sense_number": lemmas[0].count(),
                "definition": {"en": synset.definition(), "zh": ""},
                "lexname": synset.lexname(),
                "lemmas": [lemma.name() for lemma in lemmas],
                "tag_count": sum(brown_fd.get(lemma.name().lower(), 0) for lemma in lemmas),
                "relations": self.get_synset_relations(synset, true_pos),
                "examples": [{"en": ex, "zh": ""} for ex in synset.examples()]
            })
            
            if len(synsets_batch) >= batch_size:
                self.synsets_col.insert_many(synsets_batch)
                synsets_batch = []
        
        if synsets_batch:
            self.synsets_col.insert_many(synsets_batch)
        
        print(f"导出完成！Words集合: {self.words_col.count_documents({})} 条")
        print(f"Synsets集合: {self.synsets_col.count_documents({})} 条")

    def close(self):
        self.client.close()

if __name__ == "__main__":
    exporter = WordNetExporter()
    try:
        # 示例：导出前10000个条目（设为None则导出全部）
        # exporter.export_to_mongodb(max_entries=10000, batch_size=500)
        exporter.export_to_mongodb()
        
        # 验证数据
        print("\n示例数据验证:")
        sample_word = exporter.words_col.find_one({"lemma": "able"})
        print(f"Word 'able': {sample_word['pos_list'].keys()}")
        
        sample_synset = exporter.synsets_col.find_one({"pos": "s"})
        print(f"Satellite synset: {sample_synset['synset_id']}")
    finally:
        exporter.close()