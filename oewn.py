import lemminflect
from nltk.corpus import cmudict
import nltk
from g2p_en import G2p

# 初始化 CMU 发音词典
try:
    cmu = cmudict.dict()
except LookupError:
    nltk.download("cmudict")
    cmu = cmudict.dict()

# 初始化 g2p-en（用于预测发音）
g2p = G2p()

# ARPABET 到 IPA 的映射表（含重音符号）
ARPABET_TO_IPA = {
    # 元音
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 
    'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 
    'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u',
    # 辅音
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'F': 'f', 
    'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 
    'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'R': 'r', 
    'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

def convert_arpabet_to_ipa(phonemes):
    """将 ARPABET 音标转换为 IPA，并添加重音符号"""
    ipa_phonemes = []
    for p in phonemes:
        # 提取音素和重音数字（如 'AH0' → ('AH', '0')）
        base_p = ''.join([c for c in p if not c.isdigit()])
        stress = next((c for c in p if c.isdigit()), None)
        
        # 转换为 IPA 并添加重音符号
        if base_p in ARPABET_TO_IPA:
            ipa = ARPABET_TO_IPA[base_p]
            if stress == '1':  # 主重音
                ipa = 'ˈ' + ipa
            elif stress == '2':  # 次重音
                ipa = 'ˌ' + ipa
            ipa_phonemes.append(ipa)
    return ''.join(ipa_phonemes)

def get_phonetics(word, output_format="arpabet"):
    """获取音标（优先 CMUdict，回退到 g2p-en）"""
    word_lower = word.lower()
    phonemes = None
    
    # 1. 优先从 CMUdict 获取
    if word_lower in cmu:
        phonemes = cmu[word_lower][0]  # 使用第一个发音变体
    else:
        # 2. 回退到 g2p-en 预测
        phonemes = g2p(word_lower)
        # 过滤掉非 ARPABET 符号
        phonemes = [p for p in phonemes if p in ARPABET_TO_IPA or p.isalpha()]
    
    if not phonemes:
        return None

    # 3. 转换为指定格式
    if output_format == "ipa":
        return convert_arpabet_to_ipa(phonemes)
    else:
        return " ".join(phonemes)  # 原始 ARPABET 格式

def get_word_forms(word):
    """获取词形变化"""
    inflections = lemminflect.getAllInflections(word)
    return {pos: forms for pos, forms in inflections.items() if forms}

def get_word_data(word):
    """获取单词的完整数据"""
    return {
        "word": word,
        "phonetics_arpabet": get_phonetics(word, "arpabet"),
        "phonetics_ipa": get_phonetics(word, "ipa"),
        "inflections": get_word_forms(word)
    }

# 测试单词（含重音）
test_words = ["photograph", "banana", "interesting", "America", "about"]
for word in test_words:
    data = get_word_data(word)
    print(f"\n{word.upper()}:")
    print(f"  - ARPABET: {data['phonetics_arpabet']}")
    print(f"  - IPA: {data['phonetics_ipa']}")
    print(f"  - Inflections: {data['inflections']}")