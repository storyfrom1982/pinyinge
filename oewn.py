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

g2p = G2p()

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

# ARPABET_TO_IPA = {
#     # 元音（英式标准）
#     'AA': 'ɑː',   # father /ˈfɑː.ðə/
#     'AE': 'æ',     # cat /kæt/
#     'AH': 'ə',     # about /əˈbaʊt/
#     'AO': 'ɔː',    # thought /θɔːt/
#     'AW': 'aʊ',    # mouth /maʊθ/
#     'AY': 'aɪ',    # price /praɪs/
#     'EH': 'e',     # dress /dres/
#     'ER': 'ɜː',    # nurse /nɜːs/
#     'EY': 'eɪ',    # face /feɪs/
#     'IH': 'ɪ',     # kit /kɪt/
#     'IY': 'iː',    # fleece /fliːs/
#     'OW': 'əʊ',    # goat /ɡəʊt/ (英式) vs 美式 oʊ
#     'OY': 'ɔɪ',    # choice /tʃɔɪs/
#     'UH': 'ʊ',     # foot /fʊt/
#     'UW': 'uː',    # goose /ɡuːs/
    
#     # 辅音（与美式相同）
#     'B': 'b',      # bed /bed/
#     'CH': 'tʃ',    # church /tʃɜːtʃ/
#     'D': 'd',      # dog /dɒɡ/
#     'DH': 'ð',     # this /ðɪs/
#     'F': 'f',      # fish /fɪʃ/
#     'G': 'ɡ',      # game /ɡeɪm/
#     'HH': 'h',     # hat /hæt/
#     'JH': 'dʒ',    # judge /dʒʌdʒ/
#     'K': 'k',      # cat /kæt/
#     'L': 'l',      # leg /leɡ/
#     'M': 'm',      # man /mæn/
#     'N': 'n',      # now /naʊ/
#     'NG': 'ŋ',     # sing /sɪŋ/
#     'P': 'p',      # pen /pen/
#     'R': 'r',      # red /red/ (英式常为齿龈近音)
#     'S': 's',      # sun /sʌn/
#     'SH': 'ʃ',     # shoe /ʃuː/
#     'T': 't',      # tea /tiː/
#     'TH': 'θ',     # think /θɪŋk/
#     'V': 'v',      # voice /vɔɪs/
#     'W': 'w',      # wet /wet/
#     'Y': 'j',      # yes /jes/
#     'Z': 'z',      # zoo /zuː/
#     'ZH': 'ʒ'      # pleasure /ˈpleʒ.ə/
# }

def split_syllables(phonemes):
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
                elif is_legal_onset(phonemes[next_idx:]):
                    syllables.append(current_syllable)
                    current_syllable = []
    
    if current_syllable:
        syllables.append(current_syllable)
    return syllables

def is_legal_onset(consonants):
    """检查辅音序列是否能作为合法音节开头"""
    # 实现英语合法起始辅音组合的判断逻辑
    # 例如：tr, pl, str 合法；ng, mb 不合法
    return True  # 简化示例，实际需完整实现


def convert_arpabet_to_ipa(phonemes):
    """生成标准 IPA 音标（含正确音节分割和重音）"""
    syllables = split_syllables(phonemes)
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

def get_phonetics(word, output_format="ipa"):
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
        return convert_arpabet_to_ipa(phonemes)
    else:
        return " ".join(phonemes)

# 测试用例
test_words = ["photograph", "banana", "water", "university"]
for word in test_words:
    ipa = get_phonetics(word, "ipa")
    print(f"{word.upper():<12} {ipa}")