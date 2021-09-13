# from Levenshtein.StringMatcher import StringMatcher
# str_matcher = StringMatcher()
#
# str1 = 'deceukelaire'
# str2 = 'de ceukelairef'
#
# str_matcher.set_seqs(str1, str2)
# editops = str_matcher.get_editops()
#
# print(editops)
# involved_chars = []
# for model, pos1, pos2 in editops:
#     if model == 'delete':
#         print('delete: ', str1[pos1])
#         involved_chars.append(str1[pos1])
#     elif model == 'replace':
#         print('replace: ', str1[pos1])
#         involved_chars.append(str1[pos1])
#     elif model == 'insert':
#         print('insert: ', str2[pos1])
#         involved_chars.append(str2[pos1])
#
# print(involved_chars)
#

# ### test spllit
# split = 'research support, n.i.h., extramural'.strip('research support, ')
# print(split)
# index = split.index('e')
# print(index)
#
#
# import re
# test_string = "Geeksforgeeks,    is best @# Computer Science Portal.!!!"
# print ("The original string is : " +  test_string)
# res = re.findall(r'\w+', test_string)
# print ("The list of words is : " +  str(res))
import unicodedata

from nltk import ngrams

sentence = 'lizhang'
n = 2
grams = ngrams(list(sentence), n)
grams = [''.join(gram) for gram in grams]
print(grams)

special_chars = """
é í á ü ö ó ç ł ú . ø ã éé è ë -ç ' å éè ş ô éô â š -é .. ï ê áó ä ñ ı ğ éí ż íé î ̇ ò ž -ü áš à é- ří -éè ř áá üü --
ć öü ě í- éá çã íú éó -ë íá -ö -è ç- áé üş æ şü õ íó -å íí ė ő çğ ū ţ ış áí łł éú -éé í-é č ö- úé ăă éã ıı şı üç éï éç
áã øø -í đ öç ă óá ... .-. ý ª íã ň áç ïé óí -á çı -. ü- úí éë -ø íç ôú ôé ãé úá çğı üö ãí é-í áú şüü šá óé í-ú -î ä-
łż áô а á- íñ ę ø- éê éé- ð éâ ì ç-é ôá ā ğç ĕ šć ıç øæ íê šěá ãá ııı áê ľ óó íô -ó üğ čć é. çğş ūė -ï ù û '' óã ľí üı
öé úš öö ç-ï öğ ī úã ț úñ é-á å- ń ] 'í ğş ó- şş -ú áçã áâ -áó èé í-á ë- ș ãú úó è- ãç -ã ē üçü о íě đđ ãã şö öş м ôí
ãê êé --- øå úç áñ åø şğ ãó âç ğı ıö ãô őő ̇ı âú ʼ é' ıü üşü êã -' 'á ää žě üüö ō çç ôç éñ éçã ªé é-é е / ś íóã ú- ã- âé
äö şç âá ééé íð üüü é-ó úú α ̇ö óú èè éáó 'é ñó üış žć ğü êá ááó í-ó ūėšė âí ªá íâ ôã í-- öó éõ '-' ̇ç ê- ěá ôó íõ ôõ âã
óö ěž öı àò íçã ̇̇ şıı ι т óż åå âê ÿ á-á ̇ü üçğ áóá čí ĭ úê ̇ş '- úçã ӧ ćć ìá -ñ í. -ê çé ľš šč üöü í-í .... çõ óô ì-á çö
çş ááá óç óöé čá ş- éáçã ü̇ ãâ ñí ııç žė öá óñ аа ç-ë âó ªú ε öüş çá ăţ ç-éô ñ- üüş -à μ áü í' ığ řš ñá ĺ ö-ü -š -é- éè-
-éô -ăă áö íü ź şăţă ú-í êç κ -č öüö ё â- áóé öüü ôê œ íéí ñé в ô- ö̇ šū íö á-í ß ş̇ і ď šž к ĵ îş üşğ ïç žč řá , ëé âéú
éíí óê éü ðú ν é-ë å-ø ééè -ţ åö éö ŕ öüç üü̇ óð šė -ô öüüü éª üıı ï- çü -ä -ş æø ãôê ôâ еа ̇ğ ă- ţ-ţ í-ñ ú'é -â ó-í é-ú
éî î- áõ ǧ éĵ áň êçã çığı üşı"""

from unidecode import unidecode

print(unidecode(special_chars))
print(special_chars.upper())
print(unidecode(special_chars))

# for n in ():
#     print(strip_accents(n))
