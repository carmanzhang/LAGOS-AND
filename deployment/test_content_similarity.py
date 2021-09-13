import json

import requests

url = 'http://127.0.0.1:38080/score'
data = json.dumps({"contents": [
    [
        'the framework\'s representations. figure 2 illustrates a dsynts from a meteorological application, meteocogent (kittredge and lavoie, 1998), represented using the standard graphical notation and also the realpro ascii notation used internally in the framework (lavoie and rambow, 1997).. as figure 2 illustrates, there is a straightforward mapping between the graphical notation and the ascii notation supported in the framework.. this also applies for all the transformation rules in the framework which illustrates the declarative nature of our approach.',
        'background: the andes physics tutor. robust natural language understanding in atlas-andes is provided by rosé\'s carmel system (rosé 2000); it uses the spelling correction algorithm devised by elmi and evens (1998).. 5.2 structure of human tutorial dialogues in an earlier analysis (kim, freedman and evens 1998) we showed that a significant portion of human-human tutorial dialogues can be modeled with the hierarchical structure of task-oriented dialogues (grosz and sidner 1986).. furthermore, a main building block of the discourse hierarchy, corresponding to the transaction level in conversation analysis (sinclair and coulthard 1975), matches the tutoring episode defined by vanlehn et al.'],
    [' One of the valuable indicators of the structure of text is lexical cohesion  ( Halliday and Hasan 1976 )  . ',
     ' Magic ( templates ) is a general compilation technique for efficient bottom-up evaluation of logic programs developed in the deductive database community  ( Ramakrishnan et al. 1992 )  . '],
    [
        'Magic ( templates ) is a general compilation technique for efficient bottom-up evaluation of logic programs developed in the deductive database community  ( Ramakrishnan et al. 1992 )  .  ',
        ' Magic ( templates ) is a general compilation technique for efficient bottom-up evaluation of logic programs developed in the deductive database community  ( Ramakrishnan et al. 1992 )  . '],
    [
        'Magic ( templates ) is a general compilation technique for efficient bottom-up evaluation of logic programs developed in the deductive database community  ( Ramakrishnan et al. 1992 )  .  ',
        ''],
    [
        ' Machine Readable Dictionaries ( MRDs ) are a good source of lexical information and have been shown to be applicable to the task of LKB construction  ( Dolan et al. 1993  ,  Calzolari 1992  ,  Copestake 1990  ,  Wilks et al. 1989  ,  Byrd et al. 1987 )  .',
        'However , Pirkola  ( Pirkola 1998 )  , for example , used a subset of the TREC collection related to health topics , and showed that combination of general and domain specific ( i.e. , medical ) dictionaries improves the CLIR performance obtained with only a general dictionary .']
]})
print(data)
ret = requests.post(url, data={"content": data})
print(ret.text)
