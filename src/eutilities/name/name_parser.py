import threading

import personnamenorm as pnn
from nameparser import HumanName


def derek73_nameparser(name_str):
    name = HumanName(name_str)
    # name.as_dict()
    return [name.first, name.middle, name.last]


threadLocal = threading.local()


def thread_local_init():
    initialized = getattr(threadLocal, 'initialized', None)
    if initialized is None:
        print('init thread local and loaded pickle data')
        threadLocal.personnamenorm = pnn.namenorm('cached/p_firstname.p')
        threadLocal.initialized = True
    else:
        # print('has inited thread local')
        pass


class NameProcessor():
    def __init__(self):
        pass

    def __call__(self, au):
        thread_local_init()
        if au is None or len(au) == 0:
            return []
        else:
            splited_au = []
            for pos, au_name in au:
                # print(id(threadLocal.personnamenorm))
                personnamenorm = threadLocal.personnamenorm
                personnamenorm.unify(au_name)
                splited_au.append([pos, ' '.join(personnamenorm.name['firstname']).lower(), 'merged_to_fn',
                                   ' '.join(personnamenorm.name['lastname']).lower()])
            # print(current_thread().name, splited_au)
            return splited_au


personnamenorm = pnn.namenorm('cached/p_firstname.p')


def klauslippert_personnamenorm(name_str):
    personnamenorm.unify(name_str)
    # print(name)
    return [' '.join(personnamenorm.name['firstname']).lower(), 'merged_to_fn',
            ' '.join(personnamenorm.name['lastname']).lower()]


if __name__ == '__main__':
    names = ['Douglas H. Keefe', 'Carolina Abdala', 'Ram C. Naidu', 'David C. Mountain', 'Christopher A. Shera',
             'John J. Guinan', 'Bernhard Ross', 'Kelly L. Tremblay', 'Terence W. Picton', 'Manfred Mauermann',
             'Volker Hohmann', 'Richard L. Freyman', 'Karen S. Helfer', 'Uma Balakrishnan', 'Soha N. Garadat',
             'Ruth Y. Litovsky', 'Michael A. Akeroyd', 'John Chambers', 'David Bullock', 'Alan R. Palmer',
             'A. Quentin Summerfield']
    for n in names:
        print(n.lower(), derek73_nameparser(n), derek73_nameparser(n.lower()))
        print(n.lower(), klauslippert_personnamenorm(n), klauslippert_personnamenorm(n.lower()))
