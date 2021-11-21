from collections import OrderedDict


def load_info(env_name):
    # input: env_name: 13, 123, 8500
    # output:
    #   1. load_base: OrderedDict(key=load name, value=kw, kvar of the IEEE feeders snapshot)
    #      for 8500, value=kw only
    #   2. load_node: OrderedDict(key=bus name of load, value=phases of load)

    if str(env_name) == '13':
        # load_base: key=name of load object in the dss script
        #            value=(kw, k-var)
        load_base = OrderedDict(
            [('671', (1155, 660)),  # load name: (kw, kvar)
             ('634a', (160, 110)),
             ('634b', (120, 90)),
             ('634c', (120, 90)),
             ('645', (170, 125)),
             ('646', (230, 132)),
             ('692', (170, 151)),
             ('675a', (485, 190)),
             ('675b', (68, 60)),
             ('675c', (290, 212)),
             ('611', (170, 80)),
             ('652', (128, 86)),
             ('670a', (17, 10)),
             ('670b', (66, 38)),
             ('670c', (117, 68)),
             ])

        # load_node: key=name of load object in the dss script
        #            value= phases of the load. Must be in the order (a,b,c)
        load_node = OrderedDict(
            [('671', ''),
             ('634', 'abc'),
             ('645', ''),
             ('646', ''),
             ('692', ''),
             ('675', 'abc'),
             ('611', ''),
             ('652', ''),
             ('670', 'abc'),
            ])

    elif str(env_name) == '123':
        with open("./envs/dss_123/IEEE123Loads.DSS", 'r') as dssfile:
            dss_str = dssfile.readlines()

        load_base = OrderedDict()
        for s in dss_str:
            if 'New Load.' in s:
                idx = s.index("New Load.") + len('New Load.')
                name = []
                for c in s[idx:]:
                    if c == ' ':
                        break
                    else:
                        name.append(c)
                name = ''.join(name)

                idx_kW = s.index("kW=") + len('kW=')
                kW = []
                for c in s[idx_kW:]:
                    if c == ' ':
                        break
                    else:
                        kW.append(c)
                kW = float(''.join(kW))

                idx_kvar = s.lower().index("kvar=") + len('kvar=')
                kvar = []
                for c in s[idx_kvar:]:
                    if c == ' ':
                        break
                    else:
                        kvar.append(c)
                kvar = float(''.join(kvar))

                load_base[name] = (kW, kvar)

        load_node = OrderedDict()
        phases = {'1': 'a',
                  '2': 'b',
                  '3': 'c',
                  ' ': ''}
        for s in dss_str:
            if 'Bus1=' in s:
                idx = s.index("Bus1=") + len('Bus1=')
                name = []
                for c in s[idx:]:
                    if c == '.':
                        p = s[idx + len(name) + 1]
                        break
                    elif c == ' ':
                        p = ' '
                        break
                    else:
                        name.append(c)

                name = ''.join(name).lower()

                load_node[name] = phases[p]

    elif str(env_name) == '8500':
        with open("./envs/dss_8500/Loads.dss", 'r') as dssfile:
            dss_str = dssfile.readlines()

        load_base = OrderedDict()
        for s in dss_str:
            if 'New Load.' in s:
                idx_name = s.index("New Load.") + len('New Load.')
                name = []
                for c in s[idx_name:]:
                    if c == ' ':
                        break
                    else:
                        name.append(c)
                name = ''.join(name)

                idx_kW = s.index("kW=") + len('kW=')
                kW = []
                for c in s[idx_kW:]:
                    if c == ' ':
                        break
                    else:
                        kW.append(c)
                kW = float(''.join(kW))

                load_base[name] = kW

        load_node = OrderedDict()
        for s in dss_str:
            if 'Bus1=' in s:
                idx_name = s.index("Bus1=") + len('Bus1=')
                name = []
                for c in s[idx_name:]:
                    if c == '.':
                        break
                    else:
                        name.append(c)
                name = ''.join(name).lower()

                load_node[name] = ''

    return load_base, load_node
