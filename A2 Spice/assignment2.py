from numpy import *
from sys import argv, exit

CIRCUIT = '.circuit'
END = '.end'
AC = '.ac'
FILETYPE = '.netlist'

if len(argv)!=2:
    print('No of arguments must be 2!')
    exit(0)
SPICE_NAME = argv[1]

if SPICE_NAME[-len(FILETYPE):] != FILETYPE:  # checking the file type of the file
   print('File type is incorrect!')
   exit(0)

class resistor:
    def __init__(self, Name, N1, N2, Value):
        self.Name = Name
        self.N1 = N1
        self.N2 = N2
        self.Value = Value

class inductor:
    def __init__(self, Name, N1, N2, Value):
        self.Name = Name
        self.N1 = N1
        self.N2 = N2
        self.Value = Value

class capacitor:
    def __init__(self, Name, N1, N2, Value):
        self.Name = Name
        self.N1 = N1
        self.N2 = N2
        self.Value = Value

class currentsource:
    def __init__(self, Name, N1, N2, Value):
        self.Name = Name
        self.N1 = N1
        self.N2 = N2
        self.Value = Value

class voltagesource:
    def __init__(self, Name, N1, N2, Value):
        self.Name = Name
        self.N1 = N1
        self.N2 = N2
        self.Value = Value

try:

    with open(argv[1]) as f:
        lines = f.readlines()
        f.close()
        start = -1
        end = -2
        checkac = 0
        for line in lines:
            if CIRCUIT == line[:len(CIRCUIT)]:
               start = lines.index(line)
            elif END == line[:len(END)]:
               end = lines.index(line)
            elif AC == line[:len(AC)]:
               checkac = lines.index(line)

        if start > end or checkac == 0:
           print("INVALID!")
           exit(0)

        token = []
        X = 0

        try:
            for line in lines:
                if AC == line[:len(AC)]:
                    words = line.split()
                    if len(words) == 3:
                          freq3 = words[2]
                          freq1 = freq3.split('e')
                          freq2 = float(float(freq1[0])*(10**int(freq1[1])))
                          freq = float(2 * 3.14 * freq2)

            for line in lines[start+1:end]:
                Name, N1, N2, *Value = line.split()
                if Name[0] == 'R':
                    object = resistor(Name, N1, N2, Value)
                elif Name[0] == 'L':
                    object = inductor(Name, N1, N2, Value)
                elif Name[0] == 'C':
                    object = capacitor(Name, N1, N2, Value)
                elif Name[0] == 'I':
                    object = currentsource(Name, N1, N2, Value)
                elif Name[0] == 'V':
                    object = voltagesource(Name, N1, N2, Value)
                    X += 1
                if len(object.Value) == 1:
                    value1 = object.Value[0].split('e')
                    value2 = float((float(value1[0]))*(10 ** int(value1[1])))
                    object.Value = value2
                else:
                    object.Value = (float(object.Value[1])/2) * complex(cos(float(object.Value[2])), sin(float(object.Value[2])))
                token.append(object)

        except IndexError:
            print('ERROR!!!')
            exit(0)

    Node = {}
    for object in token:
        if object.N1 not in Node:
            if object.N1 == 'GND':
                Node['N0'] = 'GND'
            else:
                Name = 'n' + object.N1
                Node[Name] = int(object.N1)

        if object.N2 not in Node:
            if object.N2 == 'GND':
                Node['N0'] = 'GND'
            else:
                Name = 'n' + object.N2
                Node[Name] = int(object.N2)

    N = len(Node)
    Node['N0'] = 0
    Y = 0

    M = zeros(((N+X-1), (N+X-1)), dtype="complex")
    b = zeros(((N+X-1), 1), dtype="complex")

    for object in token:
        if object.Name[0] == 'R':
            if object.N1 == 'GND':
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value

            elif object.N2 == 'GND':
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value

            else:
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value
                M[int(object.N1) - 1][int(object.N2) - 1] += -1/object.Value
                M[int(object.N2) - 1][int(object.N1) - 1] += -1/object.Value

        elif object.Name[0] == 'L':
            XL = float((freq*float(object.Value)))
            object.Value = complex(0, XL)
            if object.N1 == 'GND':
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value

            elif object.N2 == 'GND':
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value

            else:
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value
                M[int(object.N1) - 1][int(object.N2) - 1] += -1/object.Value
                M[int(object.N2) - 1][int(object.N1) - 1] += -1/object.Value

        elif object.Name[0] == 'C':
            XC = float(-1/float((freq*float(object.Value))))
            object.Value = complex(0, XC)
            if object.N1 == 'GND':
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value

            elif object.N2 == 'GND':
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value

            else:
                M[int(object.N1) - 1][int(object.N1) - 1] += 1/object.Value
                M[int(object.N2) - 1][int(object.N2) - 1] += 1/object.Value
                M[int(object.N1) - 1][int(object.N2) - 1] += -1/object.Value
                M[int(object.N2) - 1][int(object.N1) - 1] += -1/object.Value

        elif object.Name[0] == 'I':
            if object.N1 == 'GND':
                b[int(object.N2) - 1][0] += object.Value

            elif object.N2 == 'GND':
                b[int(object.N1) - 1][0] += -object.Value

            else:
                b[int(object.N1) - 1][0] += object.Value
                b[int(object.N2) - 1][0] += -object.Value

        elif object.Name[0] == 'V':
            if object.N1 == 'GND':
                M[int(object.N2) - 1][N+Y-1] += 1
                M[N+Y-1][int(object.N2) - 1] += 1
                b[N+Y-1][0] += object.Value
                Y = Y + 1
            elif object.N2 == 'GND':
                M[int(object.N1) - 1][N+Y-1] += -1
                M[N+Y-1][int(object.N1) - 1] += -1
                b[N+Y-1][0] += object.Value
                Y = Y + 1
            else:
                M[int(object.N1) - 1][N+Y-1] += 1
                M[int(object.N2) - 1][N+Y-1] += -1
                M[N+Y-1][int(object.N1) - 1] += 1
                M[N+Y-1][int(object.N2) - 1] += -1
                b[N+Y-1][0] += object.Value
                Y = Y + 1

    V = linalg.solve(M, b)
    for i in range(N-1):
        print('V', i+1, '=', V[i], '\n')
    for j in range(X):
        print('I', j+1, '=', V[j+N-1], '\n')

except IndexError:
    print('INVALID!!!!!')
    exit(0)

