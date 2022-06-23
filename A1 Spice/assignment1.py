"""
        EE2703 Applied Programming Lab - 2022
            Assignment 1
            JAGAN MJ
            EE20B047
"""

from sys import argv, exit


CIRCUIT = '.circuit'      #marks start of the netlist file
END = '.end'              #marks end of the netlist file
FILETYPE = '.netlist'     #file type of the netlist file


def token4(line):
    element_type = Token[0]
    from_Node = Token[1]
    to_Node = Token[2]
    value = Token[3]
    print('Type of element : %s, From node : %s, To Node : %s, Value : %s' % (element_type, from_Node, to_Node, value))
    return [element_type, from_Node, to_Node, value]

def token5(line):
    element_type = Token[0]
    from_node = Token[1]
    to_node = Token[2]
    voltage_Source = Token[3]
    value = Token[4]
    print('Type of element : %s, From node : %s, To Node : %s, Voltage Source : %s, Value : %s' % (element_type, from_node, to_node, voltage_Source, value))
    return [element_type, from_node, to_node, voltage_Source, value]

def token6(line):
    element_type = Token[0]
    from_Node = Token[1]
    to_Node = Token[2]
    voltage_Source1 = Token[3]
    voltage_Source2 = Token[4]
    value = Token[5]
    print('Type of element : %s, From node : %s, To Node : %s, Voltage Source1 : %s, Voltage Source2 : %s, Value : %s' % (element_type, from_Node, to_Node, voltage_Source1, voltage_Source2, value))
    return [element_type, from_Node, to_Node, voltage_Source1, voltage_Source2, value]

def token_None(line):
    return []

if len(argv) != 2:                            #throws error if the number of arguments entered is not 2
   print('NO OF ARGUMENTS MUST BE 2!')
   exit(0)

SPICE_NAME = argv[1]

if SPICE_NAME[-len(FILETYPE):] != FILETYPE:   #checking the file type of the file
    print('File type is incorrect!')
    exit(0)
try:
    with open(argv[1]) as f:
        lines = f.readlines()                                                           #reading the contents of the file
        f.close()
        start = -1; end = -2; START = 0; STOP = 0
        for line in lines:                                                              #extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
                START += 1
            elif END == line[:len(END)]:
                  end = lines.index(line)
                  STOP +=1
        if START !=1 or STOP !=1:                                                        #checks if multiple '.circuit' or '.end' is encountered
            print('File containing multiple ".circuit" or".end" which is not intended')
        if start >=end:                                                                 #incase .circuit and .end is not encountered
            print('Invalid circuit definition')
            exit(0)

        for line in lines[start+1:end]:
            Words = line.split('#')[0]                                                  #ignoring the comments and taking strings n=before '#'
            Token = Words.split()                                                       #getting each token
            if len(Token) == 4:                                                         #for independent sources like R,L,C
                 token4(line)
            elif len(Token) == 5:                                                       #for CCVS and CCCS
                 token5(line)
            elif len(Token) == 6:                                                       #for VCVS and VCCS
                 token6(line)
            else:
                 token_None(line)                                                       #Invalid element


        print("Required Output :  \n")
        for line in reversed([' '.join(reversed(line.split('#')[0].split()))
            for line in lines[start+1:end]]):                                           #This takes care of reversing the lines and then individually reversing the elements of each line
                print(line)

except IOError:
    print('Error. Invalid file, Make sure that the entered file name is correct.')
    exit(0)





























