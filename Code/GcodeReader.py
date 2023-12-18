import numpy as np

# open gcode file and store contents as variable
with open('G1.mpf', 'r') as f:
  gcode = f.readlines()

WAAM = gcode[38:]
WAAM = WAAM[:1960]


X,Y,Z,A,B,C = [],[],[],[],[],[]

for line in WAAM:
    elemets = line.split( )
    x = float(elemets[2].split("=")[-1])
    y = float(elemets[3].split("=")[-1])
    z = float(elemets[4].split("=")[-1])
    a = float(elemets[5].split("=")[-1])
    b = float(elemets[6].split("=")[-1])
    c = float(elemets[7].split("=")[-1])

    X.append(x)
    Y.append(y)
    Z.append(z)
    A.append(a)
    B.append(b)
    C.append(c)

all = np.vstack((X,Y,Z,A,B,C))

np.save(f"RealG.npy", all)