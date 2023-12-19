import matplotlib.pyplot as plt
import numpy as np

# open gcode file and store contents as variable
with open('G2.mpf', 'r') as f:
  WAAM = f.readlines()

X,Y,Z,A,B,C,tilt,rot = [],[],[],[],[],[],[],[]

for line in WAAM:
    elemets = line.split( )
    try:
        x = float(elemets[2].split("=")[-1])
        y = float(elemets[3].split("=")[-1])
        z = float(elemets[4].split("=")[-1])
        a = float(elemets[5].split("=")[-1])
        b = float(elemets[6].split("=")[-1])
        c = float(elemets[7].split("=")[-1])

        t = float(elemets[8].split("=")[-1])
        r = float(elemets[9].split("=")[-1])

        X.append(x)
        Y.append(y)
        Z.append(z)
        A.append(a)
        B.append(b)
        C.append(c)
        tilt.append(t)
        rot.append(r)
    except:
        print(elemets)

#plt.plot(tilt)
plt.plot(rot)
plt.show()
all = np.vstack((X,Y,Z,A,B,C,tilt,rot))

np.save(f"RealG.npy", all)