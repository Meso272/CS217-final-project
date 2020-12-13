import os
import sys

data=sys.argv[1]
output=sys.argv[2]
repeat_time=int(sys.argv[3])
n_samples=int(sys.argv[4])
dim=int(sys.argv[5])

#kernels=["distlabel","1_1","1_2","1_3"]
cnums=[50,100]

with open(output,"w") as f:
    f.write("cnum\tdistlabel_nb\telkan\n")
    for cnum in cnums:
        f.write(str(cnum)+"\t")
        
        time=0
        for k in range(repeat_time):
                   
            command="./final %s %d %d %d 0 %d %d" % (data,n_samples,dim,cnum,0,1)
            r=os.popen(command)
            t=float(r.read())
            time+=t
        f.write(str(time/repeat_time)+"\t")
        time=0
        for k in range(repeat_time):
            command="./final %s %d %d %d 1 %d %d" % (data,n_samples,dim,cnum,0,0)
            r=os.popen(command)
            t=float(r.read())
            time+=t
        f.write(str(time/repeat_time)+"\n")
