import os
import sys

data=sys.argv[1]
output=sys.argv[2]
repeat_time=int(sys.argv[3])


#kernels=["distlabel","1_1","1_2","1_3"]
cnums=[100]

with open(output,"w") as f:
    f.write("cnum\tfull\telkan\n")
    for cnum in cnums:
        f.write(str(cnum)+"\t")
        
        time=0
        for k in range(repeat_time):
                   
            command="python kmeans.py %s %d full " % (data,cnum)
            r=os.popen(command)
            t=float(r.read())
            
            time+=t
        f.write(str(time/repeat_time)+"\t")
        time=0
        for k in range(repeat_time):
            command="python kmeans.py %s %d elkan" % (data,cnum)
            r=os.popen(command)
            t=float(r.read())
            time+=t
        f.write(str(time/repeat_time)+"\n")
