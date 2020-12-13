/******************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ******************************************************************************/
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[]){
     char * input=argv[1];
     char * output=argv[2];
     int n_samples=atoi(argv[3]);
     int dim=atoi(argv[4]);
     int k=atoi(argv[5]);
     int elkan=atoi(argv[6]);
     int iwise=0;
     int distkernelconf=0;
     if(argc>=8)
         distkernelconf=atoi(argv[7]);
     int labelconf=0;
     if(argc>=9)
         labelconf=atoi(argv[8]);
     if(argc>=10)
         iwise=atoi(argv[9]);
     int owise=1;
     if (argc>=11)
         owise=atoi(argv[10]);  
     float * data = (float*) malloc( sizeof(float)*n_samples*dim );
     float * clusters= (float *) malloc( sizeof(float)*k*dim );
     unsigned int * labels = (unsigned int *) malloc( sizeof(unsigned int)*n_samples );
     FILE *file = fopen(input, "r");
     if (iwise==0){
        for(int i=0;i<n_samples*dim;i++){
            float t;
            fscanf(file,"%f",&t);
            data[i]=t;
            
        }
     }
     
     else{
        for(int j=0;j<n_samples;j++){
            for(int i=0;i<dim;i++){
                float t;
                fscanf(file,"%f",&t);
                data[i*n_samples+j]=t;
                
            }
        }
    }
    fclose(file);
    float times[5]={0};
    int conf[2]={distkernelconf,labelconf};
    
    if(elkan==0)
        kmeans(data,clusters,labels,n_samples,dim,k,300,1e-4,iwise,owise,conf,times);
    else 
        kmeans_elkan(data,clusters,labels,n_samples,dim,k,300,1e-4,iwise,owise,conf,times);

    printf("Clustering finished in %f seconds.\n",times[0]);
    
    file=fopen(output,"w");
    for(int i=0;i<n_samples;i++)
        fprintf(file,"%d\n",labels[i]);
    fclose(file);
    return 0;
}


    


