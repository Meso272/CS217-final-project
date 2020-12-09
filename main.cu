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
     int n_samples=atoi(argv[2]);
     int dim=atoi(argv[3]);
     int k=atoi(argv[4]);
     int elkan=atoi(argv[5]);
     int iwise=0;
     int distkernelconf=0;
     if(argc>=7)
         distkernelconf=atoi(argv[6]);
     int labelconf=0;
     if(argc>=8)
         labelconf=atoi(argv[7]);
     if(argc>=9)
         iwise=atoi(argv[8]);
     int owise=1;
     if (argc>=10)
         owise=atoi(argv[9]);  
     float * data = (float*) malloc( sizeof(float)*n_samples*dim );
     float * clusters= (float *) malloc( sizeof(float)*k*dim );
     unsigned int * labels = (unsigned int *) malloc( sizeof(unsigned int)*n_samples );
     FILE *file = fopen(input, "r");
     if (iwise==0){
        for(int i=0;i<n_samples*dim;i++){
            float t;
            fscanf(file,"%f",&t);
            data[i]=t;
            //printf("%f\n",t);
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
    
    float times[5]={0};
    int conf[2]={distkernelconf,labelconf};
    //conf[0]: 0: dist+label, 1:_1 ,2:_2, 3:_3
    
    //conf[1]: label: 0: branch 1: nobranch
    //deprecatedconf[2]: 0: 1d accu 1: 2d accu,
    if(elkan==0)
        kmeans(data,clusters,labels,n_samples,dim,k,300,1e-4,iwise,owise,conf,times);
    else 
        kmeans_elkan(data,clusters,labels,n_samples,dim,k,300,1e-4,iwise,owise,conf,times);

    printf("%f\n",times[0]);
    /*
    int counts[k]={0};
    for(int i=0;i<n_samples;i++)
        
        counts[labels[i]]++;
    for(int i=0;i<k;i++)
        printf("%d\n",counts[i]);

    */
    /*
    for(int i=0;i<n_samples;i++)
        if(labels[i]!=0)
            printf("%d\n",labels[i]);
    */
    
    /*
    float *data_d, *clusters_d;
    unsigned int * labels_d;
    float *distances_h,*newclusters_h,*interia_h;
    unsigned int *counts_h;
    float *distances_d,*newclusters_d,*interia_d,*centerdists_d;
    unsigned int*counts_d;
    unsigned int interia_size=(n_samples-1)/(2*BLOCK_SIZE)+1;
    distances_h=(float*) malloc( sizeof(float)*n_samples*k );
    newclusters_h=(float*) malloc( sizeof(float)*k*dim );
    counts_h=(unsigned int*) malloc( sizeof(unsigned int)*k );
    interia_h=(float*) malloc( sizeof(float)*interia_size );
    memset(distances_h,0,sizeof(float)*n_samples*k);
    memset(newclusters_h,0,sizeof(float)*k*dim);
    memset(counts_h,0,sizeof(unsigned int)*k);
    cudaMalloc((void **)&data_d,sizeof(float)*n_samples*dim);
    cudaMalloc((void **)&clusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&labels_d,sizeof(unsigned int)*n_samples);
    cudaMalloc((void **)&distances_d,sizeof(float)*n_samples*k); 
    cudaMalloc((void **)&newclusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&counts_d,sizeof(unsigned int)*k);
    cudaMalloc((void **)&interia_d,sizeof(float)*interia_size);
    cudaMalloc((void **)&centerdists_d,sizeof(float)*n_samples);
    cudaMemcpy(data_d,data,sizeof(float)*n_samples*dim,cudaMemcpyHostToDevice);
    cudaMemcpy(clusters_d,clusters,sizeof(float)*k*dim,cudaMemcpyHostToDevice);
    cudaMemcpy(newclusters_d,newclusters_h,sizeof(float)*k*dim,cudaMemcpyHostToDevice);
    cudaMemcpy(counts_d,counts_h,sizeof(unsigned int)*k,cudaMemcpyHostToDevice);
    Timer timer;
    startTime(&timer);
    calcDistances_2D(data_d,clusters_d,distances_d,n_samples,dim,k,iwise,owise);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    cudaMemcpy(distances_h,distances_d,sizeof(float)*n_samples*k,cudaMemcpyDeviceToHost);


    float* distances_ans=(float*) malloc( sizeof(float)*n_samples*k );
    memset(distances_ans,0,sizeof(float)*n_samples*k );
    for(int i=0;i<n_samples;i++){
        for(int j=0;j<k;j++){
            float dist=0;
            for(int l=0;l<dim;l++){
                float t;
                if(iwise==0)
                    
                    t=data[i*dim+l]-clusters[j*dim+l];
                else
                    t=data[l*n_samples+i]-clusters[j*dim+l];
                dist+=t*t;
            }
            if (owise==0)
                distances_ans[i*k+j]=dist;
            else
                distances_ans[j*n_samples+i]=dist;

        }
    }
    int correct=1;
    for(int i=0;i<n_samples*k;i++){
        float diff=(distances_h[i]-distances_ans[i])/distances_ans[i];
        
        if(diff>1e-4 or diff<-1e-4){
            printf("%d,%.4f,%.4f\n",i,distances_h[i],distances_ans[i]);
            correct=0;
            break;
        }

    }
    printf("%d\n",correct);
    unsigned int * labels=(unsigned int *)malloc(sizeof(unsigned int)*n_samples);
    startTime(&timer);
    calcLabels(distances_d,labels_d,n_samples,dim,k,owise);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    cudaMemcpy(labels,labels_d,sizeof(float)*n_samples,cudaMemcpyDeviceToHost);
    

    unsigned int * labels_ans=(unsigned int *)malloc(sizeof(unsigned int)*n_samples);


    for(int i=0;i<n_samples;i++){
        
        unsigned int start_pos,stride;
        if (owise==0){
            start_pos=i*k;
               
            stride=1;
        }
        else{
            start_pos=i;
		
            stride=n_samples;
        }
	unsigned int argmin=0;
        float mindist=distances_ans[start_pos];
        for(int j=0;j<k;j++){
            float t=distances_ans[start_pos];
            if(t<mindist){
                argmin=j;
                mindist=t;       
     
            }
	    start_pos+=stride;

        }
        labels_ans[i]=argmin;
    }
    correct=1;
    for(int i=0;i<n_samples;i++){
        
        
        if(labels[i]!=labels_ans[i]){
            printf("%d %d %d\n",i,labels[i],labels_ans[i]);
            correct=0;
            break;
        }

    }
    printf("%d\n",correct);
    
    startTime(&timer);
    accumulateNewClusters_2d(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,iwise);
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    cudaMemcpy(newclusters_h,newclusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToHost);
    cudaMemcpy(counts_h,counts_d,sizeof(unsigned int)*k,cudaMemcpyDeviceToHost);
    float * new_clusters_ans=(float *)malloc(sizeof(float)*k*dim); 
    unsigned int counts_ans[k]={0};
    memset(new_clusters_ans,0,sizeof(float)*k*dim);
    
    for(int i=0;i<n_samples;i++){
        unsigned int lbl=labels_ans[i];
        counts_ans[lbl]++;
        for(int j=0;j<dim;j++){
            if(iwise==0)
                new_clusters_ans[lbl*dim+j]+=data[i*dim+j];
            else
                new_clusters_ans[lbl*dim+j]+=data[j*n_samples+i];


        }
   
    }

    correct=1;
    for(int i=0;i<k;i++){
        
        
        if(counts_h[i]!=counts_ans[i]){
            printf("%d %d %d\n",i,counts_h[i],counts_ans[i]);
            correct=0;
            break;
        }

    }
    printf("%d\n",correct);


    correct=1;
    for(int i=0;i<k*dim;i++){
        
        float diff=(newclusters_h[i]-new_clusters_ans[i])/new_clusters_ans[i];
        //printf("%.6f\n",diff);
        if(diff>1e-4 or diff<-1e-4){
            printf("%d,%.4f,%.4f\n",i,newclusters_h[i],new_clusters_ans[i]);
            correct=0;
            break;
        }

    }

    printf("%d\n",correct);

    startTime(&timer);
    averageNewClusters_1d(newclusters_d,counts_d,dim,k);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    cudaMemcpy(newclusters_h,newclusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<k;i++){
        for(int j=0;j<dim;j++){
            new_clusters_ans[i*dim+j]/=counts_ans[i];
        }
    }

    correct=1;
    for(int i=0;i<k*dim;i++){
        
        float diff=(newclusters_h[i]-new_clusters_ans[i])/new_clusters_ans[i];
        //printf("%.6f\n",diff);
        if(diff>1e-4 or diff<-1e-4){
            printf("%d,%.4f,%.4f\n",i,newclusters_h[i],new_clusters_ans[i]);
            correct=0;
            break;
        }

    }
    printf("%d\n",correct);


    startTime(&timer);
    calcInteria_1d(data_d,newclusters_d,labels_d,centerdists_d,interia_d,n_samples,dim,iwise);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    cudaMemcpy(interia_h,interia_d,sizeof(float)*interia_size,cudaMemcpyDeviceToHost);
    float interia=0;
    for(int i=0;i<interia_size;i++)
        interia+=interia_h[i];
  
    float interia_ans=0;
    for(int i=0;i<n_samples;i++){
        unsigned int lbl=labels_ans[i];
        
        for(int j=0;j<dim;j++){
            float t;
            if(iwise==0)
                t=new_clusters_ans[lbl*dim+j]-data[i*dim+j];
            else
                t=new_clusters_ans[lbl*dim+j]-data[j*n_samples+i];
            interia_ans+=t*t;

        }
   
    } 
    printf("%.4f %.4f\n", interia, interia_ans);



    

    free(data);
    free(clusters);
    free(labels);
    free(labels_ans);
    free(distances_h);
    free(newclusters_h);
    free(counts_h);
    free(distances_ans);
    free(new_clusters_ans);
    free(interia_h);
    */
    return 0;
}


    


