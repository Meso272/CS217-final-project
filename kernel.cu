/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include <math.h>
#define BLOCK_SIZE 512
#define THREAD_RATIO 0.9



//calcDistances_kernel version 1.1: each thread calculate all k distances for one datapoint.
__global__
void calcDistances_kernel_1_1(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int d_start_pos,j_stride;
        if (i_c==0){
            d_start_pos=idx*dim;
               
            j_stride=1;
        }
        else{
            d_start_pos=idx;
		
            j_stride=n_samples;
        }
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int c_pos=i*dim,d_pos=d_start_pos;
           
            for(int j=0;j<dim;j++){
       
                float t=(data_d[d_pos]-clusters_d[c_pos+j]);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if(o_c==0)
                distances_d[idx*k+i]=sqrt(dist);
            else
                distances_d[i*n_samples+idx]=sqrt(dist);
         }
         idx+=stride;
    }
}

//#define MAX_SHARED_CLUSTER_SIZE 4096
//calcDistances_kernel version 1.2: shared memory for cluster centers.

__global__
void calcDistances_kernel_1_2(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c,int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<shared_memory_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    while(idx<n_samples){
       
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=i*dim;
            if (i_c==0){
                d_pos=idx*dim;
               
		j_stride=1;
            }
            else{
                d_pos=idx;
		
                j_stride=n_samples;
            }
            for(int j=0;j<dim;j++){
                float c;
                if(c_pos+j<shared_memory_size)
                    c=clusters[c_pos+j];
                else
                    c=clusters_d[c_pos+j];
                float t=(data_d[d_pos]-c);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if(o_c==0)
                distances_d[idx*k+i]=sqrt(dist);
            else
                distances_d[i*n_samples+idx]=sqrt(dist);
         }
         idx+=stride;
    }
}


//#define MAX_SHARED_DATA_SIZE_2 4096
//#define MAX_SHARED_CLUSTER_SIZE_2 4096
//calcDistances_kernel version 1.3: shared memory for cluster centers and data points.

__global__
void calcDistances_kernel_1_3(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c,int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    int part_size=shared_memory_size/2;
    float *data=(float*)&clusters[part_size];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<part_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    for(int n=0;n<(n_samples-1)/stride+1;n++){
        unsigned int d_start_pos=idx*dim,start_pos=tx*dim;
        if (idx<n_samples){
            if (i_c==0){
                for(int j=0;j<dim;j++){
                data[start_pos+j]=data_d[d_start_pos+j];
                } 
            }    
            else{
                unsigned int start_pos=tx,start_pos_d=idx;
                for(int j=0;j<dim;j++){
                    data[start_pos]=data_d[start_pos_d];
                    start_pos+=blockDim.x;
                    start_pos_d+=n_samples;
                }   
             }  
        }
        __syncthreads(); 
        if (idx<n_samples){
            for(int i=0;i<k;i++){

                float dist=0;
                unsigned int d_pos,j_stride,c_pos=i*dim;
                if (i_c==0){
                    d_pos=tx*dim;
               
		    j_stride=1;
                }
                else{
                    d_pos=tx;
		
                    j_stride=blockDim.x;
                }
                for(int j=0;j<dim;j++){
                   
                    float t=(data[d_pos]-clusters[c_pos+j]);
                    dist+=t*t;
                    d_pos+=j_stride;
                }
                if(o_c==0)
                    distances_d[idx*k+i]=sqrt(dist);
                else
                    distances_d[i*n_samples+idx]=sqrt(dist);
             }
         }
         idx+=stride;
    }
}



// calcDistances version 1: each thread calculate all k distances for one datapoint.
void calcDistances(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c,int kernel_mode,int maxsharedmemoryperblock,int maxthreadspermultiprocessor,int maxsharedmemorypermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*0.9);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    unsigned int a=(maxsharedmemorypermultiprocessor*block_size)/maxthreadspermultiprocessor;
    unsigned int shared_memory_size=a<maxsharedmemoryperblock?a:maxsharedmemoryperblock;
    shared_memory_size=(unsigned int)((float)shared_memory_size * THREAD_RATIO);
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    if (kernel_mode==1)
        calcDistances_kernel_1_1<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c);
    else if(kernel_mode==2)
        calcDistances_kernel_1_2<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c,shared_memory_size/sizeof(float));
    else
        calcDistances_kernel_1_3<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c,shared_memory_size/sizeof(float));
}
//calcDistances_kernel version 2.1

__global__
void calcDistances_kernel_2_1(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c){
    unsigned int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    unsigned int idx=bx*blockDim.x+tx,stridex=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int idy=by*blockDim.y+ty,stridey=gridDim.y*blockDim.y;
       
        while(idy<k){
           
           
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=idy*dim;
            if (i_c==0){
                d_pos=idx*dim;
               
		j_stride=1;
            }
            else{
                d_pos=idx;
		
                j_stride=n_samples;
            }
            for(int j=0;j<dim;j++){
                float t=(data_d[d_pos]-clusters_d[c_pos+j]);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if(o_c==0)
                distances_d[idx*k+idy]=sqrt(dist);
            else
                distances_d[idy*n_samples+idx]=sqrt(dist);
            idy+=stridey;
       }
        
    idx+=stridex;
    }
}


//calcDistances version 2: 2D blocks.
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
void calcDistances_2D(float * data_d,float *clusters_d,float * distances_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,unsigned int o_c){
    const unsigned int block_size_x=BLOCK_SIZE_X, block_size_y=BLOCK_SIZE_Y;
    unsigned int grid_size_x=(n_samples-1)/block_size_x+1;
    unsigned int grid_size_y=(k-1)/block_size_y+1;
    grid_size_x=grid_size_x<16?grid_size_x:16;
    grid_size_y=grid_size_y<2?grid_size_y:2;
    dim3 dimGrid(grid_size_x,grid_size_y,1);
    dim3 dimBlock(block_size_x,block_size_y,1);
    calcDistances_kernel_2_1<<<dimGrid,dimBlock>>>(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c);
}

//calcLabelskernel: 1D version 1. 1D thread block, each thread for each point .
__global__
void calcLabels_kernel_1_1(float *distances_d,unsigned int *labels_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c){


    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int d_start_pos,i_stride;
        if (i_c==0){
            d_start_pos=idx*k;
               
            i_stride=1;
        }
        else{
            d_start_pos=idx;
		
            i_stride=n_samples;
        }
        unsigned int argmin=0;
        float mindist=distances_d[d_start_pos];
        unsigned int d_pos=d_start_pos+i_stride;
        for(int i=1;i<k;i++){
            float dist=distances_d[d_pos];
            if(dist<mindist){
                argmin=i;
                mindist=dist;
            } 
            d_pos+=i_stride;
           
  
            
         }
        labels_d[idx]=argmin;
        idx+=stride;
    }
}

//calcLabelskernel: 1D version 2. No branch.
__global__
void calcLabels_kernel_1_2(float *distances_d,unsigned int *labels_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c){


    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int d_start_pos,i_stride;
        if (i_c==0){
            d_start_pos=idx*k;
               
            i_stride=1;
        }
        else{
            d_start_pos=idx;
		
            i_stride=n_samples;
        }
        unsigned int argmin=0;
        float mindist=distances_d[d_start_pos];
        unsigned int d_pos=d_start_pos+i_stride;
        for(int i=1;i<k;i++){
            float dist=distances_d[d_pos];
            unsigned int sign=((int)(floor(dist-mindist)));
            sign=sign>>31;
            argmin=argmin+sign*(i-argmin);
            mindist=mindist+sign*(dist-mindist);
            d_pos+=i_stride;
           
  
            
         }
        labels_d[idx]=argmin;
        idx+=stride;
    }
}
//calcLabels: 1D version . 
void calcLabels(float *distances_d,unsigned int *labels_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,int label_mode, int maxthreadspermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    if (label_mode>0)
        calcLabels_kernel_1_2<<<dimGrid,dimBlock>>>(distances_d,labels_d,n_samples,dim,k,i_c);
    else
        calcLabels_kernel_1_1<<<dimGrid,dimBlock>>>(distances_d,labels_d,n_samples,dim,k,i_c);
}


//to complete: calclabels 2d version.

//#define MAX_PRIVATE_SIZE 4096
//accumulation: full privatized.
__global__
void accumulateNewClusters_kernel_privatized_full_1d(float * data_d,unsigned int * labels_d,float * newclusters_d, unsigned int * counts_d,unsigned int n_samples,unsigned int dim,unsigned int k,int i_c, int shared_memory_size){
    //extern __shared__ unsigned int private_hist[];
    extern __shared__ float private_clusters[];
    unsigned int private_clusters_size=(unsigned int)((float)shared_memory_size*0.8);
    unsigned int private_hist_size=shared_memory_size-private_clusters_size;
    unsigned int * private_hist=(unsigned int *)&private_clusters[private_clusters_size];
    
    int j=threadIdx.x;
    while(j<private_clusters_size && j< dim*k){
        private_clusters[j]=0;
        j+=blockDim.x;
    }

    j=threadIdx.x;
    while(j<private_hist_size && j< k){
        private_hist[j]=0;
        j+=blockDim.x;
    }
    __syncthreads();

    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int stride=blockDim.x*gridDim.x;

    while (i<n_samples){
        unsigned int label=labels_d[i];
        if (label<private_hist_size)
            atomicAdd( & ( private_hist[ label ] ),1 );
            
        else
            atomicAdd( & ( labels_d[ label ] ),1 );
        
        for (int j=0;j<dim;j++){
            int idx=label*dim+j;
            float element;
            if(i_c==0)
                element=data_d[i*dim+j];
            else
                element=data_d[j*n_samples+i];
            if(idx<private_clusters_size)
                atomicAdd( & ( private_clusters[ idx] ),element );    
            else
                atomicAdd( & ( newclusters_d[ idx] ),element );    
         }
        i+=stride;
    }

    __syncthreads();
    j=threadIdx.x;
    while(j<private_clusters_size && j< dim*k){
        atomicAdd( & ( newclusters_d[j]),private_clusters[j]);
        j+=blockDim.x;
    }
    j=threadIdx.x;
    while(j<private_hist_size && j< k){
        atomicAdd( & ( counts_d[j]),private_hist[j]);
        j+=blockDim.x;
    }
    

}

//accumulation 1d
void accumulateNewClusters_1d(float * data_d,unsigned int * labels_d,float * newclusters_d, unsigned int * counts_d,unsigned int n_samples,unsigned int dim,unsigned int k,int i_c,int maxsharedmemoryperblock,int maxthreadspermultiprocessor,int maxsharedmemorypermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    unsigned int a=(maxsharedmemorypermultiprocessor*block_size)/maxthreadspermultiprocessor;
    unsigned int shared_memory_size=a<maxsharedmemoryperblock?a:maxsharedmemoryperblock;
    shared_memory_size=(unsigned int)((float)shared_memory_size * THREAD_RATIO);
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);

    accumulateNewClusters_kernel_privatized_full_1d<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,i_c,shared_memory_size/sizeof(float));
}

/*
//accumulation: full privatized 2d.
__global__
void accumulateNewClusters_kernel_privatized_full_2d(float * data_d,unsigned int * labels_d,float * newclusters_d, unsigned int * counts_d,unsigned int n_samples,unsigned int dim,unsigned int k,int i_c){
    __shared__ unsigned int private_hist[MAX_PRIVATE_SIZE];
    __shared__ float private_clusters[MAX_PRIVATE_SIZE];
    unsigned int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    unsigned int idx=bx*blockDim.x+tx,stridex=gridDim.x*blockDim.x;
    unsigned int idy=by*blockDim.y+ty,stridey=gridDim.y*blockDim.y;

    unsigned int idj=threadIdx.y*blockDim.x+threadIdx.x,blocksize=blockDim.x*blockDim.y;
    while(idj<MAX_PRIVATE_SIZE && idj< k){
        private_hist[idj]=0;
        idj+=blocksize;
    }
    idj=threadIdx.y*blockDim.x+threadIdx.x;
    while(idj<MAX_PRIVATE_SIZE && idj< dim*k){
        private_clusters[idj]=0;
        idj+=blocksize;
    }
    __syncthreads();

    int i=threadIdx.x+blockIdx.x*blockDim.x;
   
    
    while (i<n_samples){
        unsigned int label=labels_d[i];
        if(blockIdx.y==0 and threadIdx.y==0){
            if (label<MAX_PRIVATE_SIZE)
                atomicAdd( & ( private_hist[ label ] ),1 );
            
            else
                atomicAdd( & ( labels_d[ label ] ),1 );
        }
 
        
        for (int j=blockIdx.y*blockDim.y+threadIdx.y;j<dim;j+=stridey){
            int idx=label*dim+j;
            float element;
            if(i_c==0)
                element=data_d[i*dim+j];
            else
                element=data_d[j*n_samples+i];
            if(idx<MAX_PRIVATE_SIZE)
                atomicAdd( & ( private_clusters[ idx] ),element );    
            else
                atomicAdd( & ( newclusters_d[ idx] ),element );    
         }
        i+=stridex;
    }

    __syncthreads();
    idj=threadIdx.y*blockDim.x+threadIdx.x;
    while(idj<MAX_PRIVATE_SIZE && idj< k){
        atomicAdd( & ( counts_d[idj]),private_hist[idj]);
        idj+=blocksize;
    }
    idj=threadIdx.y*blockDim.x+threadIdx.x;
    while(idj<MAX_PRIVATE_SIZE && idj< dim*k){
        atomicAdd( & ( newclusters_d[idj]),private_clusters[idj]);
        idj+=blocksize;
    }

}

//accumulation 2d

void accumulateNewClusters_2d(float * data_d,unsigned int * labels_d,float * newclusters_d, unsigned int * counts_d,unsigned int n_samples,unsigned int dim,unsigned int k,int i_c){
    const unsigned int block_size_x=BLOCK_SIZE_X, block_size_y=BLOCK_SIZE_Y;
    unsigned int grid_size_x=(n_samples-1)/block_size_x+1;
    unsigned int grid_size_y=(dim-1)/block_size_y+1;
    grid_size_x=grid_size_x<16?grid_size_x:16;
    grid_size_y=grid_size_y<2?grid_size_y:2;
    dim3 dimGrid(grid_size_x,grid_size_y,1);
    dim3 dimBlock(block_size_x,block_size_y,1);
    accumulateNewClusters_kernel_privatized_full_2d<<<dimGrid,dimBlock>>>(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,i_c);
}
*/


__global__
void averageNewClusters_kernel_1d(float * newclusters_d,float * clusters_d,unsigned int * counts_d,unsigned int  dim,unsigned int k){
    unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x, stride=blockDim.x*gridDim.x;
    while(idx<dim*k){
        unsigned int cur=idx/dim;
        if (counts_d[cur]>0)
            newclusters_d[idx]=newclusters_d[idx]/counts_d[cur];
        else
            newclusters_d[idx]=clusters_d[idx];
        idx+=stride;
    }
    

}

void averageNewClusters_1d(float * newclusters_d,float * clusters_d,unsigned int * counts_d,unsigned int  dim,unsigned int k,int maxthreadspermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=dim*k<max_thread_num?dim*k:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    averageNewClusters_kernel_1d<<<dimGrid,dimBlock>>>(newclusters_d,clusters_d,counts_d,dim,k);
}

/*
//calc interia 1d kernel, aggregate and reduce todo: privatized
__global__
void calcInteria_kernel_1d(float * data_d, float * clusters_d, unsigned int * labels_d, float * centerdists_d,float * interia_d,unsigned int n_samples,unsigned int dim,int i_c){
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    while(idx<n_samples){
        float t=0;
        unsigned int lbl=labels_d[idx];
        for(int j=0;j<dim;j++){
            float r;
            if(i_c==0)
                r=data_d[idx*dim+j]-clusters_d[lbl*dim+j];
            else
                r=data_d[j*n_samples+idx]-clusters_d[lbl*dim+j];
            t+=r*r;
        }
        centerdists_d[idx]=t;
        idx+=stride;
    }
    __syncthreads();

    __shared__ float psum[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    if(start+t<n_samples){
        psum[t]=centerdists_d[start+t];
    }
    else{
	centerdists_d[t]=0;
    }
    if(start+blockDim.x+t<n_samples){
        psum[blockDim.x+t]=centerdists_d[start + blockDim.x+t];
    }
    else{
        psum[blockDim.x+t]=0;
    }
    for (unsigned int stride = 1;stride <= blockDim.x; stride *= 2){
	__syncthreads();
	if (t % stride == 0){
	    psum[2*t]+= psum[2*t+stride];
        }
    }
    __syncthreads();
    interia_d[blockIdx.x]=psum[0];
}
              

void calcInteria_1d(float * data_d, float * clusters_d, unsigned int * labels_d, float * centerdists_d,float * interia_d,unsigned int n_samples,unsigned int dim,int i_c)  {
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int grid_size=(n_samples-1)/(2*block_size)+1;
    //grid_size=grid_size<32?grid_size:32;
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    calcInteria_kernel_1d<<<dimGrid,dimBlock>>>(data_d,clusters_d,labels_d,centerdists_d,interia_d,n_samples,dim,i_c);
}
*/
__global__
void calcClusterDist_kernel(float * clusters_d, float *newclusters_d,float * clusterdists_d,unsigned int dim, unsigned int k){
    unsigned int idx=blockDim.x*blockIdx.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    while(idx<k){
        float t=0;
        
        for(int j=0;j<dim;j++){
            float r=newclusters_d[idx*dim+j]-clusters_d[idx*dim+j];
            
            t+=r*r;
        }
        clusterdists_d[idx]=sqrt(t);
        idx+=stride;
    }
}

void calcClusterDist(float * clusters_d, float *newclusters_d,float * clusterdists_d,unsigned int dim, unsigned int k){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int grid_size=(k-1)/block_size+1;
    grid_size=grid_size<32?grid_size:32;
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    calcClusterDist_kernel<<<dimGrid,dimBlock>>>(clusters_d, newclusters_d,clusterdists_d, dim,  k);
}

/*
//calcLabelskernel elkan: 1D version 1. 1D thread block, each thread for each point .
__global__
void calcLabels_elkan_kernel_1_1(float *distances_d,unsigned int *labels_d,float *lb_d,float * ub_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c){


    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int d_start_pos,i_stride;
        if (i_c==0){
            d_start_pos=idx*k;
               
            i_stride=1;
        }
        else{
            d_start_pos=idx;
		
            i_stride=n_samples;
        }
        unsigned int argmin=0;
        float mindist=distances_d[d_start_pos],secondmindist=-1;
        unsigned int d_pos=d_start_pos+i_stride;
        for(int i=1;i<k;i++){
            float dist=distances_d[d_pos];
            if(dist<mindist){
                argmin=i;
                secondmindist=mindist;
                mindist=dist;
            } 
            else if(secondmindist==-1 or dist<secondmindist)
                secondmindist=dist;
            d_pos+=i_stride;
           
  
            
         }
        labels_d[idx]=argmin;
        idx+=stride;
        ub_d[idx]=mindist;
        lb_d[idx]=secondmindist;
    }
}

//calcLabelskernel elkan: 1D version 2. No branch.

__global__
void calcLabels_kernel_elkan_1_2(float *distances_d,unsigned int *labels_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c){


    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;

    while(idx<n_samples){
        unsigned int d_start_pos,i_stride;
        if (i_c==0){
            d_start_pos=idx*k;
               
            i_stride=1;
        }
        else{
            d_start_pos=idx;
		
            i_stride=n_samples;
        }
        unsigned int argmin=0;
        float mindist=distances_d[d_start_pos];
        unsigned int d_pos=d_start_pos+i_stride;
        for(int i=1;i<k;i++){
            float dist=distances_d[d_pos];
            unsigned int sign=((int)(floor(dist-mindist)));
            sign=sign>>31;
            argmin=argmin+sign*(i-argmin);
            mindist=mindist+sign*(dist-mindist);
            d_pos+=i_stride;
           
  
            
         }
        labels_d[idx]=argmin;
        idx+=stride;
    }
}

//calcLabels elkan: 1D version . 
void calcLabels_elkan(float *distances_d,unsigned int *labels_d,float * lb_d,float * ub_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int grid_size=(n_samples-1)/block_size+1;
    grid_size=grid_size<32?grid_size:32;
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    calcLabels_elkan_kernel_1_1<<<dimGrid,dimBlock>>>(distances_d,labels_d,lb_d,ub_d,n_samples,dim,k,i_c);
}

*/
__global__
void calcDistancesLabels_kernel_1_2(float * data_d,float *clusters_d,unsigned int * labels_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<shared_memory_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    while(idx<n_samples){
        float mindist=-1;
        unsigned int argmin=-1;
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=i*dim;
            if (i_c==0){
                d_pos=idx*dim;
               
		j_stride=1;
            }
            else{
                d_pos=idx;
		
                j_stride=n_samples;
            }
            for(int j=0;j<dim;j++){
                float c;
                if(c_pos+j<shared_memory_size)
                    c=clusters[c_pos+j];
                else
                    c=clusters_d[c_pos+j];
                float t=(data_d[d_pos]-c);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if (mindist==-1 or dist<mindist){
                mindist=dist;
                argmin=i;
            }

         }
         labels_d[idx]=argmin;
         idx+=stride;
    }
}


__global__
void calcDistancesLabels_kernel_1_3(float * data_d,float *clusters_d,unsigned int * labels_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<shared_memory_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    while(idx<n_samples){
        float mindist=-1;
        unsigned int argmin=-1;
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=i*dim;
            if (i_c==0){
                d_pos=idx*dim;
               
		j_stride=1;
            }
            else{
                d_pos=idx;
		
                j_stride=n_samples;
            }
            for(int j=0;j<dim;j++){
                float c;
                if(c_pos+j<shared_memory_size)
                    c=clusters[c_pos+j];
                else
                    c=clusters_d[c_pos+j];
                float t=(data_d[d_pos]-c);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if (mindist==-1 ){
                mindist=dist;
                argmin=i;
            }
            else{
                unsigned int sign=((int)(floor(dist-mindist)));
                sign=sign>>31;
                argmin=argmin+sign*(i-argmin);
                mindist=mindist+sign*(dist-mindist); 
            }

         }
         labels_d[idx]=argmin;
         idx+=stride;
    }
}


void calcDistancesLabels(float * data_d,float *clusters_d,unsigned int * labels_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,int mode,int maxsharedmemoryperblock,int maxthreadspermultiprocessor,int maxsharedmemorypermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    unsigned int a=(maxsharedmemorypermultiprocessor*block_size)/maxthreadspermultiprocessor;
    unsigned int shared_memory_size=a<maxsharedmemoryperblock?a:maxsharedmemoryperblock;
    shared_memory_size=(unsigned int)((float)shared_memory_size * THREAD_RATIO);
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    if (mode==0)   calcDistancesLabels_kernel_1_2<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,labels_d,n_samples,dim,k,i_c,shared_memory_size/sizeof(float));
    else
    calcDistancesLabels_kernel_1_3<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,labels_d,n_samples,dim,k,i_c,shared_memory_size/sizeof(float));
}

__global__
void calcDistancesLabels_elkan_kernel_1_2(float * data_d,float *clusters_d,unsigned int * labels_d, float *lb_d,float *ub_d,unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c, int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<shared_memory_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    while(idx<n_samples){
        float mindist=-1,secondmindist=-1;
        unsigned int argmin=-1;
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=i*dim;
            if (i_c==0){
                d_pos=idx*dim;
               
		j_stride=1;
            }
            else{
                d_pos=idx;
		
                j_stride=n_samples;
            }
            for(int j=0;j<dim;j++){
                float c;
                if(c_pos+j<shared_memory_size)
                    c=clusters[c_pos+j];
                else
                    c=clusters_d[c_pos+j];
                float t=(data_d[d_pos]-c);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if (mindist==-1 or dist<mindist){
                secondmindist=mindist;
                mindist=dist;
                argmin=i;
            }
            else if(secondmindist==-1 or dist<secondmindist)
                secondmindist=dist;

         }
         labels_d[idx]=argmin;
         lb_d[idx]=sqrt(secondmindist);
         ub_d[idx]=sqrt(mindist);
         idx+=stride;
    }
}

void calcDistancesLabels_elkan(float * data_d,float *clusters_d,unsigned int * labels_d,float *lb_d,float *ub_d, unsigned int n_samples,unsigned int dim,unsigned int k,unsigned int i_c,int maxsharedmemoryperblock,int maxthreadspermultiprocessor,int maxsharedmemorypermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    unsigned int a=(maxsharedmemorypermultiprocessor*block_size)/maxthreadspermultiprocessor;
    unsigned int shared_memory_size=a<maxsharedmemoryperblock?a:maxsharedmemoryperblock;
    shared_memory_size=(unsigned int)((float)shared_memory_size * THREAD_RATIO);
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    calcDistancesLabels_elkan_kernel_1_2<<<dimGrid,dimBlock,shared_memory_size>>>(data_d,clusters_d,labels_d,lb_d,ub_d,n_samples,dim,k,i_c,shared_memory_size/sizeof(float));
}
//Host function
//data: data points. n_samples x dim
//cluster: output clusters. k x dim
//labels: cluster id for each sample. n_samples x1
//dim: dimension of data
//k: number of clusters
//max_iter: maximum iteration rounds
//epsilon: terminate threshold
//all pointers are in host


void kmeans(float *data,float *clusters,unsigned int *labels, unsigned int n_samples,unsigned int dim, unsigned int k, unsigned int max_iter, float epsilon, int i_c,int o_c, int* configurations, float * times){
    
    int maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount;

    //cudaDeviceGetAttribute(&maxthreadsperblock,cudaDevAttrMaxThreadsPerBlock,0);
    cudaDeviceGetAttribute(&maxsharedmemoryperblock,cudaDevAttrMaxSharedMemoryPerBlock,0);
    cudaDeviceGetAttribute(&maxthreadspermultiprocessor,cudaDevAttrMaxThreadsPerMultiProcessor,0);
    cudaDeviceGetAttribute(&maxsharedmemorypermultiprocessor,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);
    cudaDeviceGetAttribute(&multiprocessorcount,cudaDevAttrMultiProcessorCount,0);

   // printf("%d %d %d %d %d\n",maxthreadsperblock,maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount);

    Timer totaltimer,timer;
    startTime(&totaltimer);
    //part 1 alloc and memsets

    float *data_d, *clusters_d;
    unsigned int * labels_d;
    float *clusterdists_h;
   
    float *distances_d,*newclusters_d,*centerdists_d,*clusterdists_d;
    unsigned int *counts_d;
    //unsigned int interia_size=(n_samples-1)/(2*BLOCK_SIZE)+1;
    
    
    //interia_h=(float*) malloc( sizeof(float)*interia_size );
    clusterdists_h=(float*) malloc( sizeof(float)*k );
    cudaMalloc((void **)&data_d,sizeof(float)*n_samples*dim);
    cudaMalloc((void **)&clusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&labels_d,sizeof(unsigned int)*n_samples);
    if(configurations[0]!=0)
       cudaMalloc((void **)&distances_d,sizeof(float)*n_samples*k); 
    cudaMalloc((void **)&newclusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&counts_d,sizeof(unsigned int)*k);
    //cudaMalloc((void **)&interia_d,sizeof(float)*interia_size);
    cudaMalloc((void **)&centerdists_d,sizeof(float)*n_samples);
    cudaMalloc((void **)&clusterdists_d,sizeof(float)*k);
    cudaMemcpy(data_d,data,sizeof(float)*n_samples*dim,cudaMemcpyHostToDevice);
    cudaMemcpy(clusters_d,clusters,sizeof(float)*k*dim,cudaMemcpyHostToDevice);
    
    
   
    //part 1 end


    //part 2: initialize clusters and copy
    unsigned int initial_clusters[k];
    for(int i=0;i<k;i++){
        while(1){
            unsigned int r=rand()%n_samples; 
            bool duplicated=false;
            for(int j=0;j<i;j++){
                if (r==initial_clusters[j]){
                    duplicated=true;
                    break;
                }
            }
            if (!duplicated){
                initial_clusters[i]=r;
                break;
            }
       }
    }
    for(int i=0;i<k;i++){
        unsigned int idx=initial_clusters[i];
        for(int j=0;j<dim;j++){
            if(i_c==0)
                clusters[i*dim+j]=data[idx*dim+j];
            else
                clusters[i*dim+j]=data[j*n_samples+idx];

        }
    }
        
    //part 2 end
  
    
    //part 3 gogogo
    
    //float last_interia=0;
    int kernel_mode=configurations[0],label_mode=configurations[1];
    for(int round=0;round<max_iter;round++){
        
        cudaMemset(newclusters_d,0,sizeof(float)*k*dim);
	cudaMemset(counts_d,0,sizeof(int)*k);
        //cudaMemset(centerdists_d,0,sizeof(float)*n_samples);
        
	if(kernel_mode!=0){

            //startTime(&timer);
            //if (configurations[1]==0)
            calcDistances(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c,kernel_mode,maxsharedmemoryperblock,maxthreadspermultiprocessor, maxsharedmemorypermultiprocessor,multiprocessorcount);
            /*
            else
                calcDistances_2D(data_d,clusters_d,distances_d,n_samples,dim,k,i_c,o_c);
             */
            //cudaDeviceSynchronize();
            //stopTime(&timer); 
            //times[0]+=elapsedTime(timer);


            //startTime(&timer);
            int label_mode=configurations[1];
            calcLabels(distances_d,labels_d,n_samples,dim,k,o_c,label_mode,maxthreadspermultiprocessor,multiprocessorcount);
            //cudaDeviceSynchronize();
            //stopTime(&timer); 
            //times[1]+=elapsedTime(timer);
        }
        else{
            //startTime(&timer);
            calcDistancesLabels(data_d,clusters_d,labels_d,n_samples,dim,k,i_c,label_mode,maxsharedmemoryperblock,maxthreadspermultiprocessor, maxsharedmemorypermultiprocessor,multiprocessorcount);
            //cudaDeviceSynchronize();
            //stopTime(&timer); 
            //times[0]+=elapsedTime(timer);

        }
        //startTime(&timer);
        
            accumulateNewClusters_1d(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,i_c,maxsharedmemoryperblock,maxthreadspermultiprocessor, maxsharedmemorypermultiprocessor,multiprocessorcount); 
        
            //accumulateNewClusters_2d(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,i_c); 
        //cudaDeviceSynchronize();
        //stopTime(&timer); 
        //times[2]+=elapsedTime(timer);
     
        //startTime(&timer);
        averageNewClusters_1d(newclusters_d,clusters_d,counts_d,dim,k,maxthreadspermultiprocessor,multiprocessorcount);
        //cudaDeviceSynchronize();
        //stopTime(&timer); 
        //times[3]+=elapsedTime(timer);
        //cudaMemcpy(newclusters_h,newclusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToHost);
   
        /*
        startTime(&timer);
        calcInteria_1d(data_d,clusters_d,labels_d,centerdists_d,interia_d,n_samples,dim,i_c);
        cudaDeviceSynchronize();
        stopTime(&timer); 
        times[4]+=elapsedTime(timer);
        cudaMemcpy(interia_h,interia_d,sizeof(float)*interia_size,cudaMemcpyDeviceToHost);

        
        float interia=0;
        for(int i=0;i<interia_size;i++)
            interia+=interia_h[i];
        */
        //printf("%f %f\n",last_interia,interia);

        //startTime(&timer);
        calcClusterDist(clusters_d, newclusters_d,clusterdists_d, dim,  k);
       // cudaDeviceSynchronize();
        //stopTime(&timer); 
       // times[4]+=elapsedTime(timer);
        cudaMemcpy(clusterdists_h,clusterdists_d,sizeof(float)*k,cudaMemcpyDeviceToHost);

        cudaMemcpy(clusters_d,newclusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToDevice);
        /*
        if(round>max_iter/2 && last_interia-interia<epsilon){
            
            printf("stopped on round %d\n",round);
            
            break;
            
        }
        */
        bool terminate=true;
      
        for(int i=0;i<k;i++){
            //printf("%f\n",clusterdists_h[i]);
            if(clusterdists_h[i]>epsilon){
                
                terminate=false;
                break;
            }
        }

        if(terminate){
            
            //printf("stopped on round %d\n",round);
            
            break;
            
        }
  
        //last_interia=interia;
        //cudaMemcpy(clusters_d,clusters,sizeof(float)*n_samples*k,cudaMemcpyHostToDevice);
        
        
        
     

    }
    cudaMemcpy(clusters,clusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToHost);
    cudaMemcpy(labels,labels_d,sizeof(float)*n_samples,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&totaltimer); times[0]=elapsedTime(totaltimer);
    //free(interia_h);
    free(clusterdists_h);
    cudaFree(data_d);
    cudaFree(clusters_d);
    cudaFree(labels_d);
    if(configurations[0]==0)
        cudaFree(distances_d);
    cudaFree(newclusters_d);
    cudaFree(counts_d);
    //cudaFree(interia_d);
    cudaFree(centerdists_d);

}
//to complete: calclabels 2d version.


//globalfilterkernel
__global__

void globalFilter_kernel(float * data_d,float *lb_d,float *ub_d,unsigned int *labels_d,float *clusterdists_d,float * clusters_d,float *filtered_data_d,unsigned int * filtered_indices_d,unsigned int *filtered_num_d,float maxdelta,unsigned int n_samples,unsigned int dim,int i_c){

    unsigned int idx=blockIdx.x*blockDim.x+threadIdx.x,stride=blockDim.x*gridDim.x;
    while(idx<n_samples){

    float ub=ub_d[idx],lb=lb_d[idx]; 
    unsigned int curcenter=labels_d[idx];
    ub+=clusterdists_d[curcenter];
    lb-=maxdelta;
    if(lb>=ub){
        lb_d[idx]=lb;
        ub_d[idx]=ub;
        idx+=stride;
        continue;
    }
    float dist=0;
    
    unsigned int d_pos,dstride,c_pos=curcenter*dim;
    if(i_c==0){
        d_pos=idx*dim;
        dstride=1;
    }
    else{
        d_pos=idx;
        dstride=n_samples;
    }
    for(int i=0;i<dim;i++){
        float t=data_d[d_pos]-clusters_d[c_pos+i];
        dist+=t*t;
        d_pos+=dstride;
    }
    ub=sqrt(dist);
    if(lb>=ub){
        lb_d[idx]=lb;
        ub_d[idx]=ub;
        idx+=stride;
        continue;
    }
    //lb_d[idx]=lb;
    ub_d[idx]=ub;
    unsigned int pos=atomicAdd(filtered_num_d,1);
    filtered_indices_d[pos]=idx;
    if(i_c==0){
        d_pos=idx*dim;
        dstride=1;
    }
    else{
        d_pos=idx;
        dstride=n_samples;
    }
    unsigned int dest_pos=pos*dim;
    for(int i=0;i<dim;i++){
        filtered_data_d[dest_pos+i]=data_d[d_pos];
        d_pos+=dstride;
    }
    idx+=stride;
    }

}

void globalFilter(float * data_d,float *lb_d,float *ub_d,unsigned int *labels_d,float *clusterdists_d,float * clusters_d,float *filtered_data_d,unsigned int * filtered_indices_d,unsigned int *filtered_num_d,float maxdelta,unsigned int n_samples,unsigned int dim,int i_c,int maxthreadspermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=n_samples<max_thread_num?n_samples:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
                   globalFilter_kernel<<<dimGrid,dimBlock>>>(data_d,lb_d,ub_d,labels_d,clusterdists_d,clusters_d,filtered_data_d,filtered_indices_d,filtered_num_d,maxdelta,n_samples,dim,i_c);

}

__global__
void localFilter_kernel(float * filtered_data_d,unsigned int * filter_indices_d,float * clusters_d,float * lb_d,float * ub_d, unsigned int *labels_d,unsigned int filtered_num,unsigned int maxdelta,unsigned int k,unsigned int dim,int shared_memory_size){
    unsigned int bx=blockIdx.x,tx=threadIdx.x;
    unsigned int idx=bx*blockDim.x+tx,stride=gridDim.x*blockDim.x;
    extern __shared__ float clusters[];
    unsigned int clusteridx=tx;
    while(clusteridx<k*dim && clusteridx<shared_memory_size){
        clusters[clusteridx]=clusters_d[clusteridx];
        clusteridx+=blockDim.x;
    }
    __syncthreads();
    
    while(idx<filtered_num){
        unsigned int data_idx=filter_indices_d[idx];
        float mindist=-1,secondmindist=-1;
        unsigned int argmin=-1;
        for(int i=0;i<k;i++){
            float dist=0;
            unsigned int d_pos,j_stride,c_pos=i*dim;
          
            d_pos=idx*dim;
               
            j_stride=1;
            
            
            for(int j=0;j<dim;j++){
                float c;
                if(c_pos+j<shared_memory_size)
                    c=clusters[c_pos+j];
                else
                    c=clusters_d[c_pos+j];
                float t=(filtered_data_d[d_pos]-c);
                dist+=t*t;
                d_pos+=j_stride;
            }
            if (mindist==-1 or dist<mindist){
                secondmindist=mindist;
                mindist=dist;
                argmin=i;
            }
            else if(secondmindist==-1 or dist<secondmindist)
                secondmindist=dist;

         }
         labels_d[data_idx]=argmin;
         lb_d[data_idx]=sqrt(secondmindist);
         //ub_d[idx]=sqrt(mindist);
         idx+=stride;
    }
}

void localFilter(float * filtered_data_d,unsigned int * filter_indices_d,float * clusters_d,float * lb_d,float * ub_d, unsigned int *labels_d,unsigned int filtered_num,unsigned int maxdelta,unsigned int k,unsigned int dim,int maxsharedmemoryperblock,int maxthreadspermultiprocessor,int maxsharedmemorypermultiprocessor,int multiprocessorcount){
    const unsigned int block_size=BLOCK_SIZE;
    unsigned int max_thread_num=(unsigned int)((float)maxthreadspermultiprocessor*(float)multiprocessorcount*THREAD_RATIO);
    unsigned int thread_num=filtered_num<max_thread_num?filtered_num:max_thread_num;
    unsigned int grid_size=(thread_num-1)/block_size+1;
    unsigned int a=(maxsharedmemorypermultiprocessor*block_size)/maxthreadspermultiprocessor;
    unsigned int shared_memory_size=a<maxsharedmemoryperblock?a:maxsharedmemoryperblock;
    shared_memory_size=(unsigned int)((float)shared_memory_size * THREAD_RATIO);
    dim3 dimGrid(grid_size,1,1);
    dim3 dimBlock(block_size,1,1);
    localFilter_kernel<<<dimGrid,dimBlock,shared_memory_size>>>(filtered_data_d,filter_indices_d,clusters_d,lb_d,ub_d,labels_d,filtered_num,maxdelta,k,dim,shared_memory_size/sizeof(float));
}

void kmeans_elkan(float *data,float *clusters,unsigned int *labels, unsigned int n_samples,unsigned int dim, unsigned int k, unsigned int max_iter, float epsilon, int i_c,int o_c, int* configurations, float * times){
    int maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount;

    //cudaDeviceGetAttribute(&maxthreadsperblock,cudaDevAttrMaxThreadsPerBlock,0);
    cudaDeviceGetAttribute(&maxsharedmemoryperblock,cudaDevAttrMaxSharedMemoryPerBlock,0);
    cudaDeviceGetAttribute(&maxthreadspermultiprocessor,cudaDevAttrMaxThreadsPerMultiProcessor,0);
    cudaDeviceGetAttribute(&maxsharedmemorypermultiprocessor,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);
    cudaDeviceGetAttribute(&multiprocessorcount,cudaDevAttrMultiProcessorCount,0);

    Timer totaltimer,timer;
    startTime(&totaltimer);
    //part 1 alloc and memsets

    float *data_d, *clusters_d;
    unsigned int * labels_d,*filtered_indices_d;
    float *clusterdists_h;
   
    float *distances_d,*newclusters_d,*centerdists_d,*clusterdists_d;
    float *delta_d,*lb_d,*ub_d,*filtered_data_d;
    unsigned int *counts_d,*filtered_num_d;
    unsigned int interia_size=(n_samples-1)/(2*BLOCK_SIZE)+1;
    
    
    //interia_h=(float*) malloc( sizeof(float)*interia_size );
    clusterdists_h=(float*) malloc( sizeof(float)*k );
    cudaMalloc((void **)&data_d,sizeof(float)*n_samples*dim);
    cudaMalloc((void **)&clusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&labels_d,sizeof(unsigned int)*n_samples);
    //cudaMalloc((void **)&distances_d,sizeof(float)*n_samples*k); 
    cudaMalloc((void **)&newclusters_d,sizeof(float)*k*dim);
    cudaMalloc((void **)&counts_d,sizeof(unsigned int)*k);
    //cudaMalloc((void **)&interia_d,sizeof(float)*interia_size);
    cudaMalloc((void **)&centerdists_d,sizeof(float)*n_samples);
    cudaMalloc((void **)&clusterdists_d,sizeof(float)*k);
    //cudaMalloc((void **)&delta_d,sizeof(float)*k);
    cudaMalloc((void **)&lb_d,sizeof(float)*n_samples);
    cudaMalloc((void **)&ub_d,sizeof(float)*n_samples);
    cudaMalloc((void **)&filtered_data_d,sizeof(float)*n_samples*dim);
    cudaMalloc((void **)&filtered_indices_d,sizeof(unsigned int)*n_samples);
    cudaMalloc((void **)&filtered_num_d,sizeof(unsigned int));
    cudaMemcpy(data_d,data,sizeof(float)*n_samples*dim,cudaMemcpyHostToDevice);
    cudaMemcpy(clusters_d,clusters,sizeof(float)*k*dim,cudaMemcpyHostToDevice);
    
    
   
    //part 1 end


    //part 2: initialize clusters and copy
    unsigned int initial_clusters[k];
    for(int i=0;i<k;i++){
        while(1){
            unsigned int r=rand()%n_samples; 
            bool duplicated=false;
            for(int j=0;j<i;j++){
                if (r==initial_clusters[j]){
                    duplicated=true;
                    break;
                }
            }
            if (!duplicated){
                initial_clusters[i]=r;
                break;
            }
       }
    }
    for(int i=0;i<k;i++){
        unsigned int idx=initial_clusters[i];
        for(int j=0;j<dim;j++){
            if(i_c==0)
                clusters[i*dim+j]=data[idx*dim+j];
            else
                clusters[i*dim+j]=data[j*n_samples+idx];

        }
    }
        
    //part 2 end
  
    
    //part 3 gogogo
    
    //float last_interia=0;
    for(int round=0;round<max_iter;round++){
        
        cudaMemset(newclusters_d,0,sizeof(float)*k*dim);
	cudaMemset(counts_d,0,sizeof(unsigned int)*k);
        
        //cudaMemset(centerdists_d,0,sizeof(float)*n_samples);
	if (round==0){

     
            calcDistancesLabels_elkan(data_d,clusters_d,labels_d,lb_d,ub_d,n_samples,dim,k,i_c,maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount);
       

            //calcLabels_elkan(distances_d,labels_d,lb_d,ub_d,n_samples,dim,k,o_c);
       
            
            
            
        }

        else{
            
               accumulateNewClusters_1d(data_d,labels_d,newclusters_d,counts_d,n_samples,dim,k,i_c,maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount); 
           
            
            averageNewClusters_1d(newclusters_d,clusters_d,counts_d,dim,k,maxthreadspermultiprocessor,multiprocessorcount);
            calcClusterDist(clusters_d, newclusters_d,clusterdists_d,dim,k);
            cudaMemcpy(clusterdists_h,clusterdists_d,sizeof(float)*k,cudaMemcpyDeviceToHost);
            cudaMemcpy(clusters_d,newclusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToDevice);
            
            bool terminate=true;
            float maxdelta=0;
            for(int i=0;i<k;i++){
                float t=clusterdists_h[i];
                if(t>epsilon){
                    terminate=false;
    
                }
                if(t>maxdelta)
                    maxdelta=t;
      
            }

            if(terminate){
            
                //printf("stopped on round %d\n",round);
            
                break;
            
            }
            cudaMemset(filtered_num_d,0,sizeof(unsigned int));
            globalFilter(data_d,lb_d,ub_d,labels_d,clusterdists_d,clusters_d,filtered_data_d,filtered_indices_d,filtered_num_d,maxdelta,n_samples,dim,i_c,maxthreadspermultiprocessor,multiprocessorcount);
            unsigned int filtered_num_h;
            cudaMemcpy(&filtered_num_h,filtered_num_d,sizeof(unsigned int),cudaMemcpyDeviceToHost);
            localFilter(filtered_data_d,filtered_indices_d,clusters_d,lb_d,ub_d,labels_d,filtered_num_h,maxdelta,k,dim,maxsharedmemoryperblock,maxthreadspermultiprocessor,maxsharedmemorypermultiprocessor,multiprocessorcount);




            

            

    
        }
        //last_interia=interia;
        //cudaMemcpy(clusters_d,clusters,sizeof(float)*n_samples*k,cudaMemcpyHostToDevice);
        
        
        
     

    }
    cudaMemcpy(clusters,clusters_d,sizeof(float)*k*dim,cudaMemcpyDeviceToHost);
    cudaMemcpy(labels,labels_d,sizeof(float)*n_samples,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&totaltimer); times[0]=elapsedTime(totaltimer);
    //free(interia_h);
    free(clusterdists_h);
    cudaFree(data_d);
    cudaFree(clusters_d);
    cudaFree(labels_d);
    //cudaFree(distances_d);
    cudaFree(newclusters_d);
    cudaFree(counts_d);
    //cudaFree(interia_d);
    cudaFree(centerdists_d);
    //cudaFree(delta_d);
    cudaFree(lb_d);
    cudaFree(ub_d);
    cudaFree(filtered_data_d);
    cudaFree(filtered_indices_d);
    cudaFree(filtered_num_d);
}

/*
        calcInteria_1d(data_d,clusters_d,labels_d,centerdists_d,interia_d,n_samples,dim,i_c);
    
        cudaMemcpy(interia_h,interia_d,sizeof(float)*interia_size,cudaMemcpyDeviceToHost);

        
        float interia=0;
        for(int i=0;i<interia_size;i++)
            interia+=interia_h[i];
        
        //printf("%f %f\n",last_interia,interia);
        */

