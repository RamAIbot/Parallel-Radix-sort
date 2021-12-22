#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#ifdef OPENMP_HARNESS
#include <omp.h>
#endif

#ifdef MPI_HARNESS
#include <mpi.h>
#endif

#ifdef HYBRID_HARNESS
#include <omp.h>
#include <mpi.h>
#endif

#include "sort.h"
#include "graph.h"

struct Graph *countSortEdgesBySource (struct Graph *graph)
{

    int i;
    int key;
    int pos;
    struct Edge *sorted_edges_array = newEdgeArray(graph->num_edges);

    // auxiliary arrays, allocated at the start up of the program
    int *vertex_count = (int *)malloc(graph->num_vertices * sizeof(int)); // needed for Counting Sort

    for(i = 0; i < graph->num_vertices; ++i)
    {
        vertex_count[i] = 0;
    }

    // count occurrence of key: id of a source vertex
    for(i = 0; i < graph->num_edges; ++i)
    {
        key = graph->sorted_edges_array[i].src;
        vertex_count[key]++;
    }

    // transform to cumulative sum
    for(i = 1; i < graph->num_vertices; ++i)
    {
        vertex_count[i] += vertex_count[i - 1];
    }

    // fill-in the sorted array of edges
    for(i = graph->num_edges - 1; i >= 0; --i)
    {
        key = graph->sorted_edges_array[i].src;
        pos = vertex_count[key] - 1;
        sorted_edges_array[pos] = graph->sorted_edges_array[i];
        vertex_count[key]--;
    }



    free(vertex_count);
    free(graph->sorted_edges_array);

    graph->sorted_edges_array = sorted_edges_array;

    return graph;

}


//#ifdef OPENMP_HARNESS
struct Graph *radixSortEdgesBySourceOpenMP (struct Graph *graph)
{

    bool verbose = false;
    bool verbose_print = false;
    int radix = 8;
    int num_of_elements_in_radix = (unsigned int)(pow(2,radix));
    int num_of_traversals = (unsigned int) (32/radix);
    printf("*** START Radix Sort Edges By Source OpenMP *** \n");
    int number_of_edges = graph->num_edges;
    int num_threads = omp_get_max_threads();
    int i,j,tid;
    int from,to;
    int temp;
    //int max_out[] = {0,0,0,0};
    int t=0,k=0,temp_num=0,sum=0;
    struct Edge* out_array;
    struct Edge* out_array1;

    //out_array = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    out_array = newEdgeArray(number_of_edges);
    //out_array1 = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    out_array1 = newEdgeArray(number_of_edges);

    if(verbose)
    {
        printf("\nGRAPH\n");

         for(int y=0;y<number_of_edges;y++)
         {
             printf("%d -> %d ",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
         }
    }

    //#pragma omp parallel for num_threads(2)
    for(int y=0;y<number_of_edges;y++)
    {
        out_array[y] = graph->sorted_edges_array[y];
        out_array1[y] = graph->sorted_edges_array[y];
    }
    // memcpy(&out_array, &graph->sorted_edges_array, sizeof(graph->sorted_edges_array));
    // memcpy(&out_array1, &graph->sorted_edges_array, sizeof(graph->sorted_edges_array));
    
    if(verbose)
    {
        printf("\n");
        for(int y=0;y<number_of_edges;y++)
        {
            printf("%d -> %d ",out_array[y].src,out_array[y].dest);
        }
    }

    for(i=1;i<=num_of_traversals;i++)
    {
        int r = num_threads;
        int c = num_of_elements_in_radix;
        int** count_array = (int**)malloc(r * sizeof(int*));
        for (int y = 0; y < r; y++)
            count_array[y] = (int*)malloc(c * sizeof(int));

        for(int x=0;x<r;x++)
        {
            for(int y=0;y<c;y++)
                count_array[x][y] = 0;
        }

        sum = 0;
        //memcpy(&out_array1, &graph->sorted_edges_array, sizeof(graph->sorted_edges_array));

        #pragma omp parallel default(shared) private(j,tid,from,to,t,k,temp_num) shared(i,num_threads,count_array,graph,sum) num_threads(num_threads)
        {
            tid = omp_get_thread_num();

            from = (number_of_edges/num_threads)*tid;
            if(verbose)
                printf("FROM: %d\n",from);
            
            to = ((number_of_edges/num_threads)*(tid+1)) - 1;
            if(tid == num_threads-1)
                to = number_of_edges - 1;
            
            if(verbose)
                printf("TO: %d\n",to);

            for(j=from;j<=to;j++)
            {
                t = out_array[j].src;
                temp_num = (unsigned int)(((t >> ((i-1)*radix))) & (num_of_elements_in_radix - 1));
                if(verbose)
                    printf("\n TEMP NUM: %d\n",temp_num);

                count_array[tid][temp_num] += 1;
            
            }
            #pragma omp barrier
            
            if(verbose)
            {
                #pragma omp single
                //printing count array
                for(int x=0;x<num_threads;x++)
                {
                    for(int y=0;y<num_of_elements_in_radix;y++)
                        printf("%d ",count_array[x][y]);

                    printf("\n");
                }
                printf("\n");
            }

            //Prefix sum
            #pragma omp single
            {
                for(int j=0;j<num_of_elements_in_radix;j++)
                {
                    for(int k=0;k<num_threads;k++)
                    {
                        if(k==0 && j==0)
                        {
                            //count_array[k][j] = 0;
                            //continue;
                        }
                        int var = count_array[k][j];
                        sum += var;
                        count_array[k][j] = sum;
                        
                    }
                }
            }


            if(verbose)
            {
                #pragma omp single
                //printing count array
                for(int x=0;x<num_threads;x++)
                {
                    for(int y=0;y<num_of_elements_in_radix;y++)
                        printf("%d ",count_array[x][y]);

                    printf("\n");
                }
                printf("\n");
            }
            
            for(int j=to;j>=from;j--)
            {
                t = out_array[j].src;
                if(verbose)
                    printf("\nt_%d\n",t);
                temp_num = (unsigned int)(((t >> ((i-1)*radix))) & (num_of_elements_in_radix - 1));
                    
                
                if(verbose)
                    printf("\n TEMP NUM: %d\n",temp_num);
                
                #pragma omp critical
                {
                
                //if(temp_num!=0)
                //{
                    if(verbose)
                    {
                        printf("\n Thread number: %d\n",tid);
                        printf("\n POS: %d\n",(count_array[tid][temp_num]) - 1);
                        printf("\n SRC: %d\n",graph->sorted_edges_array[j].src);

                    }
                    out_array1[(count_array[tid][temp_num]) - 1] = out_array[j];
                    count_array[tid][temp_num] -= 1;
                //}

                }
                
                
            }
            #pragma omp barrier
                
        }

    
    //memcpy(&out_array, &out_array1, sizeof(out_array));

    for(int y=0;y<number_of_edges;y++)
    {
        out_array[y] = out_array1[y];
    }

    if(verbose)
    {
        printf("\n OUTPUT LOCAL\n");
        for(int y=0;y<number_of_edges;y++)
        {
            printf("%d ->  %d \n",out_array[y].src,out_array[y].dest);
        }
    }
        
    }


    #pragma omp parallel for num_threads(num_threads)
    for(int y=0;y<number_of_edges;y++)
    {
        graph->sorted_edges_array[y] = out_array[y];
    }

    if(verbose_print)
    {
        printf("\n FINAL OUTPUT\n");
        for(int y=0;y<number_of_edges;y++)
        {
            printf("%d ->  %d \n",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
        }
    }
    

    return graph;
}
//#endif

#ifdef MPI_HARNESS
struct Graph *radixSortEdgesBySourceMPI (struct Graph *graph,int argc, char **argv)
{

    printf("*** START Radix Sort Edges By Source MPI*** \n");
    int number_of_edges = graph->num_edges;

    struct Edge* out_array = NULL;
    struct Edge* out_array1 = NULL;
    
    struct Edge* local_out_array_dummy = NULL;

    //out_array = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    //out_array1 = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    out_array = newEdgeArray(number_of_edges);
    
    out_array1 = newEdgeArray(number_of_edges);
    

    bool verbose = false;
    bool verbose_print = true;
    int radix = 8;
    int num_of_elements_in_radix = (unsigned int)(pow(2,radix));
    int num_of_traversals = (unsigned int) (32/radix);

    if(verbose)
    {
        printf("\nGRAPH\n");

         for(int y=0;y<number_of_edges;y++)
         {
             printf("%d -> %d ",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
         }
    }

    //#pragma omp parallel for num_threads(2)
    for(int y=0;y<number_of_edges;y++)
    {
        out_array[y] = graph->sorted_edges_array[y];
        out_array1[y] = graph->sorted_edges_array[y];
    }
    //memcpy(&out_array, &graph->sorted_edges_array, sizeof(graph->sorted_edges_array));
    
    if(verbose)
    {
        printf("\n");
        for(int y=0;y<number_of_edges;y++)
        {
            printf("TEST_%d -> %d ",out_array[y].src,out_array[y].dest);
        }
    }

    int u;
    
    MPI_Init(NULL,NULL);

    //   /* Create 2 more processes - this example must be called spawn_example.exe for this to work. */
    // int errcodes[2];
    // MPI_Comm parentcomm, intercomm;
    
    // MPI_Comm_get_parent( &parentcomm );

    // if (parentcomm == MPI_COMM_NULL)
    // {
    //     /* Create 2 more processes - this example must be called spawn_example.exe for this to work. */
    //     MPI_Comm_spawn( "spawn_example", MPI_ARGV_NULL, 2, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes );
    //     printf("I'm the parent.\n");
    // }

    MPI_Datatype types[2] = {MPI_INT,MPI_INT};
    MPI_Datatype mpi_graph;
    const int nitems = 2;
    int blocklengths[2] = {1,1};
    MPI_Aint offsets[2],lb,extent;

    
    struct Edge edge;
    offsets[0] = (char *) & (edge.src) - (char *)&edge;
    //MPI_Type_get_extent(MPI_INT,&lb,&extent);
    offsets[1] = (char *) & (edge.dest) - (char *)&edge;

     
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_graph);
    MPI_Type_commit(&mpi_graph);

    

    for(u=1;u<=num_of_traversals;u++)
    {
        int np,pid;
        //printf("U_%d\n",u);
        MPI_Comm_rank(MPI_COMM_WORLD,&pid);
        MPI_Comm_size(MPI_COMM_WORLD,&np);
        MPI_Status status;

        int index,i,temp_num=0;
        int elements_per_process;

        if(verbose)
            printf("\n NUMBER OF PROCESS:%d\n",np);

        int r = np;
        int c = num_of_elements_in_radix;
        int* count_array = (int*)malloc(c * sizeof(int*));

        

        if(pid == 0)
        {
            //Master process
            if(verbose)
                printf("\n MASTER PROCESS\n");

            for(int p=0;p<c;p++)
                count_array[p] = 0;

            int count = (int)(number_of_edges/np);
            //elements_per_process = (int)(number_of_edges/np);
            int remainder = (int)(number_of_edges%np);
            //elements_per_process = (int)((number_of_edges/np) + (number_of_edges % np)) - 1;
            

            
            if(np > 1)
            {
                for(i=1;i<np-1;i++)
                {
                    int start = i*count + (i<remainder?i:remainder);
                    int stop = (i + 1) * count + ((i+1)<remainder?(i+1):remainder);
                    //index = i*elements_per_process;

                    index = start;
                    elements_per_process = stop - start;
                    MPI_Send(&elements_per_process,1,MPI_INT,i,0,MPI_COMM_WORLD);
                    for(int x=0;x<elements_per_process;x++)
                        MPI_Send(&out_array[index+x],1,mpi_graph,i,0,MPI_COMM_WORLD);
                }

                //index = i*elements_per_process;
                int start = i*count + (i<remainder?i:remainder);
                int stop = (i + 1) * count + ((i+1)<remainder?(i+1):remainder);
                index = i*elements_per_process;

                index = start;
                
                int elements_left = number_of_edges - index;
                elements_left = stop - start;

                if(1)
                printf("\nELEMENTSLEFT_%d\n",elements_left);

                MPI_Send(&elements_left,1,MPI_INT,i,0,MPI_COMM_WORLD);
                for(int x=0;x<elements_left;x++)
                    MPI_Send(&out_array[index+x],1,mpi_graph,i,0,MPI_COMM_WORLD);

            }

            int t=0;
            int start = pid*count + (pid<remainder?pid:remainder);
            int stop = (pid + 1) * count + ((pid+1)<remainder?(pid+1):remainder);
            index = pid*elements_per_process;

            index = start;
            elements_per_process = stop - start;
            if(1)
                printf("\nELEMENTS_%d\n",elements_per_process);
            for(i=0;i<elements_per_process;i++)
            {
                t = out_array[i].src;
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                count_array[temp_num] += 1;
            }

            if(verbose)
            {
                for(i=0;i<num_of_elements_in_radix;i++)
                {
                    printf("M_%d ",count_array[i]);
                }
                printf("\n");
            }

            //Prefix Sum

            for(i=1;i<num_of_elements_in_radix;i++)
            {
                
                count_array[i] += count_array[i-1];
            }

            //TESTING OCT 4
            if(count_array[0] == elements_per_process)
            {
                printf("\n EPP\n");
                continue;
            }
            //ENDS
            

            //Loop 4

           //struct Edge* local_out_array_dummy = (struct Edge *)malloc ( elements_per_process * sizeof(struct Edge) ) ;
            
           struct Edge* local_out_array_dummy = newEdgeArray(elements_per_process);

            for(i=elements_per_process-1;i>=0;i--)
            {
                t = out_array[i].src;
                // printf("tttt_%d ",t);
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                    

                if(verbose)
                {   
                    printf("TEMP_NUM_%d ",temp_num);
                    int jjj = (count_array[temp_num]) - 1;
                    printf("COUNT_DEF_%d ",jjj);
                }
                
                local_out_array_dummy[(count_array[temp_num]) - 1] = out_array[i];
                count_array[temp_num] -= 1;

            }

            if(verbose)
            {
                 for(i=0;i<elements_per_process;i++)
                {
                    printf("COUT_%d ",local_out_array_dummy[i]);
                }
                printf("\n");
            }

            for(int i=0;i<elements_per_process;i++)
            {
                out_array1[i] = local_out_array_dummy[i];
            }
            int master_length = elements_per_process;
            
            //MPI_Barrier(MPI_COMM_WORLD);

            for(i=1;i<np;i++)
            {
                //struct Edge* final_arr = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
                struct Edge* final_arr = newEdgeArray(number_of_edges);
                //struct Edge* a2 = (struct Edge *)malloc ( elements_per_process * sizeof(struct Edge) ) ;
                
                MPI_Recv(&elements_per_process,1,MPI_INT,i,1,MPI_COMM_WORLD,&status);
                struct Edge* a2 = newEdgeArray(elements_per_process);
                int o=0;
                for(int x=0;x<elements_per_process;x++)
                {
                    struct Edge ax;
                    MPI_Recv(&ax,1,mpi_graph,i,1,MPI_COMM_WORLD,&status);
                    //printf("MS_%d\n",status);
                    a2[o++] = ax;
                    //free(&a1);
                }
                
                //printf("\n MASTER RECEV\n");
                if(verbose)
                {
                    for(int x=0;x<elements_per_process;x++)
                    {
                        
                        printf("Masterrecv_%d_pid_%d\n",a2[x],pid);
                        
                    }
                }

                int new_length = master_length + elements_per_process;
                int x=0,y=0,z=0;

                while(x<master_length && y<elements_per_process)
                {
                    //printf("\n HERE\n");
                    if(out_array1[x].src < a2[y].src)
                    {
                        final_arr[z++] = out_array1[x++];
                    }

                    //added oct 4
                    // else if(out_array1[x].src == a2[y].src)
                    // {
                    //     final_arr[z++] = out_array1[x++];
                    //     final_arr[z++] = a2[y++];
                    // }
                    //
                    else // if(child_arr[x] > arr[y])
                    {
                        final_arr[z++] = a2[y++];
                    }


                }

                while(x<master_length)
                {
                    final_arr[z++] = out_array1[x++];
                }

                while(y<elements_per_process)
                {
                    final_arr[z++] = a2[y++];
                }

                //printf("\n HERE1\n");
                master_length = new_length;

                for(int i=0;i<master_length;i++)
                {
                    out_array1[i] = final_arr[i];
                }

                

                free(final_arr);
                free(a2);

            }
            
            for(int i=0;i<number_of_edges;i++)
            {
                out_array[i] = out_array1[i];
            }

            if(1)
            {
                for(int i=0;i<number_of_edges;i++)
                {
                    printf("FOUT_%d -> %d",out_array[i].src,out_array[i].dest);
                }  
            }

            
            

       }

        else

        {
            //Slave process
            int n_elements_received;
            MPI_Recv(&n_elements_received,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
            
            //struct Edge *a = (struct Edge *)malloc ( n_elements_received * sizeof(struct Edge) ) ;

            struct Edge *a = newEdgeArray(n_elements_received);
            int o=0;
            for(int x=0;x<n_elements_received;x++)
            {
                struct Edge a1;
                MPI_Recv(&a1,1,mpi_graph,0,0,MPI_COMM_WORLD,&status);
                //printf("S_%d\n",status);
                a[o++] = a1;
                
            }
                
            
            if(verbose)
                printf("\n Received\n");

            if(verbose)
            {
                for(int x=0;x<n_elements_received;x++)
                {
                    //printf("\nX_%d",x);
                    printf("MAP_%d -> %d \n",a[x].src,a[x].dest);
                }
                printf("\n");
            }
            //struct Edge* out_array_local = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
            struct Edge* out_array_local = newEdgeArray(n_elements_received);

            int local_count_array[num_of_elements_in_radix];

            for(i=0;i<num_of_elements_in_radix;i++)
            {
                local_count_array[i] = 0;
            }

            int t = 0;
            for(i=0;i<n_elements_received;i++)
            {
                t = a[i].src;
                //printf(" A_%d",a[i]);
                
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                    
                
                //printf("\n TEMP NUM: %d\n",temp_num);
                local_count_array[temp_num] += 1;
            }

            if(verbose)
            {
                printf("\n slave PROCESS: %d\n",pid);
                for(i=0;i<num_of_elements_in_radix;i++)
                {
                    printf("P_%d ",local_count_array[i]);
                }
            }

            //Prefix sum

            for(i=1;i<num_of_elements_in_radix;i++)
            {
                
                local_count_array[i] += local_count_array[i-1];
            }

            //TESTING OCT 4
            if(local_count_array[0] == n_elements_received)
            {
                printf("\n EPP_Slave\n");
                continue;
            }
            //ENDS

            if(verbose)
            {
                printf("\n slave PROCESS: %d\n",pid);
                for(i=0;i<num_of_elements_in_radix;i++)
                {
                    printf("PP_%d ",local_count_array[i]);
                }
                printf("\n");
            }

            for(i=n_elements_received-1;i>=0;i--)
            {
                t = a[i].src;
                
                if(verbose)
                    printf("\nt_%d",t);

                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                

                out_array_local[local_count_array[temp_num] - 1] =  a[i];
                local_count_array[temp_num] -= 1;
            }

            if(verbose)
            {
                 for(i=0;i<n_elements_received;i++)
                {
                    printf("OUT_%d -> %d",out_array_local[i].src,out_array_local[i].dest);
                }
                printf("\n");
            }
            
            MPI_Send(&n_elements_received,1,MPI_INT,0,1,MPI_COMM_WORLD);

            for(int x=0;x<n_elements_received;x++)
                MPI_Send(&out_array_local[x],1,mpi_graph,0,1,MPI_COMM_WORLD);

            
        }MPI_Barrier(MPI_COMM_WORLD);

        //free(local_out_array_dummy);
    }
    MPI_Type_free(&mpi_graph);
    //MPI_Finalize();

    for(int y=0;y<number_of_edges;y++)
    {
        graph->sorted_edges_array[y] = out_array[y];
    }

    if(verbose_print)
    {
        printf("\n FINAL OUTPUT\n");
        for(int y=0;y<number_of_edges;y++)
        {
            printf("%d ->  %d \n",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
        }
    }
    
    return graph;
}
#endif

#ifdef HYBRID_HARNESS
struct Graph *radixSortEdgesBySourceHybrid (struct Graph *graph)
{

    printf("*** START Radix Sort Edges By Source Hybrid*** \n");
    return graph;
}
#endif