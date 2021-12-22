#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <memory.h>

#include "graph.h"
#include "bfs.h"
#include "sort.h"
#include "edgelist.h"
#include "vertex.h"
#include "timer.h"

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

int numThreads;

void printMessageWithtime(const char *msg, double time)
{

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", msg);
    printf(" -----------------------------------------------------\n");
    printf("| %-51f | \n", time);
    printf(" -----------------------------------------------------\n");

}

static void usage(void)
{
    printf("\nUsage: ./main -f <graph file> -r [root] -n [num threads]\n");
    printf("\t-f <graph file.txt>\n");
    printf("\t-h [help]\n");
    printf("\t-r [root/source]: BFS \n");
    printf("\t-n [num threads] default:max number of threads the system has\n");
    // _exit(-1);
}


int main(int argc, char **argv)
{


    #ifdef OPENMP_HARNESS
    bool testing = true;
    struct Graph *graph1;
    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;


    numThreads = omp_get_max_threads();
    char *fnameb = NULL;

    int c;
    opterr = 0;

    while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
    {
        switch (c)
        {
        case 'h':
            usage();
            break;
        case 'f':
            fvalue = optarg;
            fnameb = fvalue;
            break;
        case 'r':
            rvalue = optarg;
            root = atoi(rvalue);
            break;
            break;
        case 'n':
            nvalue = optarg;
            numThreads = atoi(nvalue);
            break;
        case '?':
            if (optopt == 'f')
                fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
            else if (optopt == 'r')
                fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
            else if (optopt == 'n')
                fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            usage();
            return 1;
        default:
            abort ();
        }
    }


    //Set number of threads for the program
    omp_set_nested(1);
    omp_set_num_threads(numThreads);

#ifdef OPENMP_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "OPENMP Implementation");
#endif

#ifdef MPI_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "MPI Implementation");
#endif

#ifdef HYBRID_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Hybrid (OMP+MPI) Implementation");
#endif

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "File Name");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", fnameb);
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Number of Threads");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", numThreads);
    printf(" -----------------------------------------------------\n");


    struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "New graph calculating size");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph = newGraph(fnameb);
    Stop(timer);
    printMessageWithtime("New Graph Created", Seconds(timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Populate Graph with edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    // populate the edge array from file
    loadEdgeArray(fnameb, graph);
    Stop(timer);
    printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));



    // you need to parallelize this function
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "COUNT Sort Graph");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    graph1 = countSortEdgesBySource(graph); // you need to parallelize this function
    
    Stop(timer);
    printMessageWithtime("Time Sorting serial count sort(Seconds)", Seconds(timer));

    Start(timer);
    graph = radixSortEdgesBySourceOpenMP(graph); // you need to parallelize this function
    Stop(timer);
    printMessageWithtime("Time Sorting for Parallel Radix Sort (Seconds)", Seconds(timer));

    //Local Testing
    if(testing)
    {

    bool flag = true;
    int ind = -1;
    for(int y=0;y<graph->num_edges;y++)
    {
        if((graph->sorted_edges_array[y].src!=graph1->sorted_edges_array[y].src)||(graph->sorted_edges_array[y].dest != graph1->sorted_edges_array[y].dest))
        {
            ind = y;
            flag = false;
            break;
        }
    }

    if(flag == true)
        printf("\n TEST SUCCESS\n");
    else
        printf("\n TEST FAILED at index %d\n",ind);

    }
    //local testing ends

    // For testing purpose.

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Map vertices to Edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    mapVertices(graph);
    Stop(timer);
    printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

    printf(" *****************************************************\n");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
    printf(" -----------------------------------------------------\n");

    printf("| %-51s | \n", "PUSH");

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "ROOT/SOURCE");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", root);
    printf(" -----------------------------------------------------\n");
    Start(timer);

    breadthFirstSearchGraphPush(root, graph);

    // for (int p = 0; p < 10; ++p)
    // {
    //   breadthFirstSearchGraphPush(p, graph);
    // }

    Stop(timer);
    printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

    Start(timer);
    freeGraph(graph);
    Stop(timer);
    printMessageWithtime("Free Graph (Seconds)", Seconds(timer));

    
    #endif

    #ifdef MPI_HARNESS

    MPI_Init(&argc,&argv);

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

    int np,pid;
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Status status;

    bool verbose = false;
    bool verbose_print = false;
    int testing = true;
    int radix = 8;
    int num_of_elements_in_radix = (unsigned int)(pow(2,radix));
    int num_of_traversals = (unsigned int) (32/radix);

    if(pid == 0)
    {
    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;


    numThreads = omp_get_max_threads();
    char *fnameb = NULL;

    int c;
    opterr = 0;

    while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
    {
        switch (c)
        {
        case 'h':
            usage();
            break;
        case 'f':
            fvalue = optarg;
            fnameb = fvalue;
            break;
        case 'r':
            rvalue = optarg;
            root = atoi(rvalue);
            break;
            break;
        case 'n':
            nvalue = optarg;
            numThreads = atoi(nvalue);
            break;
        case '?':
            if (optopt == 'f')
                fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
            else if (optopt == 'r')
                fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
            else if (optopt == 'n')
                fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            usage();
            return 1;
        default:
            abort ();
        }
    }


    //Set number of threads for the program
    omp_set_nested(1);
    omp_set_num_threads(numThreads);

#ifdef OPENMP_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "OPENMP Implementation");
#endif

#ifdef MPI_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "MPI Implementation");
#endif

#ifdef HYBRID_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Hybrid (OMP+MPI) Implementation");
#endif

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "File Name");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", fnameb);
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Number of Threads");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", numThreads);
    printf(" -----------------------------------------------------\n");


    struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "New graph calculating size");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph = newGraph(fnameb);
    Stop(timer);
    printMessageWithtime("New Graph Created", Seconds(timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Populate Graph with edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    // populate the edge array from file
    loadEdgeArray(fnameb, graph);
    Stop(timer);
    printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));



    // you need to parallelize this function
    
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "COUNT Sort Graph");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph1;
    graph1 = countSortEdgesBySource(graph); // you need to parallelize this function
    Stop(timer);
    printMessageWithtime("Time taken for serial count sort (Seconds)", Seconds(timer));

    Start(timer);
    

    #ifdef MPI_HARNESS
    //graph = radixSortEdgesBySourceMPI(graph,argc,&argv);

    printf("*** START Radix Sort Edges By Source MPI*** \n");
    int number_of_edges = graph->num_edges;

    struct Edge* out_array = NULL;
    struct Edge* out_array1 = NULL;
    
    struct Edge* local_out_array_dummy = NULL;

    //out_array = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    //out_array1 = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    out_array = newEdgeArray(number_of_edges);
    
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

    for(u=1;u<=num_of_traversals;u++)
    {

    
     int index,i,temp_num=0;
    int elements_per_process;

    if(verbose)
        printf("\n NUMBER OF PROCESS:%d\n",np);

    int r = np;
    c = num_of_elements_in_radix;
    int* count_array = (int*)malloc(c * sizeof(int*));

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
                    MPI_Send(&elements_per_process,1,MPI_INT,i,i,MPI_COMM_WORLD);
                    
                    for(int x=0;x<elements_per_process;x++)
                        MPI_Send(&out_array[index+x],1,mpi_graph,i,i,MPI_COMM_WORLD);
                }

                //index = i*elements_per_process;
                int start = i*count + (i<remainder?i:remainder);
                int stop = (i + 1) * count + ((i+1)<remainder?(i+1):remainder);
                index = i*elements_per_process;

                index = start;
                
                int elements_left = number_of_edges - index;
                elements_left = stop - start;

                if(verbose)
                printf("\nELEMENTSLEFT_%d\n",elements_left);

                MPI_Send(&elements_left,1,MPI_INT,i,i,MPI_COMM_WORLD);
                for(int x=0;x<elements_left;x++)
                    MPI_Send(&out_array[index+x],1,mpi_graph,i,i,MPI_COMM_WORLD);

            }

            int t=0;
            int start = pid*count + (pid<remainder?pid:remainder);
            int stop = (pid + 1) * count + ((pid+1)<remainder?(pid+1):remainder);
            index = pid*elements_per_process;

            index = start;
            elements_per_process = stop - start;
            if(verbose)
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

            struct Edge* local_out_array_dummy = newEdgeArray(elements_per_process);

            for(i=elements_per_process-1;i>=0;i--)
            {
                t = out_array[i].src;
                // printf("tttt_%d ",t);
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                    

                if(verbose)
                {   
                    printf("TEMP_NUM_MAS_%d ",temp_num);
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

            if(verbose)
            {
                for(int i=0;i<number_of_edges;i++)
                {
                    printf("FOUT_%d -> %d",out_array[i].src,out_array[i].dest);
                }  
            }


            for(int y=0;y<graph->num_edges;y++)
            {
                graph->sorted_edges_array[y] = out_array[y];
            }

            if(verbose_print)
            {
                printf("\n FINAL OUTPUT\n");
                for(int y=0;y<graph->num_edges;y++)
                {
                    printf("%d ->  %d \n",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
                }
            }
            
    }
    
    #endif
    Stop(timer);
    printMessageWithtime("Time Sorting for Parallel Radix Sort (Seconds)", Seconds(timer));

    //Local Testing
    if(testing)
    {

    bool flag = true;
    int ind = -1;
    for(int y=0;y<graph->num_edges;y++)
    {
        if((graph->sorted_edges_array[y].src!=graph1->sorted_edges_array[y].src)||(graph->sorted_edges_array[y].dest != graph1->sorted_edges_array[y].dest))
        {
            ind = y;
            flag = false;
            break;
        }
    }

    if(flag == true)
        printf("\n TEST SUCCESS\n");
    else
        printf("\n TEST FAILED at index %d\n",ind);

    }
    //local testing ends

    // For testing purpose.

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Map vertices to Edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    mapVertices(graph);
    Stop(timer);
    printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

    printf(" *****************************************************\n");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
    printf(" -----------------------------------------------------\n");

    printf("| %-51s | \n", "PUSH");

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "ROOT/SOURCE");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", root);
    printf(" -----------------------------------------------------\n");
    Start(timer);

    breadthFirstSearchGraphPush(root, graph);

    // for (int p = 0; p < 10; ++p)
    // {
    //   breadthFirstSearchGraphPush(p, graph);
    // }

    Stop(timer);
    printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

    Start(timer);
    freeGraph(graph);
    Stop(timer);
    printMessageWithtime("Free Graph (Seconds)", Seconds(timer));
    
    
    }


    else
    {

    
    //slave processes
    int u;
    for(u=1;u<=num_of_traversals;u++)
    {

    
    int n_elements_received;
            MPI_Recv(&n_elements_received,1,MPI_INT,0,pid,MPI_COMM_WORLD,&status);
            
            //struct Edge *a = (struct Edge *)malloc ( n_elements_received * sizeof(struct Edge) ) ;

            struct Edge *a = newEdgeArray(n_elements_received);
            int o=0;
            for(int x=0;x<n_elements_received;x++)
            {
                struct Edge a1;
                MPI_Recv(&a1,1,mpi_graph,0,pid,MPI_COMM_WORLD,&status);
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
            int i,temp_num;
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
                //printf("\n L_%d ",local_count_array[temp_num]);
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


          
        }
    }
    MPI_Type_free(&mpi_graph);
    MPI_Finalize();

    

    

    #endif
    




    #ifdef HYBRID_HARNESS

    MPI_Init(&argc,&argv);

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

    int np,pid;
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Status status;

    bool verbose = false;
    bool verbose_print = false;
    int testing = true;
    int radix = 8;
    int num_of_elements_in_radix = (unsigned int)(pow(2,radix));
    int num_of_traversals = (unsigned int) (32/radix);

    if(pid == 0)
    {
    char *fvalue = NULL;
    char *rvalue = NULL;
    char *nvalue = NULL;

    int root = 0;


    numThreads = omp_get_max_threads();
    char *fnameb = NULL;

    int c;
    opterr = 0;

    while ((c = getopt (argc, argv, "f:r:n:h")) != -1)
    {
        switch (c)
        {
        case 'h':
            usage();
            break;
        case 'f':
            fvalue = optarg;
            fnameb = fvalue;
            break;
        case 'r':
            rvalue = optarg;
            root = atoi(rvalue);
            break;
            break;
        case 'n':
            nvalue = optarg;
            numThreads = atoi(nvalue);
            break;
        case '?':
            if (optopt == 'f')
                fprintf (stderr, "Option -%c <graph file> requires an argument  .\n", optopt);
            else if (optopt == 'r')
                fprintf (stderr, "Option -%c [root] requires an argument.\n", optopt);
            else if (optopt == 'n')
                fprintf (stderr, "Option -%c [num threads] requires an argument.\n", optopt);
            else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                         "Unknown option character `\\x%x'.\n",
                         optopt);
            usage();
            return 1;
        default:
            abort ();
        }
    }


    //Set number of threads for the program
    omp_set_nested(1);
    omp_set_num_threads(numThreads);

#ifdef OPENMP_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "OPENMP Implementation");
#endif

#ifdef MPI_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "MPI Implementation");
#endif

#ifdef HYBRID_HARNESS
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Hybrid (OMP+MPI) Implementation");
#endif

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "File Name");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", fnameb);
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Number of Threads");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", numThreads);
    printf(" -----------------------------------------------------\n");


    struct Timer *timer = (struct Timer *) malloc(sizeof(struct Timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "New graph calculating size");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph = newGraph(fnameb);
    Stop(timer);
    printMessageWithtime("New Graph Created", Seconds(timer));


    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Populate Graph with edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    // populate the edge array from file
    loadEdgeArray(fnameb, graph);
    Stop(timer);
    printMessageWithtime("Time load edges to graph (Seconds)", Seconds(timer));



    // you need to parallelize this function
    
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "COUNT Sort Graph");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    struct Graph *graph1;
    graph1 = countSortEdgesBySource(graph); // you need to parallelize this function
    Stop(timer);
    printMessageWithtime("Time taken for serial count sort (Seconds)", Seconds(timer));

    Start(timer);
    

    #ifdef HYBRID_HARNESS
    //graph = radixSortEdgesBySourceMPI(graph,argc,&argv);

    
    int number_of_edges = graph->num_edges;

    struct Edge* out_array = NULL;
    struct Edge* out_array1 = NULL;
    
    struct Edge* local_out_array_dummy = NULL;

    //out_array = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    //out_array1 = (struct Edge *)malloc ( number_of_edges * sizeof(struct Edge) ) ;
    out_array = newEdgeArray(number_of_edges);
    
    out_array1 = newEdgeArray(number_of_edges);


    if(verbose)
    {
        printf("\nGRAPH\n");

         for(int y=0;y<number_of_edges;y++)
         {
             printf("%d -> %d ",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
         }
    }

    #pragma omp parallel for num_threads(2)
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

    for(u=1;u<=num_of_traversals;u++)
    {

    
     int index,i,temp_num=0;
    int elements_per_process;

    if(verbose)
        printf("\n NUMBER OF PROCESS:%d\n",np);

    int r = np;
    c = num_of_elements_in_radix;
    int* count_array = (int*)malloc(c * sizeof(int*));

    #pragma omp parallel for num_threads(2)
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
                    MPI_Send(&elements_per_process,1,MPI_INT,i,i,MPI_COMM_WORLD);
                    
                    for(int x=0;x<elements_per_process;x++)
                        MPI_Send(&out_array[index+x],1,mpi_graph,i,i,MPI_COMM_WORLD);
                }

                //index = i*elements_per_process;
                int start = i*count + (i<remainder?i:remainder);
                int stop = (i + 1) * count + ((i+1)<remainder?(i+1):remainder);
                index = i*elements_per_process;

                index = start;
                
                int elements_left = number_of_edges - index;
                elements_left = stop - start;

                if(verbose)
                printf("\nELEMENTSLEFT_%d\n",elements_left);

                MPI_Send(&elements_left,1,MPI_INT,i,i,MPI_COMM_WORLD);
                for(int x=0;x<elements_left;x++)
                    MPI_Send(&out_array[index+x],1,mpi_graph,i,i,MPI_COMM_WORLD);

            } 
            #pragma omp barrier

            int t=0;
            int start = pid*count + (pid<remainder?pid:remainder);
            int stop = (pid + 1) * count + ((pid+1)<remainder?(pid+1):remainder);
            index = pid*elements_per_process;

            index = start;
            elements_per_process = stop - start;
            if(verbose)
                printf("\nELEMENTS_%d\n",elements_per_process);
            
            #pragma omp parallel for num_threads(numThreads)
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
            #pragma omp parallel for num_threads(numThreads)
            for(i=1;i<num_of_elements_in_radix;i++)
            {
                
                count_array[i] += count_array[i-1];
            }

            struct Edge* local_out_array_dummy = newEdgeArray(elements_per_process);

            #pragma omp parallel for num_threads(numThreads)
            for(i=elements_per_process-1;i>=0;i--)
            {
                t = out_array[i].src;
                // printf("tttt_%d ",t);
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                    

                if(verbose)
                {   
                    printf("TEMP_NUM_MAS_%d ",temp_num);
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

            #pragma omp parallel for num_threads(numThreads)
            for(int i=0;i<elements_per_process;i++)
            {
                out_array1[i] = local_out_array_dummy[i];
            }
            #pragma omp barrier
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

            if(verbose)
            {
                for(int i=0;i<number_of_edges;i++)
                {
                    printf("FOUT_%d -> %d",out_array[i].src,out_array[i].dest);
                }  
            }

            #pragma omp parallel for num_threads(numThreads)
            for(int y=0;y<graph->num_edges;y++)
            {
                graph->sorted_edges_array[y] = out_array[y];
            }
            #pragma omp barrier

            if(verbose_print)
            {
                printf("\n FINAL OUTPUT\n");
                for(int y=0;y<graph->num_edges;y++)
                {
                    printf("%d ->  %d \n",graph->sorted_edges_array[y].src,graph->sorted_edges_array[y].dest);
                }
            }
            
    }
    
    #endif
    Stop(timer);
    printMessageWithtime("Time Sorting for Parallel Radix Sort (Seconds)", Seconds(timer));

    //Local Testing
    if(testing)
    {

    bool flag = true;
    int ind = -1;
    for(int y=0;y<graph->num_edges;y++)
    {
        if((graph->sorted_edges_array[y].src!=graph1->sorted_edges_array[y].src)||(graph->sorted_edges_array[y].dest != graph1->sorted_edges_array[y].dest))
        {
            ind = y;
            flag = false;
            break;
        }
    }

    if(flag == true)
        printf("\n TEST SUCCESS\n");
    else
        printf("\n TEST FAILED at index %d\n",ind);

    }
    //local testing ends

    // For testing purpose.

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "Map vertices to Edges");
    printf(" -----------------------------------------------------\n");
    Start(timer);
    mapVertices(graph);
    Stop(timer);
    printMessageWithtime("Time Mapping (Seconds)", Seconds(timer));

    printf(" *****************************************************\n");
    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "BFS Algorithm (PUSH/PULL)");
    printf(" -----------------------------------------------------\n");

    printf("| %-51s | \n", "PUSH");

    printf(" -----------------------------------------------------\n");
    printf("| %-51s | \n", "ROOT/SOURCE");
    printf(" -----------------------------------------------------\n");
    printf("| %-51u | \n", root);
    printf(" -----------------------------------------------------\n");
    Start(timer);

    breadthFirstSearchGraphPush(root, graph);

    // for (int p = 0; p < 10; ++p)
    // {
    //   breadthFirstSearchGraphPush(p, graph);
    // }

    Stop(timer);
    printMessageWithtime("Time BFS (Seconds)", Seconds(timer));

    Start(timer);
    freeGraph(graph);
    Stop(timer);
    printMessageWithtime("Free Graph (Seconds)", Seconds(timer));
    
    
    }


    else
    {

    
    //slave processes
    int u;
    for(u=1;u<=num_of_traversals;u++)
    {

    
    int n_elements_received;
            MPI_Recv(&n_elements_received,1,MPI_INT,0,pid,MPI_COMM_WORLD,&status);
            
            //struct Edge *a = (struct Edge *)malloc ( n_elements_received * sizeof(struct Edge) ) ;

            struct Edge *a = newEdgeArray(n_elements_received);
            int o=0;
            
            for(int x=0;x<n_elements_received;x++)
            {
                struct Edge a1;
                MPI_Recv(&a1,1,mpi_graph,0,pid,MPI_COMM_WORLD,&status);
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
            int i,temp_num;
            #pragma omp parallel for num_threads(numThreads)
            for(i=0;i<num_of_elements_in_radix;i++)
            {
                local_count_array[i] = 0;
            }
            #pragma omp barrier
            int t = 0;
            #pragma omp parallel for num_threads(numThreads)
            for(i=0;i<n_elements_received;i++)
            {
                t = a[i].src;
                //printf(" A_%d",a[i]);
                
                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                    
                
                //printf("\n TEMP NUM: %d\n",temp_num);
                local_count_array[temp_num] += 1;
                //printf("\n L_%d ",local_count_array[temp_num]);
            }
            #pragma omp barrier
            if(verbose)
            {
                printf("\n slave PROCESS: %d\n",pid);
                for(i=0;i<num_of_elements_in_radix;i++)
                {
                    printf("P_%d ",local_count_array[i]);
                }
            }

            //Prefix sum
            #pragma omp parallel for num_threads(numThreads)
            for(i=1;i<num_of_elements_in_radix;i++)
            {
                
                local_count_array[i] += local_count_array[i-1];
            }
            #pragma omp barrier
            if(verbose)
            {
                printf("\n slave PROCESS: %d\n",pid);
                for(i=0;i<num_of_elements_in_radix;i++)
                {
                    printf("PP_%d ",local_count_array[i]);
                }
                printf("\n");
            }
            #pragma omp parallel for num_threads(numThreads)
            for(i=n_elements_received-1;i>=0;i--)
            {
                t = a[i].src;
                
                if(verbose)
                    printf("\nt_%d",t);

                temp_num = (unsigned int)(((t >> ((u-1)*radix))) & (num_of_elements_in_radix - 1));
                

                out_array_local[local_count_array[temp_num] - 1] =  a[i];
                local_count_array[temp_num] -= 1;
            }
            #pragma omp barrier
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


          
        }
    }
    MPI_Type_free(&mpi_graph);
    MPI_Finalize();

    

    

    #endif

    return 0;
}


