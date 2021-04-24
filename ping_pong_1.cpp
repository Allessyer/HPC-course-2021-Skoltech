#include <mpi.h>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <cstdlib> 

using namespace std;

void add_to_passed_list(vector<int>& passed_list, int prank)
{
    passed_list[0]++; // update number of played players. 
    passed_list[passed_list[0]] = prank;
}

int send_to_process(vector<int>& passed_list, int psize)
{
    srand(time(NULL));
    vector<int> not_passed;

    for (int i = 0; i< psize;i++)
    {
        std::vector<int>::iterator it;
        it = find(passed_list.begin()+1,passed_list.end(), i);
        if (it == passed_list.end())
        {
            not_passed.push_back(i);
        }
    }
    
    if (not_passed.empty())
    {
         cout << "All players passed the ball!" << endl;
         return 100;
    }
    else
    {
        cout << "PRINT NOT PASSED: " << endl;

        for (int i =0; i < not_passed.size(); i++)
        {
	    cout << not_passed[i] << " ";
        }
        cout << endl;
        
        int rand_index = rand() % not_passed.size();
        cout << not_passed[rand_index] << endl;
        return not_passed[rand_index];
    }

    

}

void print_array(vector<int>& arr, int N) {
    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");
}





int main(int argc, char ** argv) 
{
    int psize;
    int prank;
    int  nameLen;
    char processorName[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Status status;
    time_t t;

    int ball = 0;

    int ierr;
    char name;
    
    //vector<int> passed_list;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &prank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &psize);
    vector<int> passed_list;
    passed_list.push_back(0); // number of played processors
    for (int i = 0; i < psize; i++)
    {
        passed_list.push_back(100); // played vector
    }


    // ПРАВИЛА ИГРЫ
    // принт имени процессора
    // добавление в список 
    // rand() между процессорами, которые не в списке 
    // отправление списка другому процессору
    


    if (prank == 0)
    {
        //init_players(not_passed,psize);
        
        add_to_passed_list(passed_list, prank);
        print_array(passed_list, passed_list.size());

        int to_process = 1;
 
	MPI_Ssend(&passed_list[0],psize+1, MPI_INT, 1, 13, MPI_COMM_WORLD);
    }
    else
    {
    	MPI_Recv(&passed_list[0],psize+1, MPI_INT, MPI_ANY_SOURCE, 13, MPI_COMM_WORLD, &status);
    	print_array(passed_list, passed_list.size());
	add_to_passed_list(passed_list, prank);
	
	int to_process = send_to_process(passed_list, psize);
	if (to_process != 100)
	{
	    MPI_Ssend(&passed_list[0],psize+1, MPI_INT, to_process, 13, MPI_COMM_WORLD);
	} else 
	{
	    cout << "Game was played in this order: " << endl;
	    for (int i = 1; i < psize + 1; i++)
	    {
	        cout << passed_list[i] << endl;
	    }
	}
	 
    	
    }
    
    ierr = MPI_Finalize();
    return 0;

}
