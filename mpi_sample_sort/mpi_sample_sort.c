/**
sample sort MPI
*/

#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<mpi.h>
#include<math.h>

typedef struct
{
	int *ptr;
	int len; // Length or Next Index
} Bucket;

void bucket_push(Bucket* bucket, int i)
{
	bucket->ptr[bucket->len++] = i;
}

// For qsort() callback
int compare(const void *a, const void *b)
{
	return (*((int*) a) - *((int *) b));
}

void sort(const int rank, const int size, const char *file, const int debug)
{
	if(debug > 1) printf("[COMMON] Working %u/%u\n", rank, size);
	
	int count_buckets = size; // As the same of process count
	int size_input; // The total count of input data 
	
	double start, end; // Timer
	
	MPI_Request request;
	MPI_Status status;
	
	int* int_buf = NULL; // Store data read by rank 0
	if(rank == 0){
		if(debug > 1) printf("[MASTER] Read file: %s\n", file);
		FILE *fp;
		fp = fopen(file,"r");
		if(fp == NULL){
			fprintf(stderr, "sort(): '%s' is not a valid file for read.\n", file);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
		int* ptr_temp; // Memory safe
		for(size_input = 0; !feof(fp); size_input++){
			ptr_temp = (int*) realloc(int_buf, sizeof(int) * (size_input + 1));
			if(ptr_temp){
				int_buf = ptr_temp;
				fscanf(fp,"%d",int_buf + size_input);
			}else{
				if(int_buf) free(int_buf);
				fprintf(stderr, "sort(): '%s' is not a valid file for read.\n", file);
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
		}
		start = MPI_Wtime();
		if(debug > 1) printf("[MASTER] File read OK, %u numbers %u-%u.\n", size_input, int_buf[0], int_buf[size_input - 1]);
		for(int i = 1; i < size; i++) MPI_Isend(&size_input, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
		if(debug > 1) printf("[MASTER] Isend(size_input).\n", size_input);
	}
	if(rank != 0){
		MPI_Recv(&size_input, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		if(debug > 1) printf("[SLAVE] %u Recv(size_input): %u\n", rank, size_input);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	int size_bucket = size_input % count_buckets > 0 ? size_input / count_buckets + 1 : size_input / count_buckets;
	int size_small_bucket = size_input - (size_bucket * (count_buckets - 1));
	if(rank == 0) printf("Each bucket will be put %u items.\n", size_bucket);
	// End of data read
	
	// Start sample
	// Step 1: Sort the parts divided by count of buckets (k)
	int* initial_sort;
	int size_initial_bucket = (rank == size - 1) ? size_small_bucket : size_bucket;
	initial_sort = (int*) calloc(sizeof(int), size_initial_bucket);
	MPI_Scatter(int_buf, size_bucket, MPI_INT, initial_sort, size_bucket, MPI_INT, 0, MPI_COMM_WORLD); // No need to identify rank
	//MPI_Barrier(MPI_COMM_WORLD);
	if(debug) printf("[VERBOSE] %u: Initial before sort %u - %u\n", rank, initial_sort[0], initial_sort[size_initial_bucket - 1]);
	qsort(initial_sort, size_initial_bucket, sizeof(int), compare);
	if(debug) printf("[VERBOSE] %u: Initial after sort %u - %u\n", rank, initial_sort[0], initial_sort[size_initial_bucket - 1]);
	if(debug) printf("[COMMON] %u: MPI_Scatter and step 1 OK\n", rank);
	// Step 2: Select 2k - 1 evenly spaced items and send to master
	int size_initial_collect = 2 * count_buckets - 1;
	int interval = size_bucket / size_initial_collect;
	Bucket initial_bucket; // master use only
	initial_bucket.ptr = (int*) calloc(sizeof(int), count_buckets * size_initial_collect);
	initial_bucket.len = 0;
	for(int i = 0; i < size_initial_collect; i++){
		int index = i * interval;
		if(index >= size_initial_bucket){ // no more data
			fprintf(stderr, "[ERROR] %u: no enough sample, try smaller processes. This=%u, Expected index=%u, Max Index=%u.\n", rank, i, index, size_initial_bucket - 1);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
		if(rank > 0){
			MPI_Isend(initial_sort + index, 1, MPI_INT, 0, i, MPI_COMM_WORLD, &request);
		}else{
			bucket_push(&initial_bucket, *(initial_sort + index));
		}
	}
	if(debug) printf("[COMMON] %u: STEP2 distributed.\n", rank);
	// Step 3: Master sort the k(2k - 1) items and select k - 1 splitters
	int* splitters = calloc(sizeof(int), count_buckets - 1);
	if(rank == 0){ // Master recv
		for(int i = 1; i < count_buckets; i++){
			for(int j = 0; j < size_initial_collect; j++){
				MPI_Recv(initial_bucket.ptr + i * size_initial_collect + j, 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				initial_bucket.len++;
			}
		}
		qsort(initial_bucket.ptr, initial_bucket.len, sizeof(int), compare);
		if(debug > 1){
			printf("[MASTER] STEP3 sorted:");
			for(int i = 0; i < initial_bucket.len; i++) printf(" %u", initial_bucket.ptr[i]);
			printf("\n");
		}
		for(int i = 0; i < count_buckets - 1; i++){
			splitters[i] = initial_bucket.ptr[(i + 1) * size_initial_collect];
			if(debug && rank == 0) printf("[MASTER] Splitter: %u.\n", splitters[i]);
		}
		for(int i = 1; i < size; i++) MPI_Isend(splitters, count_buckets - 1, MPI_INT, i, 1, MPI_COMM_WORLD, &request);
		if(debug) printf("[MASTER] STEP3 Isend(splitters).\n");
	}
	// Step 4: Others get copy of splitters
	if(rank != 0){
		MPI_Recv(splitters, count_buckets - 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if(debug) printf("[SLAVE] %u: STEP4 Recv(splitters).\n", rank);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	free(initial_bucket.ptr);
	initial_bucket.len = 0;
	// Step 5: Real Bucket Sort
	// Create data structure based on count_buckets and size_of_bucket, TODO: assert malloc
	int max_size_bucket = (int) floor(size_bucket * 1.5);
	Bucket ** buckets = (Bucket**) calloc(sizeof(Bucket*), count_buckets);
	for(int i = 0; i < count_buckets; i++){
		buckets[i] = (Bucket*) malloc(sizeof(Bucket));
		buckets[i]->ptr = (int*) calloc(sizeof(int), max_size_bucket * 2);
		buckets[i]->len = 0;
	}
	// Check every item already in my initial_sort
	for(int i = 0; i < size_initial_bucket; i++){
		for(int j = 0; j < count_buckets; j++){
			if((j == count_buckets - 1 && initial_sort[i] > splitters[j - 1]) || (j < count_buckets - 1 && initial_sort[i] <= splitters[j])){
				bucket_push(buckets[j], initial_sort[i]);
				break;
			}
		}
	}
	if(debug){
		for(int i = 0; i < count_buckets; i++) printf("[COMMON] %u: Bucket %u=%u\n", rank, i, buckets[i]->len);
	}
	// Send data belongs to others
	for(int i = 0; i < size; i++){
		if(i != rank) MPI_Isend(buckets[i]->ptr, max_size_bucket, MPI_INT, i, buckets[i]->len, MPI_COMM_WORLD, &request);
	}
	if(debug) printf("[COMMON] %u: Isend(buckets[i]->ptr).\n", rank);
	// Recv data belongs to me
	int size_current_bucket = buckets[rank]->len;
	for(int i = 0; i < size - 1; i++){
		MPI_Recv(&buckets[rank]->ptr[size_current_bucket], max_size_bucket, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		size_current_bucket += status.MPI_TAG; // real quantity read
	}
	buckets[rank]->len = size_current_bucket; // update
	if(debug) printf("[COMMON] %u: Recv(buckets[rank]->ptr) LEN=%u.\n", rank, size_current_bucket);
	// Sort own data
	if(debug) printf("[VERBOSE] %u: Main Before sort %u - %u\n", rank, buckets[rank]->ptr[0], buckets[rank]->ptr[size_current_bucket - 1]);
	qsort(buckets[rank]->ptr, size_current_bucket, sizeof(int), compare);
	if(debug){
		printf("[VERBOSE] %u: Main Sorted %u - %u\n", rank, buckets[rank]->ptr[0], buckets[rank]->ptr[size_current_bucket - 1]);
		if(debug > 2) for(int i = 0; i < size_current_bucket; i++) printf("%u|%u|%u\n", rank, i, buckets[rank]->ptr[i]);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Gather bucket sizes of everyone
	int* sizes_buckets = (int*) calloc(sizeof(int), size);
	MPI_Gather(&size_current_bucket, 1, MPI_INT, sizes_buckets, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(debug) printf("[COMMON] %u: Bucket size gathered.\n", rank);
	// Offset of first element in each bucket in final array
	int* offsets_buckets = (int*) calloc(sizeof(int), size);
	if(rank == 0){
		offsets_buckets[0] = 0;
		for(int i = 1; i < size; i++) offsets_buckets[i] = offsets_buckets[i - 1] + sizes_buckets[i - 1];
	}
	// Gather with known offset to master
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gatherv(buckets[rank]->ptr, size_current_bucket, MPI_INT, int_buf, sizes_buckets, offsets_buckets, MPI_INT, 0, MPI_COMM_WORLD);
	if(debug) printf("[COMMON] %u: Data gathered.\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// Do output as required
	if(rank == 0){
		end = MPI_Wtime();
		if(debug){
			for(int i = 0; i < size_input; i++) printf("%u|%u\n", i, int_buf[i]);
		}
		printf("The n/2-th sorted element: %d\n",int_buf[size_input / 2 - 1]);
		free(int_buf);
		fprintf(stderr,"Endtime()-Starttime() = %.5f sec\n",end - start);
	}
   
	/*free(offsets_buckets);
	free(sizes_buckets);
	for(int i = 0; i < count_buckets; i++){
		free(buckets[i]->ptr);
		free(buckets + i);
	}
	free(buckets);
	free(splitters);*/
}

int main(int argc, char *argv[])
{
	int rank;
	int size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Entry Check for filename
	if(argc != 2 && argc != 3){
		if(rank == 0) fprintf(stderr, "Usage: %s <file: Data file to read>\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(EXIT_FAILURE);
	}
	
	// Start working
	sort(rank, size, argv[1], (argc == 3 ? atoi(argv[2]) : 0));
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
