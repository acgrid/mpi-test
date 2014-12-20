/**
radix sort MPI
*/

#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<mpi.h>
#include<math.h>

#define TAG_SIZE_INPUT 1
#define TAG_LOOP 2

typedef struct
{
	int *ptr;
	int len; // Length or Next Index
	int max;
} Bucket;

void bucket_extend(Bucket *bucket){
	size_t max = bucket->max * 2;
	int* temp = realloc(bucket->ptr, max * sizeof(int));
	if(!temp){
		fprintf(stderr, "push_back(): Realloc ERROR\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(EXIT_FAILURE);
	}
	bucket->ptr = temp;
	bucket->max = max;
}

void bucket_push(Bucket *bucket, int i)
{
	if(bucket != NULL){
		if(bucket->len >= bucket->max) bucket_extend(bucket);
		bucket->ptr[bucket->len++] = i;
	}else{
		fprintf(stderr, "push_back(): Invalid List\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(EXIT_FAILURE);
	}
}

int number_digits(int value, const int radix);
int number_digit_at(int value, const int radix, int position);

int number_digits(int value, const int radix)
{
	value = abs(value);
	return (int) (log(value > 0 ? value : 1) / log(radix)) + 1;
}

int number_digit_at(int value, const int radix, int position)
{
	value = abs(value);
	return (int) ((value % ((int) pow((double) radix, (double) position))) / pow((double) radix, (double) position - 1));
}

void sort(const int rank, const int size, const char *file, const int debug)
{
	if(debug > 1) printf("[COMMON] Working %u/%u\n", rank, size);
	
	int radix = size; // As the same of process count
	int size_input; // The total count of input data
	int loop; // The max digits in selected radix
	
	double start, end; // Timer
	
	MPI_Request request;
	MPI_Status status;
	
	int* int_buf = NULL; // Store data read by rank 0
	if(rank == 0){
		if(debug > 1) printf("[MASTER] Read file: %s\n", file);
		FILE *fp;
		int max_element = -1;
		int this_element;
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
				fscanf(fp,"%d",&this_element);
				int_buf[size_input] = this_element;
				if(this_element > max_element) max_element = this_element;
			}else{
				if(int_buf) free(int_buf);
				fprintf(stderr, "sort(): '%s' is not a valid file for read.\n", file);
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
		}
		start = MPI_Wtime();
		if(debug > 1) printf("[MASTER] File read OK, %u numbers %u-%u.\n", size_input, int_buf[0], int_buf[size_input - 1]);
		loop = number_digits(max_element, radix);
		for(int i = 1; i < size; i++){
			MPI_Isend(&size_input, 1, MPI_INT, i, TAG_SIZE_INPUT, MPI_COMM_WORLD, &request);
			MPI_Isend(&loop, 1, MPI_INT, i, TAG_LOOP, MPI_COMM_WORLD, &request);
		}
		if(debug > 1) printf("[MASTER] Isend(size_input) Isend(loop).\n", size_input);
	}
	if(rank != 0){
		MPI_Recv(&size_input, 1, MPI_INT, 0, TAG_SIZE_INPUT, MPI_COMM_WORLD, &status);
		MPI_Recv(&loop, 1, MPI_INT, 0, TAG_LOOP, MPI_COMM_WORLD, &status);
		if(debug > 1) printf("[SLAVE] %u Recv(size_input): %u Recv(loop): %u\n", rank, size_input, loop);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	// Distribute initial data
	int size_batch = size_input % radix > 0 ? size_input / radix + 1 : size_input / radix;
	int size_small_batch = size_input - (size_batch * (radix - 1));
	
	int* initial_sort;
	int size_my_batch = (rank == size - 1) ? size_small_batch : size_batch;
	initial_sort = (int*) calloc(sizeof(int), size_my_batch);
	
	// Now work on initial_sort for loops
	Bucket ** buckets = (Bucket**) calloc(sizeof(Bucket*), radix);
	int size_init_bucket = 2 * size_batch;
	for(int i = 0; i < radix; i++){
		buckets[i] = (Bucket*) malloc(sizeof(Bucket));
		buckets[i]->ptr = (int*) calloc(sizeof(int), size_init_bucket);
		buckets[i]->max = size_init_bucket;
	}
	Bucket bucket_collected;
	bucket_collected.ptr = (int*) calloc(sizeof(int), size_init_bucket);
	bucket_collected.max = size_init_bucket;
	int* len_buckets = (int*) calloc(sizeof(int), size);
	for(int digit = 1; digit <= loop; digit++){
		for(int i = 0; i < radix; i++){
			buckets[i]->len = 0; // Reset
			len_buckets[i] = 0;
		}
		bucket_collected.len = 0;
		MPI_Scatter(int_buf, size_batch, MPI_INT, initial_sort, size_batch, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		
		if(debug) printf("[VERBOSE] %u: Scatter OK LOOP %u - %u\n", rank, initial_sort[0], initial_sort[size_my_batch - 1]);
		// Find target of sending
		for(int i = 0; i < size_my_batch; i++){
			int this_digit = number_digit_at(initial_sort[i], radix, digit);
			bucket_push(buckets[this_digit], initial_sort[i]);
		}
		if(debug) printf("[COMMON] %u: bucket_push END\n", rank);
		// Send data count
		for(int i = 0; i < size; i++){
			if(i != rank) MPI_Isend(&buckets[i]->len, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
		}
		//if(debug) printf("[COMMON] %u: Isend(buckets[i]->len).\n", rank);
		// recv data count
		len_buckets[rank] = buckets[rank]->len; // myself
		for(int i = 0; i < size; i++){
			if(i != rank){
				MPI_Recv(len_buckets + i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				//if(debug) printf("[COMMON] %u: Recv: len_buckets[%u] = %u.\n", rank, i, len_buckets[i]);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		// send data including my own
		for(int i = 0; i < size; i++){
			MPI_Isend(buckets[i]->ptr, buckets[i]->len, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
		}
		// recv data itself
		for(int i = 0; i < size; i++){
			if(bucket_collected.max < bucket_collected.len + len_buckets[i]) bucket_extend(&bucket_collected); // check capacity and extend if needed
			MPI_Recv(&bucket_collected.ptr[bucket_collected.len], len_buckets[i], MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			bucket_collected.len += len_buckets[i];
			//if(debug) printf("[COMMON] %u: Recv: len_buckets[%u] = %u.\n", rank, i, len_buckets[i]);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if(debug > 2){
			printf("[COMMON] %u: Main Queue Completed, LEN=%u\n", rank, bucket_collected.len);
			for(int i = 0; i < bucket_collected.len; i++) printf("DUMP: LOOP %u RADIX %u = %u\n", digit, rank, bucket_collected.ptr[i]);
		}
		// gather
		int* sizes_buckets = (int*) calloc(sizeof(int), size);
		MPI_Gather(&bucket_collected.len, 1, MPI_INT, sizes_buckets, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
		MPI_Gatherv(bucket_collected.ptr, bucket_collected.len, MPI_INT, int_buf, sizes_buckets, offsets_buckets, MPI_INT, 0, MPI_COMM_WORLD);
		if(debug) printf("[COMMON] %u: Data gathered.\n", rank);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if(rank == 0){
		end = MPI_Wtime();
		if(debug > 2){
			for(int i = 0; i < size_input; i++) printf("%u|%u\n", i, int_buf[i]);
		}
		printf("The n/2-th sorted element: %d\n",int_buf[size_input / 2 - 1]);
		free(int_buf);
		fprintf(stderr,"Endtime()-Starttime() = %.5f sec\n",end - start);
	}
}

int main(int argc,char* argv[])
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