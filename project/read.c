#include <stdio.h>
#include <stdlib.h>

/*
for each superblock, we will store size of superblock (close to a page), and indices of source/destination that they will hit

Then, for each superblock, we will have to iterate through the blocks and perform the block-vector computation
*/

int main(int argc, char *argv[])
{
        // READ THE MATRIX 
        FILE *matrix_in;
        matrix_in = fopen(argv[1], "r");
        if (matrix_in == NULL)
        {
                printf("Could not open matrix file");
                exit(1);
        }
        int num_rows, num_cols; 
        int num_sblocks;
        int **superblocks;
        
        // read in the number of superblocks
        fscanf(matrix_in, "%d %d %d ", &num_rows, &num_cols, &num_sblocks);
        // allocate space for num_sblocks int pointers (int pointer is the 'same' as int array)
        superblocks = malloc(sizeof(int *)*num_sblocks);
        
        // will be reused 
        int sblock_i, block_i;
        int num_blocks, sblock_size;
        int block_id, block_size;
        int* cur_sblock;
        for (sblock_i = 0; sblock_i < num_sblocks; sblock_i++)
        {
                // read in the current superblock size and the number of blocks
                fscanf(matrix_in, "%d %d ", &sblock_size, &num_blocks);
                superblocks[sblock_i] = malloc(sizeof(int)*(sblock_size + 1));
                cur_sblock = superblocks[sblock_i];

                cur_sblock[0] = num_blocks;
                cur_sblock += 1;
                for (block_i=0; block_i < num_blocks; block_i++)
                {
                        // read the block id into the current sblock array
                        fscanf(matrix_in, "%d ", cur_sblock);
                        switch(cur_sblock[0])
                        {
                                case 1:
                                        // 1x1 matrix
                                        fscanf(matrix_in, "%d %d %d ", &cur_sblock[1], &cur_sblock[2], &cur_sblock[3]);

                                        cur_sblock += 4;
                                        break;
                                case 2:
                                        // 2x2 matrix
                                        fscanf(matrix_in, "%d %d %d %d %d %d ",
                                                                &cur_sblock[1], &cur_sblock[2], 
                                                                &cur_sblock[3], &cur_sblock[4],
                                                                &cur_sblock[5], &cur_sblock[6]);
                                                              
                                        /**
                                        printf("R: %d C: %d Vals: %d %d %d %d\n", 
                                                                cur_sblock[1], cur_sblock[2], 
                                                                cur_sblock[3], cur_sblock[4],
                                                                cur_sblock[5], cur_sblock[6]);
                                                                */
                                        
                                        cur_sblock += 7; 
                                        break;
                                case 3:
                                        // 3x3 matrix
                                        fscanf(matrix_in, "%d %d %d %d %d %d %d %d %d %d %d ",
                                                                &cur_sblock[1], &cur_sblock[2], 
                                                                &cur_sblock[3], &cur_sblock[4],
                                                                &cur_sblock[5], &cur_sblock[6],
                                                                &cur_sblock[7], &cur_sblock[8],
                                                                &cur_sblock[9], &cur_sblock[10],
                                                                &cur_sblock[11]);
                                        
                                        /**
                                        printf("R: %d C: %d Vals: %d %d %d %d %d %d %d %d %d\n", 
                                                                cur_sblock[1], cur_sblock[2], 
                                                                cur_sblock[3], cur_sblock[4],
                                                                cur_sblock[5], cur_sblock[6],
                                                                cur_sblock[7], cur_sblock[8],
                                                                cur_sblock[9], cur_sblock[10],
                                                                cur_sblock[11]); 
                                                                */
                                        
                                        cur_sblock += 12; 
                                        break;
                                case 4:
                                        // 4x4 matrix
                                        fscanf(matrix_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ", 
                                                                &cur_sblock[1], &cur_sblock[2], 
                                                                &cur_sblock[3], &cur_sblock[4],
                                                                &cur_sblock[5], &cur_sblock[6],
                                                                &cur_sblock[7], &cur_sblock[8],
                                                                &cur_sblock[9], &cur_sblock[10],
                                                                &cur_sblock[11], &cur_sblock[12],
                                                                &cur_sblock[13], &cur_sblock[14],
                                                                &cur_sblock[15], &cur_sblock[16],
                                                                &cur_sblock[17], &cur_sblock[18]);
                                        
                                        /*
                                        printf("R: %d C: %d Vals: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", 
                                                                cur_sblock[1], cur_sblock[2], 
                                                                cur_sblock[3], cur_sblock[4],
                                                                cur_sblock[5], cur_sblock[6],
                                                                cur_sblock[7], cur_sblock[8],
                                                                cur_sblock[9], cur_sblock[10],
                                                                cur_sblock[11], cur_sblock[12],
                                                                cur_sblock[13], cur_sblock[14],
                                                                cur_sblock[15], cur_sblock[16],
                                                                cur_sblock[17], cur_sblock[18]);
                                                                */
                                        
                                        cur_sblock += 19; 
                                        break;
                                default:
                                        exit(1);
                        
                        }                        
                        
                }
        }
        fclose(matrix_in);

        /*
        READ THE SOURCE VECTOR
        */

        FILE *src_vec_in;
        src_vec_in = fopen(argv[2], "r");
        if (src_vec_in == NULL)
        {
                printf("Could not open src vector file");
                exit(1);
        }
        int x_len;
        fscanf(src_vec_in, "%d ", &x_len);
        if (x_len != num_cols)
        {
                printf("Matrix and vector have incompatible dims\n");
                exit(1);
        }
        printf(" About ot alloc X len");
        int* x = malloc(sizeof(int) * x_len);

        printf(" About ot print X len\n");
        printf("X len %d\n", x_len);
        if (x == NULL)
        {
                printf("Could not malloc x");
                exit(1);
        }
        int i;
        for(i = 0; i < x_len; i++)
        {            
                fscanf(src_vec_in, "%d ", &x[i]);                
        }
        
        // allocate the destination vector
        printf("Beforn\n");
        int* y = malloc(sizeof(int) * num_rows);       

        for(i=0; i < num_rows; i++)
        {
                y[i] = 0;
        }

        if (y == NULL)
        {
                printf("Could not malloc y");
                exit(1);
        }
        

        /*
        Do the multiplication
        */
        // iterate through the superblocks in outer loop
        int r, c;
        for (sblock_i = 0; sblock_i < num_sblocks; sblock_i++)
        {
                //get current superblock
                cur_sblock = superblocks[sblock_i];
                num_blocks = cur_sblock[0];

                // move the pointer up by 1
                cur_sblock += 1;
                for(block_i = 0; block_i < num_blocks; block_i++)
                {
                        printf("Cur s block %d \n", cur_sblock[0]);

                        // cur_sblock[1] -> row
                        // cur_sblock[2] -> col
                        r = cur_sblock[1];
                        c = cur_sblock[2]; 
                        

                        // cur_sblock[0] -> block_id of current block 
                        switch(cur_sblock[0])
                        {                                

                                case 1:
                                        // 1x1 block
                                        y[r] += cur_sblock[3] * x[c];
                                        // move to next block
                                        cur_sblock += 4;
                                        break;
                                case 2:
                                        // 2x2 block                                    
                                        y[r]   += cur_sblock[3] * x[c];
                                        y[r]   += cur_sblock[4] * x[c+1];

                                        y[r+1] += cur_sblock[5] * x[c];                                                                         
                                        y[r+1] += cur_sblock[6] * x[c+1];
                                        
                                        // move to next block
                                        cur_sblock += 7;
                                        break;
                                case 3:
                                        // 3x3 block
                                        y[r]   += cur_sblock[3] * x[c]; 
                                        y[r]   += cur_sblock[4] * x[c+1];
                                        y[r]   += cur_sblock[5] * x[c+2];
                                        
                                        y[r+1] += cur_sblock[6] * x[c]; 
                                        y[r+1] += cur_sblock[7] * x[c+1]; 
                                        y[r+1] += cur_sblock[8] * x[c+2]; 

                                        y[r+2] += cur_sblock[9] * x[c]; 
                                        y[r+2] += cur_sblock[10] * x[c+1]; 
                                        y[r+2] += cur_sblock[11] * x[c+2]; 

                                        // move to next block
                                        cur_sblock += 12;
                                        break;
                                case 4: 
                                        // 4 x 4 block
                                        y[r]   += cur_sblock[3] * x[c];
                                        y[r]   += cur_sblock[4] * x[c+1];
                                        y[r]   += cur_sblock[5] * x[c+2];
                                        y[r]   += cur_sblock[6] * x[c+3];

                                        y[r+1] += cur_sblock[7] * x[c];
                                        y[r+1] += cur_sblock[8] * x[c+1];
                                        y[r+1] += cur_sblock[9] * x[c+2];
                                        y[r+1] += cur_sblock[10] * x[c+3];

                                        y[r+2] += cur_sblock[11] * x[c];
                                        y[r+2] += cur_sblock[12] * x[c+1];
                                        y[r+2] += cur_sblock[13] * x[c+2];
                                        y[r+2] += cur_sblock[14] * x[c+3];

                                        y[r+3] += cur_sblock[15] * x[c];
                                        y[r+3] += cur_sblock[16] * x[c+1];
                                        y[r+3] += cur_sblock[17] * x[c+2];
                                        y[r+3] += cur_sblock[18] * x[c+3];
                                        
                                        // move to next block
                                        cur_sblock += 19;
                                        break;
                                default:
                                        printf("Unsupported block id %d", cur_sblock[0]);
                                        exit(1);
                        }
                }
        }        

        FILE *result_writer;
        result_writer = fopen("calculated_result.txt", "w");

        // print out the result 
        for (i=0; i<num_rows; i++)
        {
                printf("%d ", y[i]);
                fprintf(result_writer, "%d ", y[i]);
        }

        fclose(result_writer);
        return 0; 
}


// assume we have one block size that is known at compile time (4 x 4) 
void sparsity_4x4(int argc, char *argv[])
{
        // READ THE MATRIX 
        FILE *matrix_in;
        matrix_in = fopen(argv[1], "r");
        if (matrix_in == NULL)
        {
                printf("Could not open matrix file");
                exit(1);
        }
        int num_rows, num_cols, num_sblocks; 
        int **superblocks;
        
        // read in the number of superblocks
        fscanf(matrix_in, "%d %d %d ", &num_rows, &num_cols, &num_sblocks);
        // allocate space for num_sblocks int pointers (int pointer is the 'same' as int array)
        superblocks = malloc(sizeof(int *)*num_sblocks);
        
        // will be reused 
        int sblock_i, block_i;
        int num_blocks, sblock_size;
        int block_id, block_size;
        int* cur_sblock;
        // superblocks are processed in row-major order
        for (sblock_i = 0; sblock_i < num_sblocks; sblock_i++)
        {
                // read in the current superblock size and the number of blocks
                fscanf(matrix_in, "%d %d ", &sblock_size, &num_blocks);
                superblocks[sblock_i] = malloc(sizeof(int)*(sblock_size + 1));
                cur_sblock = superblocks[sblock_i];
                cur_sblock[0] = num_blocks;
                cur_sblock += 1;
                // could potentially create RSE array here        
                for (block_i=0; block_i < num_blocks; block_i++)
                {
						// each block contains row, col, then 16 vals                        
						// 4x4 matrix
						fscanf(matrix_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ", 
												&cur_sblock[0], &cur_sblock[1], 
												&cur_sblock[2], &cur_sblock[3],
												&cur_sblock[4], &cur_sblock[5],
												&cur_sblock[6], &cur_sblock[7],
												&cur_sblock[8], &cur_sblock[9],
												&cur_sblock[10], &cur_sblock[11],
												&cur_sblock[12], &cur_sblock[13],
												&cur_sblock[14], &cur_sblock[15],
												&cur_sblock[16], &cur_sblock[17]);
						
						/*
						printf("R: %d C: %d Vals: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", 
												cur_sblock[1], cur_sblock[2], 
												cur_sblock[3], cur_sblock[4],
												cur_sblock[5], cur_sblock[6],
												cur_sblock[7], cur_sblock[8],
												cur_sblock[9], cur_sblock[10],
												cur_sblock[11], cur_sblock[12],
												cur_sblock[13], cur_sblock[14],
												cur_sblock[15], cur_sblock[16],
												cur_sblock[17], cur_sblock[18]);
												*/
						
						cur_sblock += 18; 
                        
                }                        
                        
        }
        
        fclose(matrix_in);
		
		// READ THE SOURCE VECTOR
        FILE *src_vec_in;
        src_vec_in = fopen(argv[2], "r");
        if (src_vec_in == NULL)
        {
                printf("Could not open src vector file");
                exit(1);
        }
        int x_len;
        fscanf(src_vec_in, "%d ", &x_len);
        if (x_len != num_cols)
        {
                printf("Matrix and vector have incompatible dims\n");
                exit(1);
        }
        printf(" About to alloc X len");
        int* x = malloc(sizeof(int) * x_len);

        printf(" About to print X len\n");
        printf("X len %d\n", x_len);
        if (x == NULL)
        {
                printf("Could not malloc x");
                exit(1);
        }
        int i;
        for(i = 0; i < x_len; i++)
        {            
                fscanf(src_vec_in, "%d ", &x[i]);                
        }
        
        // allocate the destination vector
        int* y = malloc(sizeof(int) * num_rows);       
		// initialize y to 0
        for(i=0; i < num_rows; i++)
        {
                y[i] = 0;
        }

        if (y == NULL)
        {
                printf("Could not malloc y");
                exit(1);
        }

		// do the multiplication
		int r, c, cur_row;
		r = 0; cur_row = 0;
        for (sblock_i = 0; sblock_i < num_sblocks; sblock_i++)
		{
			// process current superblock		
			cur_sblock = superblocks[sblock_i];
			num_blocks = cur_sblock[0];
			// move the pointer up by 1
			
			// get the current row
			cur_row = cur_sblock[1];
			r = cur_row; 
			// read in the register integers
			cur_sblock += 1;
			
			register int i0 = y[r];
			register int i1 = y[r+1];
			register int i2 = y[r+2];
			register int i3 = y[r+3];

			for(block_i=0; block_i < num_blocks; block_i++)
			{
				
				printf("Cur s block %d \n", cur_sblock[0]);
	
				// cur_sblock[1] -> row
				// cur_sblock[2] -> col
				r = cur_sblock[0];
				c = cur_sblock[1]; 
				if (r != cur_row)
				{
					// write the value of the i0, i1, i2, i3 to y
					y[cur_row]   = i0;
					y[cur_row+1] = i1;
					y[cur_row+2] = i2;
					y[cur_row+3] = i3;
					
					// update register variables to new destination locations
					cur_row = r;
					i0 = y[r];
					i1 = y[r+1];
					i2 = y[r+2];
					i3 = y[r+3];
				}
				// 4 x 4 block
				i0 += cur_sblock[2] * x[c];
				i0 += cur_sblock[3] * x[c+1];
				i0 += cur_sblock[4] * x[c+2];
				i0 += cur_sblock[5] * x[c+3];

				i1 += cur_sblock[6] * x[c];
				i1 += cur_sblock[7] * x[c+1];
				i1 += cur_sblock[8] * x[c+2];
				i1 += cur_sblock[9] * x[c+3];

				i2 += cur_sblock[10] * x[c];
				i2 += cur_sblock[11] * x[c+1];
				i2 += cur_sblock[12] * x[c+2];
				i2 += cur_sblock[13] * x[c+3];

				i3 += cur_sblock[14] * x[c];
				i3 += cur_sblock[15] * x[c+1];
				i3 += cur_sblock[16] * x[c+2];
				i3 += cur_sblock[17] * x[c+3];
				
				// move to next block
				cur_sblock += 18;
								

			}
			
		}

		// Write the result
		FILE *result_writer;
        result_writer = fopen("calculated_result.txt", "w");

        // print out the result 
        for (i=0; i<num_rows; i++)
        {
                printf("%d ", y[i]);
                fprintf(result_writer, "%d ", y[i]);
        }

        fclose(result_writer); 
        
        // free memory?
}


void smvm_2x2( int bm, const int *b_row_start, const int *b_col_idx, const double *b_value, const double *x, double *y)
{
        int i, jj;
        for (i =0; i<bm; i++, y+=2){
                register double d0 = y[0];
                register double d1 = y[1];
                for (jj = b_row_start[i]; jj<b_row_start[i+1];
                         jj++, b_col_idx++, b_value += 2*2) {
                        d0 += b_value[0] * x[b_col_idx[0]+0];
                        d1 += b_value[2] * x[b_col_idx[0]+0];
                        d0 += b_value[1] * x[b_col_idx[0]+1];
                        d1 += b_value[3] * x[b_col_idx[0]+1];
                }
                y[0]=d0;
                y[1]=d1;
        }
}

