/* gcc -Wall -O3 -funroll-all-loops -mavx2  bnn-exec_4096_4096.c */

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include <time.h>
#include <string.h>
#include <stdlib.h> 
#include <unistd.h>

#define W_LEN 4096
#define X_LEN 4096/(sizeof(uint64_t)*8) 
#define SIGN_TRESH 4096/2
#define Y_LEN W_LEN/(sizeof(uint64_t)*8)

uint64_t start_w[W_LEN][Y_LEN] __attribute__ ((aligned (64))) = {0};
uint64_t X[X_LEN] = {0};
uint64_t Y[Y_LEN] = {0};


#define unlikely(x)    __builtin_expect(!!(x), 0)


uint32_t popcnt(const uint64_t* buf, int len) 
{
	assert(len % 4 == 0);
	uint64_t cnt[4] = {0};

	for (int i = 0; i < len; i+=4) {
		__asm__(
			"popcnt %4, %4  \n "
			"add %4, %0     \n\t"
			"popcnt %5, %5  \n\t"
			"add %5, %1     \n\t"
			"popcnt %6, %6  \n\t"
			"add %6, %2     \n\t"
			"popcnt %7, %7  \n\t"
			"add %7, %3     \n\t"
			: "+r" (cnt[0]), "+r" (cnt[1]), "+r" (cnt[2]), "+r" (cnt[3])
			: "r"  (buf[i]), "r"  (buf[i+1]), "r"  (buf[i+2]), "r"  (buf[i+3]));
	}
	return cnt[0] + cnt[1] + cnt[2] + cnt[3];
}


void inference(int sign_th,int x_len, int w_len, int y_len,  uint64_t X[x_len] ,uint64_t W[w_len][x_len], uint64_t Y[y_len])
{
	__m256i xor_res;
	uint32_t pop_res=0;
	uint32_t f_idx;
	uint64_t f = 0x8000000000000000;
	int i,j;
	__m256i w;
	__m256i x ;

	f_idx =0;
	uint64_t *ww;

	assert(x_len % 4 == 0);

	for (i = 0; i < w_len; i++) {

		pop_res = 0;
		ww = W[i];
		for (j = 0; j < x_len/4; j++) {

			w = _mm256_load_si256((__m256i*) ww+j);
			x = _mm256_load_si256((__m256i*) X+j);

			xor_res = _mm256_xor_si256(w, x);
			pop_res += popcnt((uint64_t *)&xor_res, 4);
			
		}
		

		if (pop_res <= sign_th)
			Y[f_idx] = Y[f_idx] | (f);
	
		if ( !((i+1) % (sizeof(uint64_t)*8)) ){
			f_idx++;
		}


		if (unlikely(f==1))
			f = 0x8000000000000000;
		else 
			f = (f>>1);
	}
}




int main()
{
	clock_t end;
	clock_t start;
	double i_time;

	start = clock();

	inference(SIGN_TRESH, X_LEN, W_LEN, Y_LEN, X, start_w, Y);

	end = clock();

	printf("Layer output: ");
	for (int i = 0; i < Y_LEN; i++)
		printf("%lx ",Y[i]);
	printf("\n");

	i_time = (end-start)/(double)CLOCKS_PER_SEC ;
	printf("inference time %lf\n",i_time);

	return 0;
}


