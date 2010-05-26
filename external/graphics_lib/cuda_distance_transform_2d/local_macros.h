#ifndef __LOCAL_MACROS_H__
#define __LOCAL_MACROS_H__


#define BLOCK_DIM = 16
#
// scan_best_kernel.cu (SDK 1_1)
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS 
#define ALMOST_ZERO_BANK_CONFLICTS 

#ifdef ZERO_BANK_CONFLICTS
	#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + ((index) >> (2 * LOG_NUM_BANKS)))
#else
	#ifdef ALMOST_ZERO_BANK_CONFLICTS
		#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
	#else
		#define CONFLICT_FREE_OFFSET(index) ((index))
	#endif
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

#define TMPF(index)  (tempf[(index) + CONFLICT_FREE_OFFSET(index)])
#define TMP(index)  (temp[(index) + CONFLICT_FREE_OFFSET(index)])

#endif
