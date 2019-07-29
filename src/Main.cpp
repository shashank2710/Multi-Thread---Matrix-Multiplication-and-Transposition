//////////////////////////////////////////////////////////////////
 // main.cpp
 //
 //  Created on: Jun 10, 2019
 //      Author: ssatyanarayana
 //
///////////////////////////////////////////////////////////////////
// CPP Program for Matrix Multiplication and Transposition using threads
///////////////////////////////////////////////////////////////////

#include "Main.h"

//////////////////////////////////////////////////////////////////
//User-Defined Parameters
/////////////////////////////////////////////////////////////////
//Assuming Square Matrix, define Matrix Size
static const long MATRIX_SIZE = 4;
//Define Maximum Number of Threads Allowed
static const int MAX_THREADS = 4;
//Define the Range of Numbers in the Matrix
static const long LOW_NUM = 0;
static const long HIGH_NUM = 100;
//Pre-Processor to time each execution
#define TIME_EXECUTION
//Pre-Processor to Display the Output
#define PRINT_OUTPUT
/////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////
//Function Definitions
/////////////////////////////////////////////////////////////////
void singleThreadExecution();
void multiThreadExecution();
void multiplyThread(Matrix& R1, const int thread_number, const Matrix& M1, const Matrix& M2);
void transposeMatrix();
void transposeThread(Matrix& transposeMatrix, const int thread_number, const Matrix& M1);
clock_t startTime, endTime;
/////////////////////////////////////////////////////////////////

int main()
{
	std::cout<<"Single Execution"<<std::endl;
	singleThreadExecution();
	std::cout<<"Multi-Thread Execution"<<std::endl;
	multiThreadExecution();
	std::cout<<"Transpose of Matrix"<<std::endl;
	transposeMatrix();
	std::cout<<"End of Program"<<std::endl;
}


//////////////////////////////////////////////////////////////////
//Function to perform Single-thread Matrix Multiplication
/////////////////////////////////////////////////////////////////
void singleThreadExecution()
{
	startTime = clock();
	//Declare and Initialize the Matrix
	Matrix A, B, Result;
	A.initializeRandom();
	B.initializeRandom();
	Result.initializeZero();

	for (int i = 0; i < MATRIX_SIZE; ++i) {
	    for (int j = 0; j < MATRIX_SIZE; ++j) {
	      float result = 0.0f;
	      for (int k = 0; k < MATRIX_SIZE; ++k) {
	        const float e1 = A.element[i][k];
	        const float e2 = B.element[k][j];
	        result += e1 * e2;
	      }
	      Result.element[i][j] = result;
	    }
	  }
	endTime = clock();
#ifdef TIME_EXECUTION
	std::cout<<"Execution Time:"<<double(endTime - startTime)<<std::endl;
#endif
#ifdef PRINT_OUTPUT
	std::cout<<"Matrix A"<<std::endl;
	A.print();
	std::cout<<"Matrix B"<<std::endl;
	B.print();
	std::cout<<"Matrix AxB"<<std::endl;
	Result.print();
#endif
}


//////////////////////////////////////////////////////////////////
//Function to perform Multi-thread Matrix Multiplication
/////////////////////////////////////////////////////////////////
void multiThreadExecution()
{
	startTime = clock();
	//Declare and Initialize Matrix
	Matrix A, B, Result;
	A.initializeRandom();
	B.initializeRandom();
	Result.initializeZero();

	//Initialize Thread
	std::thread threads[MAX_THREADS];

	//Evaluate Each Thread
	for (int i =0; i < MAX_THREADS; i++)
	{
		threads[i] = std::thread(multiplyThread, std::ref(Result), i, std::ref(A), std::ref(B));
	}

	for (int i = 0; i < MAX_THREADS ; i++)
	{
		threads[i].join();
	}
	endTime = clock();
#ifdef TIME_EXECUTION
	std::cout<<"Execution Time:"<<double(endTime - startTime)<<std::endl;
#endif
#ifdef PRINT_OUTPUT
	std::cout<<"Matrix A"<<std::endl;
	A.print();
	std::cout<<"Matrix B"<<std::endl;
	B.print();
	std::cout<<"Matrix AxB"<<std::endl;
	Result.print();
#endif
}

//////////////////////////////////////////////////////////////////
//Function to perform Matrix Multiplication on each thread
/////////////////////////////////////////////////////////////////
void multiplyThread(Matrix& R1, const int thread_number, const Matrix& M1, const Matrix& M2)
{
	int totalElements = (MATRIX_SIZE * MATRIX_SIZE);
	int operationsPerThread = totalElements / MAX_THREADS;
	int extraOperations = totalElements % MAX_THREADS;

	int startOps, endOps;

	  //Last Thread takes on all extra operations
	  if (thread_number == (thread_number-1))
	  {
	    startOps = operationsPerThread * thread_number;
	    endOps = operationsPerThread * (thread_number + 1) + extraOperations;
	  }
	  else {
		  startOps = operationsPerThread * thread_number;
		  endOps = operationsPerThread * (thread_number + 1);
	  }

	  for (int op = startOps; op < endOps; op++)
	  {
	    const int row = op % MATRIX_SIZE;
	    const int col = op / MATRIX_SIZE;
	    float r = 0.0f;
	    for (int i = 0; i < MATRIX_SIZE; i++)
	    {
	      float e1 = M1.element[row][i];
	      float e2 = M2.element[i][col];
	      r += e1 * e2;
	    }

	    R1.element[row][col] = r;
	  }

}

//////////////////////////////////////////////////////////////////
//Function to perform Transpose of Matrix Multiplication
/////////////////////////////////////////////////////////////////
void transposeMatrix()
{
	startTime = clock();
	Matrix transposeMatrix,inputMatrix;
	transposeMatrix.initializeZero();
	inputMatrix.initializeRandom();
	//Initialize Thread
	std::thread threads[MAX_THREADS];
	for (int i =0; i < MAX_THREADS; i++)
	{
		threads[i] = std::thread(transposeThread, std::ref(transposeMatrix), i, std::ref(inputMatrix));
	}

	for (int i = 0; i < MAX_THREADS ; i++)
	{
		threads[i].join();
	}
	endTime = clock();
#ifdef TIME_EXECUTION
		std::cout<<"Execution Time:"<<double(endTime - startTime)<<std::endl;
#endif
#ifdef PRINT_OUTPUT
	std::cout<<"Input Matrix"<<std::endl;
	inputMatrix.print();
	std::cout<<"Transpose Matrix"<<std::endl;
	transposeMatrix.print();
#endif
}


//////////////////////////////////////////////////////////////////
//Function to perform Matrix Transposition on each thread
/////////////////////////////////////////////////////////////////
void transposeThread(Matrix& transposeMatrix, const int thread_number, const Matrix& M1)
{
	int totalElements = MATRIX_SIZE;
	int operationsPerThread = totalElements / MAX_THREADS;
	int extraOperations = totalElements % MAX_THREADS;

	int startOps, endOps;

	//Last Thread takes on all extra operations
	if (thread_number == (thread_number-1))
	{
		startOps = operationsPerThread * thread_number;
		endOps = operationsPerThread * (thread_number + 1) + extraOperations;
	}
	else
	{
		startOps = operationsPerThread * thread_number;
		endOps = operationsPerThread * (thread_number + 1);
	}

	for (int op = startOps; op < endOps; op++)
	{
		for (int i = 0; i < MATRIX_SIZE; ++i)
		{
			transposeMatrix.element[i][op] = M1.element[op][i];
		}
	}

}


void Matrix::initializeZero()
{
	element = new float*[MATRIX_SIZE];
	for (int i = 0; i< MATRIX_SIZE; i++)
	{
		element[i] = new float[MATRIX_SIZE];
		for (int j = 0; j < MATRIX_SIZE; j++)
		{
			element[i][j] = 0.0f;
		}
	}

}

void Matrix::initializeRandom()
{
	element = new float*[MATRIX_SIZE];
	for (int i = 0; i < MATRIX_SIZE; i++ )
	{
		element[i] = new float[MATRIX_SIZE];
		for (int j = 0; j < MATRIX_SIZE ; j++)
		{
			element[i][j] = LOW_NUM + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HIGH_NUM-LOW_NUM)));
		}

	}

}

void Matrix::print()
{
	std::cout << std::endl;
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		std::cout << "|\t";

		for (int j = 0; j < MATRIX_SIZE; ++j) {
			std::cout << element[i][j] << "\t";
		}
		std::cout << "|" << std::endl;
	}
}




