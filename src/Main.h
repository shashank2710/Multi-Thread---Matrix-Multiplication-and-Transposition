#include <iostream>
#include <random>
#include <thread>
#include <chrono>


//Public Structs///////////////////////////////////////////////
struct Matrix
{
	float ** element;

	void initializeZero();

	void initializeRandom();

	void print();

};
