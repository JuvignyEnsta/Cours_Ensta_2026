#include <array>
#include <cstddef>
#include <iostream>
#include <omp.h> // The OpenMP header

int main() {
    // This directive creates the team of threads
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        std::cout << "Hello from thread " << ID << '\n';
    } 
    // Implicit barrier here: all threads must finish before moving on
    double begin = omp_get_wtime();
    std::array<size_t, 10000> a;
    #pragma omp parallel for collapse(2)
    for(size_t i = 0; i < a.size(); i++){
        a[i] = i + ((i+1) % 79);
        for(size_t j = 0; j < i; j ++){
            a[i] += 1;
        }
    }
    double end = omp_get_wtime();
    std::cout << "test took " << end-begin << " seconds\n";
    return 0;
}