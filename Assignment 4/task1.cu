#include <cuda_runtime.h>      // Подключение CUDA Runtime API
#include <iostream>            // Для ввода-вывода (cout)
#include <vector>              // Для использования std::vector
#include <chrono>              // Для измерения времени

#define N 100000               // Размер массива
#define THREADS 256            // Количество потоков в одном блоке

// CUDA-ядро: считает сумму элементов массива в глобальной памяти
__global__ void sumGlobal(int* data, int* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if (idx < n) {                                  // Проверка выхода за границы массива
        atomicAdd(result, data[idx]);               // Атомарное прибавление к общей сумме
    }
}

int main() {
    std::vector<int> h_array(N, 1);   // Хост-массив из N элементов, все равны 1
    int cpu_sum = 0;                  // Переменная для суммы на CPU

    // --- CPU ---
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Старт таймера CPU
    for (int i = 0; i < N; i++)       // Последовательный проход по массиву
        cpu_sum += h_array[i];        // Сложение элементов
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Конец таймера CPU
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // Время в мс

    // --- GPU ---
    int *d_array, *d_result;          // Указатели на память GPU
    cudaMalloc(&d_array, N * sizeof(int)); // Выделение памяти под массив на GPU
    cudaMalloc(&d_result, sizeof(int));    // Выделение памяти под результат

    cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Копирование массива на GPU
    cudaMemset(d_result, 0, sizeof(int));  // Обнуление результата на GPU

    int blocks = (N + THREADS - 1) / THREADS; // Расчёт количества блоков

    auto gpu_start = std::chrono::high_resolution_clock::now(); // Старт таймера GPU

    sumGlobal<<<blocks, THREADS>>>(d_array, d_result, N); // Запуск CUDA-ядра
    cudaDeviceSynchronize();   // Ожидание завершения всех потоков GPU

    auto gpu_end = std::chrono::high_resolution_clock::now(); // Конец таймера GPU
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count(); // Время в мс

    int gpu_sum = 0;                                      // Переменная для суммы с GPU
    cudaMemcpy(&gpu_sum, d_result, sizeof(int), cudaMemcpyDeviceToHost); // Копирование результата на CPU

    // --- Вывод ---
    std::cout << "Assignment 4. Task 1\n";
    std::cout << "Размер массива: " << N << " элементов\n\n";

    std::cout << "CPU (последовательно):\n";
    std::cout << "Сумма = " << cpu_sum << "\n";
    std::cout << "Время = " << cpu_time << " мс\n\n";

    std::cout << "GPU (глобальная память, atomicAdd):\n";
    std::cout << "Сумма = " << gpu_sum << "\n";
    std::cout << "Время = " << gpu_time << " мс\n\n";

    if (cpu_sum == gpu_sum)            // Сравнение результатов
        std::cout << "Результаты совпадают.\n";
    else
        std::cout << "Ошибка: результаты не совпадают!\n";

    cudaFree(d_array);                // Освобождение памяти массива на GPU
    cudaFree(d_result);               // Освобождение памяти результата

    return 0;                         // Завершение программы
}
