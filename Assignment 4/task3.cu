#include <cuda_runtime.h>       // Библиотека CUDA Runtime
#include <iostream>             // Для вывода в консоль
#include <vector>               // Для работы с массивами на CPU
#include <chrono>               // Для измерения времени

#define N 1000000               // Размер массива
#define THREADS 256             // Потоки в блоке GPU

// --- Kernel GPU: умножение элементов массива на 2 ---
__global__ void multiplyGPU(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if (idx < n)
        data[idx] *= 2;                             // Умножаем на 2
}

int main() {
    std::vector<int> h_array(N, 1);     // Исходный массив на CPU, заполнен 1
    std::vector<int> h_result(N);       // Массив для проверки CPU
    std::vector<int> h_hybrid(N);       // Массив для гибридной обработки

    // --- CPU только ---
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Старт замера времени
    for (int i = 0; i < N; i++)
        h_result[i] = h_array[i] * 2;    // Умножаем каждый элемент на 2 на CPU
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Конец замера
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU только ---
    int *d_array;
    cudaMalloc(&d_array, N * sizeof(int));                      // Выделяем память на GPU
    cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Копируем данные на GPU

    int blocks = (N + THREADS - 1) / THREADS;                   // Количество блоков
    auto gpu_start = std::chrono::high_resolution_clock::now();
    multiplyGPU<<<blocks, THREADS>>>(d_array, N);               // Запускаем ядро GPU
    cudaDeviceSynchronize();                                    // Синхронизация потоков
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::vector<int> h_gpu(N);
    cudaMemcpy(h_gpu.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost); // Копируем результат с GPU

    // --- Гибридная обработка: первая половина на CPU, вторая на GPU ---
    int mid = N / 2;

    // CPU часть гибридной обработки
    auto hybrid_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mid; i++)
        h_hybrid[i] = h_array[i] * 2;

    // GPU часть гибридной обработки
    int *d_hybrid;
    cudaMalloc(&d_hybrid, (N - mid) * sizeof(int));                        // Память для второй половины
    cudaMemcpy(d_hybrid, h_array.data() + mid, (N - mid) * sizeof(int), cudaMemcpyHostToDevice); // Копируем вторую половину
    int hybrid_blocks = ((N - mid) + THREADS - 1) / THREADS;                // Блоки для второй половины
    multiplyGPU<<<hybrid_blocks, THREADS>>>(d_hybrid, N - mid);             // GPU умножение
    cudaDeviceSynchronize();                                                // Синхронизация

    cudaMemcpy(h_hybrid.data() + mid, d_hybrid, (N - mid) * sizeof(int), cudaMemcpyDeviceToHost); // Копируем обратно
    auto hybrid_end = std::chrono::high_resolution_clock::now();
    double hybrid_time = std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start).count();

    // --- Проверка корректности ---
    bool correct_gpu = true;
    bool correct_hybrid = true;
    for (int i = 0; i < N; i++) {
        if (h_result[i] != h_gpu[i])
            correct_gpu = false;
        if (h_result[i] != h_hybrid[i])
            correct_hybrid = false;
    }

    // --- Вывод ---
    std::cout << "Assignment 4. Task 3 (Hybrid CPU+GPU)\n";
    std::cout << "Размер массива: " << N << " элементов\n\n";

    std::cout << "CPU только:\n";
    std::cout << "Время = " << cpu_time << " мс\n\n";

    std::cout << "GPU только:\n";
    std::cout << "Время = " << gpu_time << " мс\n";
    std::cout << (correct_gpu ? "Результаты корректны\n\n" : "Ошибка: результат некорректен!\n\n");

    std::cout << "Гибридная обработка (CPU+GPU):\n";
    std::cout << "Время = " << hybrid_time << " мс\n";
    std::cout << (correct_hybrid ? "Результаты корректны\n" : "Ошибка: результат некорректен!\n");

    // --- Очистка GPU памяти ---
    cudaFree(d_array);
    cudaFree(d_hybrid);

    return 0;
}
