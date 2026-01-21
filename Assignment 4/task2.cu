#include <cuda_runtime.h>           // Библиотека CUDA Runtime для работы с GPU
#include <iostream>                 // Для вывода текста в консоль
#include <vector>                   // Для хранения массивов на CPU
#include <chrono>                   // Для измерения времени выполнения

#define N 1000000                   // Размер массива
#define BLOCK_SIZE 1024             // Размер блока потоков на GPU

// --- Kernel: блочный scan (префиксная сумма, inclusive) ---
__global__ void scanBlock(int *data, int *blockSums, int n) {
    __shared__ int sdata[BLOCK_SIZE];       // Shared memory для хранения элементов блока

    int tid = threadIdx.x;                  // Индекс потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid;// Глобальный индекс потока в массиве

    // Загружаем элементы массива в shared memory
    sdata[tid] = (gid < n) ? data[gid] : 0; 
    __syncthreads();                        // Синхронизация потоков блока

    // --- Up-sweep / reduce для вычисления префиксной суммы ---
    for (int offset = 1; offset < blockDim.x; offset *= 2) { // Проход по степеням двойки
        int temp = 0;                       // Временная переменная для суммирования
        if (tid >= offset)                  // Только потоки с tid >= offset участвуют
            temp = sdata[tid - offset];    // Берем элемент на offset слева
        __syncthreads();                    // Синхронизация перед суммированием
        sdata[tid] += temp;                 // Добавляем временное значение
        __syncthreads();                    // Синхронизация после суммирования
    }

    // Записываем результат обратно в глобальную память
    if (gid < n)
        data[gid] = sdata[tid];             // Каждому элементу массива присваиваем значение из shared memory

    // Сохраняем сумму блока для последующего добавления другим блокам
    if (tid == blockDim.x - 1)
        blockSums[blockIdx.x] = sdata[tid]; // Последний поток блока записывает сумму блока
}

// --- Kernel: добавляем суммы предыдущих блоков к каждому элементу ---
__global__ void addBlockSums(int *data, int *blockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс потока
    if (blockIdx.x == 0 || gid >= n)               // Первый блок ничего не добавляет или индекс вне массива
        return;

    data[gid] += blockSums[blockIdx.x - 1];       // Добавляем сумму предыдущего блока
}

int main() {
    std::vector<int> h_array(N, 1);        // Создаем массив на CPU, заполненный единицами
    std::vector<int> h_result(N);          // Массив для результата CPU

    // --- CPU scan (последовательно) ---
    auto cpu_start = std::chrono::high_resolution_clock::now(); // Начало замера времени CPU
    h_result[0] = h_array[0];             // Первый элемент остается без изменений
    for (int i = 1; i < N; i++)
        h_result[i] = h_result[i - 1] + h_array[i]; // Последовательное вычисление префиксной суммы
    auto cpu_end = std::chrono::high_resolution_clock::now();   // Конец замера времени CPU
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); // Вычисляем время в мс

    // --- GPU scan ---
    int *d_array, *d_blockSums;                        // Указатели на память GPU
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Количество блоков на GPU

    cudaMalloc(&d_array, N * sizeof(int));             // Выделяем память под массив на GPU
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int)); // Память для сумм блоков

    cudaMemcpy(d_array, h_array.data(), N * sizeof(int), cudaMemcpyHostToDevice); // Копируем массив на GPU

    auto gpu_start = std::chrono::high_resolution_clock::now(); // Начало замера времени GPU

    // Первый проход: scan каждого блока
    scanBlock<<<numBlocks, BLOCK_SIZE>>>(d_array, d_blockSums, N); 
    cudaDeviceSynchronize();                        // Ждем завершения всех блоков

    // Если блоков больше одного, делаем scan сумм блоков на CPU
    std::vector<int> h_blockSums(numBlocks);       // Временный массив для сумм блоков
    cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost); // Копируем суммы блоков на CPU
    for (int i = 1; i < numBlocks; i++)
        h_blockSums[i] += h_blockSums[i - 1];     // Последовательный scan по блокам
    cudaMemcpy(d_blockSums, h_blockSums.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice); // Возвращаем обновленные суммы блоков на GPU

    // Второй проход: добавляем суммы предыдущих блоков к элементам каждого блока
    addBlockSums<<<numBlocks, BLOCK_SIZE>>>(d_array, d_blockSums, N);
    cudaDeviceSynchronize();                        // Ждем завершения всех блоков

    auto gpu_end = std::chrono::high_resolution_clock::now();   // Конец замера времени GPU
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count(); // Вычисляем время в мс

    // Копируем результат обратно на CPU для проверки
    std::vector<int> h_gpuResult(N);                           
    cudaMemcpy(h_gpuResult.data(), d_array, N * sizeof(int), cudaMemcpyDeviceToHost); // Копируем массив с GPU

    // --- Проверка корректности ---
    bool correct = true;                     
    for (int i = 0; i < N; i++) {
        if (h_result[i] != h_gpuResult[i]) {   // Сравниваем CPU и GPU результаты
            correct = false;                   // Если есть несоответствие — ошибка
            break;
        }
    }

    // --- Вывод ---
    std::cout << "Assignment 4. Task 2 \n";
    std::cout << "Размер массива: " << N << " элементов\n\n";
    std::cout << "CPU:\nВремя = " << cpu_time << " мс\n\n";
    std::cout << "GPU (shared memory scan):\nВремя = " << gpu_time << " мс\n\n";

    if (correct)
        std::cout << "Результаты совпадают.\n";   // Если проверка успешна
    else
        std::cout << "Ошибка: результаты не совпадают!\n";

    // Освобождение GPU памяти
    cudaFree(d_array);
    cudaFree(d_blockSums);

    return 0;                                       // Завершение программы
}
