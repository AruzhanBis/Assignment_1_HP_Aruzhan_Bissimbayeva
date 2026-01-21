#include <mpi.h>                          // Подключаем библиотеку MPI
#include <iostream>                      // Для вывода в консоль
#include <vector>                        // Для использования std::vector
#include <numeric>                      // Для числовых операций (здесь не обязателен, но допустим)

int main(int argc, char** argv) {         // Точка входа в программу
    MPI_Init(&argc, &argv);              // Инициализация среды MPI

    int world_size;                     // Переменная для общего числа процессов
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Получаем количество процессов

    int world_rank;                     // Переменная для номера текущего процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Получаем ранг (ID) текущего процесса

    const int N = 1000000;               // Общий размер массива
    int local_n = N / world_size;        // Размер части массива для одного процесса

    std::vector<int> data;              // Вектор для хранения всего массива (только у root)
    if (world_rank == 0) {              // Если это главный процесс (ранг 0)
        data.resize(N, 1);              // Создаем массив из N элементов, заполненных единицами
    }

    std::vector<int> local_data(local_n); // Вектор для локальной части массива у каждого процесса

    // Рассылаем части массива всем процессам
    MPI_Scatter(
        data.data(),                    // Указатель на полный массив (только у root)
        local_n,                        // Сколько элементов отправлять каждому
        MPI_INT,                       // Тип данных
        local_data.data(),             // Буфер приёма локального куска
        local_n,                       // Сколько элементов принимает каждый
        MPI_INT,                       // Тип принимаемых данных
        0,                              // Ранг корневого процесса
        MPI_COMM_WORLD                 // Коммуникатор
    );

    double start_time = MPI_Wtime();     // Засекаем время начала вычислений

    long long local_sum = 0;             // Переменная для локальной суммы
    for(int i = 0; i < local_n; i++) {   // Цикл по своей части массива
        local_sum += local_data[i];     // Суммируем элементы
    }

    long long total_sum = 0;             // Переменная для общей суммы
    MPI_Reduce(
        &local_sum,                     // Адрес локальной суммы
        &total_sum,                     // Адрес общей суммы (у root)
        1,                              // Количество элементов
        MPI_LONG_LONG,                 // Тип данных
        MPI_SUM,                       // Операция суммирования
        0,                              // Корневой процесс
        MPI_COMM_WORLD                 // Коммуникатор
    );

    double end_time = MPI_Wtime();       // Засекаем время окончания вычислений

    if (world_rank == 0) {              // Только главный процесс выводит результат
        std::cout << "Процессов: " << world_size
                  << " | Сумма: " << total_sum
                  << " | Время: " << (end_time - start_time) * 1000 << " ms"
                  << std::endl;         // Печать количества процессов, суммы и времени
    }

    MPI_Finalize();                     // Завершаем работу MPI
    return 0;                           // Выход из программы
}
