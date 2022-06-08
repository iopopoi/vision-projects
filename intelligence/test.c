#include "stdlib.h"
#include "time.h"
#include "stdio.h"
#include "string.h"
int K, D, N;
int **lines;

void free_matrix(int **matrix, int x)
{
    for (int i = 0; i < x; i++)
        free(matrix[i]);
    free(matrix);
}

int measure_distance(int *first, int *second)
{
    int distance = 0;
    for (int i = 0; i < D; ++i)
    {
        distance += (first[i] - second[i]) * (first[i] - second[i]);
    }
    return distance;
}

struct coordinate
{
    int *value;
    int cluster;
};

int main()
{
    FILE *input_file = fopen("input_file.txt", "r");
    FILE *output_file = fopen("output_file.txt", "w");
 
    fscanf(input_file, "%d %d %d", &K, &D, &N); // K개 cluster, input ( N * D )

    // 점들에 대한 정보를 lines 에 담기
    fseek(input_file, 5, SEEK_SET);

    struct coordinate *coordinates;
    coordinates = malloc(sizeof(int) * ((D+1)*N)); // 여기 공간 할당이 부족해요 바꿔줌

    for (int i = 0; i < N; i++)
    {   
        int *dimension_array = (int*)malloc(sizeof(int) * D);
        for (int j = 0; j < D; j++)
        {
            int numbers;
            fscanf(input_file, "%d", &numbers);
            dimension_array[j] = numbers;
        }
        // coordinates[i].value = dimension_array;
        memcpy(&coordinates[i].value, &dimension_array, sizeof(int)*D);
        for (int j = 0; j < D; j++)
        {
            printf("%d ", coordinates[i].value[j]);
        }
        printf("\n");
    }

    printf("\n%d %d %d\n", K, D, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            printf("[%d, %d]  ", coordinates[i].value[j],&coordinates[i].value[j]);
        }
        printf("\n");
    }
    printf("\n");

    // // k_list (승준님)
    // srand(time(NULL));
    // int **k_list;
    // k_list = malloc(sizeof(int *) * K); //check ^^7
    // for (int j = 0; j < K; ++j)
    // {
    //     k_list[j] = (int *)malloc(sizeof(int) * D); //check ^^7
    // }

    // int *k_number = (int *)malloc(sizeof(int) * (K - 1));
    // for (int i = 0; i < K - 1; i++)
    // {
    //     k_number[i] = -1;
    // }

    // int k_n = K; //반복문 제어값
    // while (k_n > 0)
    // {
    //     int comparison = rand() % N; // 랜덤값 1~N값 까지 숫자
    //     for (int i = 0; i < K - 1; i++)
    //     {
    //         if (k_number[i] == comparison)
    //         {
    //             comparison = -1;
    //             break;
    //         }
    //     }
    //     if (comparison != -1)
    //     { //중복 X
    //         for (int i = 0; i < D; ++i)
    //         {
    //             k_list[K - k_n] = coordinates[comparison].value; // 초기점 위치
    //         }
    //         k_number[K - k_n] = comparison;
    //         k_n--;
    //     }
    // }

    // printf("K = %d\n", K);
    // for (int i = 0; i < K; ++i)
    // {
    //     printf("%d ", k_number[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < K; ++i)
    // {
    //     for (int j = 0; j < D; ++j)
    //     {
    //         printf("%d ", k_list[i][j]);
    //     }
    //     printf("\n");
    // }

    // int token = 1;
    // int cnt = 1; // 시도 횟수

    // while (token)
    // {
    //     token = 0;

    //     // 거리측정 및 cluster 갱신 (갱신되지 않는다면 스탑)
    //     for (int i = 0; i < N; ++i)
    //     {
    //         // token 갱신 조건이 이상함
    //         int maximum = measure_distance(coordinates[i].value, k_list[0]);
    //         int temp_cluster = coordinates[i].cluster;
    //         for (int j = 1; j < K; ++j)
    //         {
    //             int temp_distance = measure_distance(coordinates[i].value, k_list[j]);
    //             if (temp_distance < maximum)
    //             {
    //                 coordinates[i].cluster = j;
    //                 maximum = temp_distance;
    //             }
    //             if (temp_cluster != coordinates[i].cluster)
    //             {
    //                 token = 1;
    //             }
    //         }
    //         printf("%d's cluster is %d\ndistance = %d\n", i, coordinates[i].cluster, maximum);
    //     }

    //     // file 에 작성
    //     for (int i = 0; i < K; ++i)
    //     {
    //         for (int j = 0; j < D; ++j)
    //         {
    //             fprintf(output_file, "%d ", k_list[i][j]);
    //         }
    //         fprintf(output_file, "\n");
    //     }

    //     fprintf(output_file, "itr = %d\n", cnt++);
    //     for (int i = 0; i < K; i++)
    //     {
    //         fprintf(output_file, "Cluster %d: ", i);
    //         for (int j = 0; j < N; j++)
    //         {
    //             if (i == coordinates[j].cluster)
    //             {
    //                 fprintf(output_file, "%d ", j);
    //             }
    //         }
    //         fprintf(output_file, "\n");
    //     }

    //     // 각 좌표마다 좌표값 다 더해서 평균값 계산하고 k_list 의 값들 갱신
    //     for (int i = 0; i < K; ++i)
    //     {
    //         for (int j = 0; j < D; ++j)
    //         {
    //             int count = 0;
    //             int sum = 0;
    //             for (int k = 0; k < N; ++k)
    //             {
    //                 if (coordinates[k].cluster == i)
    //                 {
    //                     count++;
    //                     sum += coordinates[k].value[j];
    //                 }
    //             }
    //             if (count == 0)
    //             {
    //                 k_list[i][j] = 0;
    //             }
    //             else
    //             {
    //                 k_list[i][j] = (int)(sum / count);
    //             }
    //         }
    //     }
    // }

    // // 메모리 해제
    // free(coordinates);
    // free_matrix(k_list, K);
    // free(k_number);
    // fclose(input_file);
    // fclose(output_file);

    return 0;
}