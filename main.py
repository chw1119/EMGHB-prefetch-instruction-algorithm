from str_data import str_data
import numpy as np


def make_markov(str_data):

    a2a = 0
    a2b = 0
    a2c = 0

    b2a = 0
    b2b = 0
    b2c = 0

    c2a = 0
    c2b = 0
    c2c = 0

    last = None
    length = len(str_data) - 1

    for c in str_data:
        if last is None:
            last = c
            continue

        else:

            if last == 'A':
                if c == 'A':
                    a2a += 1
                elif c == 'B':
                    a2b += 1
                else:
                    a2c += 1
            
            elif last == 'B':
                if c == 'A':
                    b2a += 1
                elif c == 'B':
                    b2b += 1
                else:
                    b2c += 1
            
            else:
                if c == 'A':
                    c2a += 1
                elif c == 'B':
                    c2b += 1
                else:
                    c2c += 1

            last = c

    a2a = a2a / length
    a2b = a2b / length
    a2c = a2c / length

    b2a = b2a / length
    b2b = a2b / length
    b2c = b2c / length

    c2a =  c2a / length
    c2b =  c2b / length
    c2c =  c2c / length


    return (
        (a2a, a2b, a2c), 
        (b2a, b2b, b2c), 
        (c2a, c2b, c2c)
    )


def matrix_multiply(matrix, vector):
    """
    matrix: 3x3 마르코프 전이 행렬 (tuple of tuples)
    vector: 3x1 초기 상태 벡터 (list or tuple)
    """
    # numpy를 사용하여 행렬 곱셈 수행
    matrix_np = np.array(matrix)
    vector_np = np.array(vector)

    # 행렬 곱셈
    result = np.dot(matrix_np, vector_np)
    return result

# 마르코프 전이 행렬 생성
transition_matrix = make_markov(str_data)

# 초기 상태 벡터 (예: A, B, C 각각에 대한 확률 벡터)
initial_vector = [0.33, 0.33, 0.34]

# 행렬 곱셈 수행
result_vector = matrix_multiply(transition_matrix, initial_vector)

print("전이 행렬:")

print(np.array(transition_matrix))

print("초기 상태 벡터:", initial_vector)
print("결과 상태 벡터:", result_vector)