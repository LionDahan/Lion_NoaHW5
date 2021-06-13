#Lion Dahan 318873338
#Noa Ben Gigi 318355633 

import math
def linear_interpolation(points_table,point):
    x1 = math.ceil(point)#if point is 2.5 its will up the number to 3
    x2 = x1-1# then we find the number who is smallwer than 2.5
    y1 = points_table[x1][1]#brings the y from the table
    y2 =points_table[x2][1]
    return calculate_linear_inter(x1,x2,y1,y2,point)


def get_unit_matrix(size):
    """
    :param size: integer that described the matrix's size- the matrix is n*n size
    :return: the unit matrix in the right size
    """
    unit_mat = [[0] * size for _ in range(size)]
    for i in range(size):
        unit_mat[i][i] = 1
    return unit_mat


def replace_line_in_matrix(mat, i):
    """ if the pivot is zero than we replace lines
    :param mat: matrix
    :param i: index
    :return: the updated mat
    """
    unit_mat = get_unit_matrix(len(mat))
    max_value_index = 0
    check = False
    for j in range(i + 1, len(mat)):
        if abs(mat[j][i]) > abs(mat[i][i]):
            max_value_index = j
            check = True
    if check:
        temp = unit_mat[i]
        unit_mat[i] = unit_mat[max_value_index]
        unit_mat[max_value_index] = temp
    return unit_mat


def Polynomial_interpolation(points_table,point):
    new_matrix=list(range(len(points_table)))
    for i in range (len(new_matrix)):
        new_matrix[i]=list(range(len(new_matrix)))
    for i in range (len(points_table)):
        new_matrix[i][0]=1
    for i in range(len(points_table)):
        for j in range(1, len(points_table)):
            new_matrix[i][j] = points_table[i][0] ** j
    #creates the y's vector
    y_vector = []
    for i in range(len(points_table)):
        y_vector.append([])
        y_vector[i].append(points_table[i][1])

    result = multiply_matrices(inverse_mat(new_matrix), y_vector)
    sum = result[0][0]
    for i in range(1, len(result)):
        sum = sum + (result[i][0] * point ** i)
    return sum


def multiply_matrices(mat1, mat2):
    """
    :param mat1: the first matrix
    :param mat2: the second matrix
    :return: multiply between matrices
    """
    if len(mat1[0]) != len(mat2):
        return None
    result_mat = [[0] * len(mat2[0]) for _ in range(len(mat1))]
    for i in range(len(mat1)):  # rows
        for j in range(len(mat2[0])):  # cols
            for k in range(len(mat2)):
                result_mat[i][j] += (mat1[i][k] * mat2[k][j])
    return result_mat


def compare_two_matrices(mat1, mat2):
    """
    :param mat1: matrix1
    :param mat2: matrix2
    :return: boolean value - true is mat1==mat2. else , false
    """
    if mat1 and mat2 is not None:
        for i in range(len(mat1)):
            for j in range(len(mat2)):
                if mat1[i][j] != mat2[i][j]:
                    return False
        return True
    else:
        return False


def find_elementary_matrix(mat):
    """this func find the elementary matrix in any level in order to find the reverse matrix
    :param mat: matrix
    :return: elementary matrix
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat


def elementary_matrix_U(mat):
    """
    this func find the elementary matrix in any level to find the reverse matrix
    :param mat:
    :return:
    """
    elementary_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i][j] != 0:
                elementary_mat[i][j] = - mat[i][j] / mat[j][j]
                return elementary_mat

def unit_diagonal(mat):
    unit_mat = get_unit_matrix(len(mat))
    for i in range(len(mat)):
        if mat[i][i] != 1:
            unit_mat[i][i] = 1 / mat[i][i]
            return unit_mat
    return unit_mat


def inverse_mat(mat):
    """ return the inverse mat with Gauss Elimination
    :param mat:matrix in size n*n
    :return: inverse matrix
    """
    unit_mat = get_unit_matrix(len(mat))  # build unit matrix
    all_elementary_mat = unit_mat  # deep copy
    for i in range(len(mat)):
        u_mat = replace_line_in_matrix(mat, i)  # pivoting
        mat = multiply_matrices(u_mat, mat)
        all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
        for j in range(i, len(mat)):
            if u_mat is not None or compare_two_matrices(u_mat, unit_mat):
                mat = multiply_matrices(u_mat, mat)
                all_elementary_mat = multiply_matrices(u_mat, all_elementary_mat)
            u_mat = find_elementary_matrix(mat)
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            l_mat = elementary_matrix_U(mat)
            if l_mat is not None:
                mat = multiply_matrices(l_mat, mat)
                all_elementary_mat = multiply_matrices(l_mat, all_elementary_mat)
    for i in range(len(mat)):
        if mat[i][i] != 1:
            diagonal_mat = unit_diagonal(mat)
            mat = multiply_matrices(diagonal_mat, mat)
            all_elementary_mat = multiply_matrices(diagonal_mat, all_elementary_mat)
    return all_elementary_mat




def Neville_interpolation(points_table,point):

    for j in range(1,len(points_table)):
        for i in range(len(points_table)-1,j-1,-1):
            points_table[i][1]=((point-points_table[i-j][0])*points_table[i][1] - (point-points_table[i][0])*points_table[i-1][1] )/ (points_table[i][0]-points_table[i-j][0])
    sum = points_table[len(points_table)-1][1]
    return sum




def lagrange_interpolation(points_table,point):
    result = 0
    for i in range(len(points_table)):
        mini_result=1
        for j in range(len(points_table)):
            if i != j:
               mini_result = mini_result * ((point-points_table[j][0]) / (points_table[i][0] - points_table[j][0]))
        result =result+mini_result*points_table[i][1]
    return result



def calculate_linear_inter(x1,x2,y1,y2,point):
    return ((y1-y2)/(x1-x2))*point +(y2*x1)- (y1*x2)/(x1-x2)

def main():
    point = 2.5
    points_table = [[0, 0], [1, 0.8415], [2, 0.9093], [3, 0.1411], [4, -0.7568], [5, -0.9589], [6, -0.2794]]
    print("Hello. In which way would you like to found the value of the point? point=",point)
    n = int(input("1= Linear Interpolation\n2=Polynomial Interpolation\n3=Lagrange Interpolation\n4=Neville Interpolation"))
    if n==1:
        print("Linear Interpolation: f(x)=", linear_interpolation(points_table,point))
        main()
    elif n==2:
        print("Polynomial Interpolation: f(x)=", Polynomial_interpolation(points_table, point))
        main()
    elif n==3:
        print("Lagrange Interpolation: f(x)=", lagrange_interpolation(points_table, point))
        main()
    elif n==4:
        print("Neville Interpolation: f(x)=", Neville_interpolation(points_table, point))
        main()
    else:
        print("Error.Please choose again")
        main()


    #points_table2 =[[1,1],[2,0],[4,1.5]]
    #points = [[1, 0.7651], [1.3, 0.62], [1.6, 0.4554], [1.9, 0.2818], [2.2, 0.1103]]
    #points3= [[1,0],[1.2,0.1124],[1.3,0.16799],[1.4,0.222709]]
    #point5 = 1.28
    #pointTT=[[1,0.8415],[2,0.9093],[3,0.1411]]
    #point4= 2.5




if __name__ == "__main__":
    main()
