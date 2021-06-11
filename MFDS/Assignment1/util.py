
import pandas as pd
import numpy as np 
from scipy.linalg import solve
from scipy.sparse import dia_matrix,diags
from scipy.linalg import lu
import math

def gauss_siedel(A, B, x, n):
    L=np.tril(A)
    U=A-L
    append_val = []
    for i in range(n):
        x = np.dot(np.linalg.inv(L), B - np.dot(U, x))
        print(x)
        append_val.append(x)
    all_val = create_iteration_df(append_val)
    return x,all_val

def gauss_jacobi(A, B, x, n):
    D=diags(A.diagonal() ).toarray()
    R = A-D
    append_val = []
    for i in range(n):
        x = np.dot(np.linalg.inv(D), B - np.dot(R, x))
        print(x)
        append_val.append(x)
    all_val = create_iteration_df(append_val)
    return x,all_val

def gauss_jacobi_earlystopping(A, B, x, early_stopping_per,max_iteration ):
    D=diags(A.diagonal() ).toarray()
    R = A-D
    append_val = []
    ANS = solve(A, B)
    esc = 1
    i = 0
    while esc != 0:
        x = np.dot(np.linalg.inv(D), B - np.dot(R, x))
        append_val.append(x)
        esc = np.sum(np.abs(((ANS-x)/ANS)*100)>early_stopping_per)
        i = i + 1
        if i == max_iteration:
            break
    print(f"Iterations completed {i}, Early stopping value {esc}")
    all_val = create_iteration_df(append_val)
    return x,all_val

def gauss_siedel_earlystopping(A, B, x, early_stopping_per,max_iteration):
    L=np.tril(A)
    U=A-L
    append_val = []
    ANS = solve(A, B)
    esc = 1
    i = 0
    while esc != 0:
        x = np.dot(np.linalg.inv(L), B - np.dot(U, x))
        append_val.append(x)
        esc = np.sum(np.abs(((ANS-x)/ANS)*100)>early_stopping_per)
        i = i + 1
        if i == max_iteration:
            break
    print(f"Iterations completed {i}, Early stopping value {esc}")
    all_val = create_iteration_df(append_val)
    return x,all_val


def create_iteration_df(append_val):
    all_val = pd.DataFrame(append_val)
    all_val.columns = [f'x{col}' for col in all_val.columns.tolist()]
    all_val.index = all_val.index+1
    return all_val.T

def solve_equations(A,B, n=100,method='simple',if_earlystopping=True,early_stopping_per=1):
    '''
    This Function solves system of matrices
    '''
    x = np.random.rand(len(A))
    print(f'''\n
    Input A is 
            {A}

    Input B is 
            {B}

    Input x is 
            {x}
    
    ''')
    if method == 'simple':
        return solve(A, B),None
    elif method == 'jacobi':
        if if_earlystopping:
            return gauss_jacobi_earlystopping(A, B, x, early_stopping_per,n)
        else:
            return gauss_jacobi(A, B, x, n)

    elif method == 'siedel':
        if if_earlystopping:
            return gauss_siedel_earlystopping(A, B, x, early_stopping_per,n)
        else:
            return gauss_siedel(A, B, x, n)
    else:
        print("Wrong method is mentioned: valid methods are simple,siedel and jacobi")
        print("Now solving using simple method")
        return solve(A, B),None


def swap_rows(arr):
    '''
    This Function perform row swap operation and attempt to convert any matrix to its diagonally significant form
    '''
    new_arr = [[] for val in range(len(arr))]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            val = sum([np.abs(v) for v in arr[i]]) - np.abs(arr[i][j])
            val = val - np.abs(arr[i][j])
            if val < 0:
                new_arr[j] = arr[i]
                break
    resultant = [col for col in arr if col not in new_arr ]
    for v in range(len(new_arr)):
        if len(new_arr[v]) ==0:
            for val1 in resultant:
                new_arr[v] = val1
                resultant.remove(val1)
    return new_arr


def perform_row_operations(arr,indexval,multipliers=[1,2,3,4,5,6,7,8,9,10]):
    '''
    This Functions performs row level operation and attempts to convert any matrix to its diagonally significant form
    '''
    for val in indexval:
        doneindicator = []
        rest_index = list(range(len(arr)))
        rest_index = rest_index[rest_index.index(val):] + rest_index[:rest_index.index(val)]
        rest_index = [ri for ri in rest_index if ri != val]
        restvaldoneind = False
        for restval in rest_index:
            if not restvaldoneind:
                multiplier_lvl1_done = False
                for multiplier_lvl1 in multipliers:
                    new_row_val = arr[val]
                    if arr[val][val] <0:
                            new_row_val = [-v for v in arr[val]]
                    new_row_val = [v*multiplier_lvl1 for v in new_row_val]
                    if multiplier_lvl1_done:
                        break
                    for multiplier in multipliers:
                        next_row_val = arr[restval]
                        if new_row_val[val]==0:
                            print("This array cannot be reduced to diagonally significant form, so returning None")
                            return None
                        if next_row_val[val]==0:
                            break
                        other_val_sum = sum([np.abs(v) for v in arr[val]]) - np.abs(arr[val][val])

                        if arr[restval][val] <0:
                            next_row_val = [-v for v in arr[restval]]

                        next_row_val = [v*multiplier for v in next_row_val]
                        tmpval = list(np.array(new_row_val) + np.array(next_row_val) )
                        other_val_sum = sum([np.abs(v) for v in tmpval]) - np.abs(tmpval[val])
                        if np.abs(tmpval[val]) <= other_val_sum:
                            continue
                        else:
                            arr[val] = tmpval
                            doneindicator.append(1)
                            restvaldoneind = True
                            multiplier_lvl1_done = True
                            break
                
        if sum(doneindicator) != len(indexval):
            print("This array cannot be reduced to diagonally significant form, so returning None")
            return None
        else:
            return arr


def to_diag_sig_matrix(arr,indexval,multipliers=[1,2,3,4,5,6,7,8,9,10]):
    arr = swap_rows(arr)
    arr = perform_row_operations(arr,indexval,multipliers)
    return arr

def create_matrix(row_num,col_num,precision=4):
    return np.around(np.random.randn(row_num,col_num),precision)

def check_convergence(A,col_num,method = 'siedel'):
    P,L,U= lu(A)
    D = np.diag(np.abs(A))
    S = np.sum(np.abs(A), axis=1) - D 
    if method == 'siedel':
        C = np.dot(-np.linalg.inv(np.identity(col_num)+L),U)
    else:
        C = np.dot(-np.linalg.inv(np.identity(col_num)),L+U)
    if np.all(D > S) and np.linalg.norm(C,'fro') < 1:
        return True
    else:
        return False

def check_non_convergence(A,col_num,method = 'siedel'):
    P,L,U= lu(A)
    D = np.diag(np.abs(A))
    S = np.sum(np.abs(A), axis=1) - D 
    if method == 'siedel':
        C = np.dot(-np.linalg.inv(np.identity(col_num)+L),U)
    else:
        C = np.dot(-np.linalg.inv(np.identity(col_num)),L+U)
    if np.all(D < S) and np.linalg.norm(C,'fro') > 1:
        return True
    else:
        return False

def get_ds_matrix(row_num,col_num,precision=4,method = 'siedel',iterations = 10000):
    conv = False
    for i in range(iterations):
        A = create_matrix(row_num,col_num,precision)
        conv = check_convergence(A,col_num,method)
        if conv:
            break
    if conv == False:
        print("No convergence achieved, returning NONE")
        A = None
        B = None
    else:
        B = np.random.rand(len(A))
    return A,B

def get_nonds_matrix(row_num,col_num,precision=4,method = 'siedel',iterations = 10000):
    conv = False
    for i in range(iterations):
        A = create_matrix(row_num,col_num,precision)
        conv = check_non_convergence(A,col_num,method)
        if conv:
            break
    if conv == False:
        print("Unable to create non diagonally significant matrix, returning NONE")
        A = None
        B = None
    else:
        B = np.random.rand(len(A))
    return A,B

def convertToSig(a_number,significant_digits):
    if(type(a_number) == np.ndarray):
        for idx,i in  np.ndenumerate(a_number):
            if(a_number[idx]!=0):
                a_number[idx] = round(a_number[idx], significant_digits - int(math.floor(math.log10(abs(a_number[idx])))) - 1)
        return(a_number)    
    elif(a_number != 0.):
        rounded_number =  round(a_number, significant_digits - int(math.floor(math.log10(abs(a_number)))) - 1)
        return(rounded_number)
    else:
        return(a_number)

def gauss_elimination_pivot(A, f,digits):
    A=np.around(A,digits)
    f=np.around(f,digits)
    print("Executing 2x2 system of equations with Pivot for ",digits," significant digits!!\n\n")

    print("Random 2x2 Coefficient Matrix A of ",digits," significant digits:\n",A)
    print("Random Vector Matrix B of ",digits," significant digits:\n",f)
   
    print("Equations which must be solved based on above matrix:")
    for i in range(A.shape[0]):
        row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
        print("[{}] = [{}]".format(" + ".join(row), f[i]))
 #Caluculating the LU using Pivoting
    n = len(f)
    for i in range(0,n-1):     # Step - Looping through the columns of the matrix
        if np.abs(A[i,i])==0:
            for k in range(i+1,n):
                if np.abs(A[k,i])>np.abs(A[i,i]):
                    A[[i,k]]=A[[k,i]]             # Step Swapping ith and kth rows to each other
                    f[[i,k]]=f[[k,i]]
                    break
        for j in range(i+1,n):     # Loop through rows below diagonal for each column
            m = convertToSig(A[j,i]/A[i,i],digits)
            A[j,:] = convertToSig(A[j,:] - convertToSig(m*A[i,:],digits),digits)
            f[j] = convertToSig(f[j] - convertToSig(m*f[i],digits),digits)        
 #Caluculating the final output
    n = f.size
    x = np.zeros(n)             # Initialize the solution vector, x, to zero
    x[n-1] = convertToSig(f[n-1]/A[n-1,n-1],digits)    # Solve for last entry first
    for i in range(n-2,-1,-1):      # Loop from the end to the beginning
        sum_ = 0
        for j in range(i+1,n):        # For known x values, sum and move to rhs
            sum_ = convertToSig(sum_ + A[i,j]*x[j],digits)
        sum_ = convertToSig(sum_,digits)
        x[i] = convertToSig((f[i] - sum_)/A[i,i],digits)
    print("Solution is as below:")
    return x

def gauss_elimination_without_pivot(A, f,digits):
    print("Executing 2x2 system of equations without Pivot for ",digits," significant digits!!\n")
    A = np.around(A,digits)
    f = np.around(f,digits)
    print("Random 2x2 Coefficient Matrix A of ",digits," significant digits:\n",A)
    print("Random Vector Matrix B of ",digits," significant digits:\n",f)
    
    print("Equations which must be solved based on above matrix:")
    for i in range(A.shape[0]):
        row = ["{}*x{}".format(A[i, j], j + 1) 
        for j in range(A.shape[1])]
        print("[{}] = [{}]".format(" + ".join(row), f[i]))
    #LU Decomposition with no pivoting
    n = len(f)
    for i in range(0,n-1):     # Loop through the columns of the matrix
        for j in range(i+1,n):     # Loop through rows below diagonal for each column
            if A[i,i] == 0:
                print("Error: Zero on diagonal!")
                print("Need algorithm with pivoting")
                break
            m = convertToSig(A[j,i]/A[i,i],digits)
            A[j,:] = convertToSig(A[j,:] - convertToSig(m*A[i,:],digits),digits)
            f[j] = convertToSig(f[j] - convertToSig(m*f[i],digits),digits)
    #Using Back Substitution
    n = f.size
    x = np.zeros(n)             # Initialize the solution vector, x, to zero
    x[n-1] = convertToSig(f[n-1]/A[n-1,n-1],digits)   # Solve for last entry first
    for i in range(n-2,-1,-1):      # Loop from the end to the beginning
        sum_ = 0
        for j in range(i+1,n):        # For known x values, sum and move to rhs
            sum_ =convertToSig(sum_ + convertToSig(A[i,j]*x[j],digits),digits)
        x[i] = convertToSig(((f[i] - sum_)/A[i,i]),digits)
    print("\nSolution is as below:")
    return x